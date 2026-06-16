// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Top-level superfile builder.
//!
//! **Naming convention.** `SuperfileBuilder` is a single-shot
//! factory — `new → add_batch×N → finish(self) → Vec<u8>`,
//! consumes self, produces one immutable artifact. Contrast
//! [`crate::supertable::SupertableWriter`], which is a long-lived
//! append handle (`append×N → commit`, repeated). The supertable
//! writer internally constructs many superfile builders, one per
//! shard per commit.
//!
//! `SuperfileBuilder` accepts user rows (Arrow batches + per-column
//! vector slices), routes FTS-text columns into a unified `FtsBuilder`,
//! routes vectors into a unified `VectorBuilder`, accumulates the
//! Parquet-bound rows, and on `finish()` produces a single byte buffer
//! that is a valid Parquet file with embedded BM25 + vector blobs
//! between the last row group and a rewritten footer carrying `inf.*`
//! KV metadata pointers.
//!
//! ## Row storage: `Vec<RecordBatch>`
//!
//! Accumulated rows are held as `Vec<RecordBatch>` rather than as
//! per-column Arrow `ArrayBuilder`s. Why:
//!
//!   1. The natural calling pattern at scale is "I already have a
//!      `RecordBatch`" — readers materialize batches, ETL pipelines
//!      build them. Accepting batches end-to-end avoids forcing
//!      callers to decompose into per-column scalars.
//!   2. `add_batch` becomes a zero-copy push: Arrow column buffers
//!      are reference-counted, so we `Arc::clone` the columns
//!      instead of memcpy-ing into builders. O(num_columns) atomic
//!      increments per batch, independent of row count or column
//!      width.
//!   3. Per-column `Box<dyn ArrayBuilder>` would require a typed
//!      downcast per cell on append — a `DataType` match statement
//!      we'd have to maintain as Arrow grows types (decimals,
//!      dictionaries, lists, structs, …).
//!   4. `ArrowWriter::write` takes `RecordBatch` directly, so
//!      `finish()` just iterates and forwards — no intermediate
//!      "drain builders into one big RecordBatch" step.
//!
//! Tradeoff: we hold strong `Arc` references to the caller's column
//! buffers until `finish()`. Callers who hand us a batch can't drop
//! it to reclaim memory mid-build; they share the buffer with us
//! until the build completes. For batch-ETL this is invisible (the
//! caller hands off and forgets); for streaming-with-backpressure it
//! could matter. There is no `add_row(scalars, vectors)` API today
//! — row-at-a-time callers must construct 1-row `RecordBatch`es
//! themselves. A typed `add_row(&[ScalarValue], ...)` helper can be
//! added later if profiling shows row-at-a-time callers need it.
//!
//! ## Tokenizer scope: one shared instance
//!
//! `BuilderOptions` carries a single `tokenizer: Option<Arc<dyn
//! Tokenizer>>` used for every FTS column. `FtsConfig` carries only
//! the column name. Why:
//!
//!   1. There is one tokenizer implementation today
//!      (`AsciiLowerTokenizer`); per-column variation has no caller.
//!   2. The underlying `FtsBuilder` takes one tokenizer for the
//!      whole index. Threading per-column tokenizers through it
//!      without inner refactor leaves only awkward options
//!      (silently use the first column's tokenizer; `Arc::ptr_eq`
//!      validate that all columns share an instance; or extend
//!      `FtsBuilder` to hold `Vec<Arc<dyn Tokenizer>>` indexed by
//!      column_id and dispatch per (col, doc) pair).
//!   3. The third is the right shape when we ship a second tokenizer
//!      — but it's a real interior refactor across `FtsBuilder`,
//!      `FtsReader`, and the `inf.fts.columns` JSON, and there is no
//!      caller asking for it.
//!
//! Forward-compat: when a second tokenizer ships (Unicode segmenter,
//! language-specific stemmers, …), `FtsConfig` grows a `tokenizer`
//! field, `BuilderOptions.tokenizer` becomes a per-column override
//! or is removed, and `FtsBuilder::new` becomes
//! `FtsBuilder::with_tokenizers(Vec<Arc<dyn Tokenizer>>)`. The
//! `inf.fts.columns` JSON already carries a `"tokenizer"` field on
//! each entry (currently always `"ascii_lower"`), so the on-disk
//! format is forward-compatible without a file rewrite.
use crate::superfile::format::footer::{encode_parquet_body, splice_index_blobs};
use crate::superfile::format::{self, kv};
use crate::superfile::fts::builder::FtsBuilder;
use crate::superfile::fts::tokenize::{AsciiLowerTokenizer, Tokenizer};
use crate::superfile::stats::SuperfileStats;
use crate::superfile::vector::builder::VectorBuilder;
use crate::superfile::vector::reader::ColumnReader;
use crate::superfile::{BuildError, SuperfileReader};
// `VectorConfig` lives in `vector::builder` and is re-exported below
// (single source of truth post-collapse — see the `pub use` line).
pub use crate::superfile::vector::builder::VectorConfig;
use crate::superfile::vector::distance::Metric;
use arrow_array::{Array, RecordBatch};
use arrow_schema::{DataType, Schema};
use parquet::basic::Compression;
use roaring::RoaringBitmap;
use std::sync::Arc;

/// Per-column FTS configuration. The `column` must exist in
/// `BuilderOptions.schema` and be `LargeUtf8`.
#[derive(Clone)]
pub struct FtsConfig {
    pub column: String,
}

// `VectorConfig` (the per-column vector config used by
// `BuilderOptions.vector_columns`) lives in
// `crate::superfile::vector::builder` and is re-exported at this
// module path above. Single source of truth — there's no outer
// wrapper struct.

/// All knobs needed to build a superfile.
#[derive(Clone)]
pub struct BuilderOptions {
    /// Arrow schema. Must contain `id_column` (typed
    /// `Decimal128(38, 0)`) and every FTS column listed in
    /// `fts_columns` (typed `LargeUtf8`).
    ///
    /// **Layering note.** When `SuperfileBuilder` is driven
    /// from the supertable, the schema passed here is the
    /// supertable's *effective* schema — the user's schema
    /// with the id column prepended. The supertable hides
    /// the id column from its public API surface;
    /// `SuperfileBuilder` sees it as a normal required field
    /// because the format spec carries primary keys in the
    /// Parquet body alongside scalar data.
    pub schema: Arc<Schema>,
    /// Name of the primary-key column in `schema`. Must be
    /// `Decimal128(38, 0)`.
    pub id_column: String,
    /// FTS columns. Each `column` must exist in `schema` as
    /// `LargeUtf8`; the same field stays in the Parquet body
    /// (readable via SQL `SELECT title …` / scalar
    /// predicates like `WHERE title LIKE …`) AND is indexed
    /// into the embedded FTS blob for BM25 ranking
    /// (`bm25_search(column, …)`). Storage cost is mild
    /// double-storage: raw text in Parquet plus the FST +
    /// PFOR-delta posting structures in the FTS blob, which
    /// dedupe terms.
    ///
    /// Contrast with [`Self::vector_columns`]: vector
    /// columns leave the Parquet body (stripped by the
    /// supertable's `vector_split` at commit time) and live
    /// only in the embedded vector blob, so they are
    /// invisible to SQL.
    ///
    /// May be empty.
    pub fts_columns: Vec<FtsConfig>,
    /// Vector columns. `column` must NOT collide with a
    /// column in `schema`, and must be unique across both
    /// `fts_columns` and `vector_columns`. May be empty.
    ///
    /// At this layer (superfile), a vector "column" is a
    /// **logical name only** — the f32 slices are passed
    /// separately to `add_batch(scalar_batch, &[&[f32]])` and
    /// the name lives in `inf.vec.columns` KV metadata, not
    /// in the Parquet schema. The "must NOT collide with a
    /// column in `schema`" rule is the format-layer
    /// disambiguation that keeps vector names out of the
    /// Parquet column namespace.
    ///
    /// At the supertable layer the constraint reads
    /// differently: there, vector columns ARE schema fields
    /// (typed `FixedSizeList<Float32, dim>`). The supertable's
    /// `vector_split` strips them at commit time and forwards
    /// `(scalar_only_batch, &[&[f32]])` down to this builder
    /// — so by the time a `BuilderOptions` reaches us, the
    /// vector names have already left the scalar schema. The
    /// supertable enforces the same cross-list uniqueness
    /// against its FTS columns at construction.
    ///
    /// To run both FTS and vector against the same business
    /// concept (e.g. semantic + lexical "description"
    /// search), model it as **two columns** — one
    /// `LargeUtf8` for the text and one `FixedSizeList<f32>`
    /// for the externally-computed embedding. Hybrid retrieval
    /// fuses results from `bm25_search(text_col, ...)` and
    /// `vector_search(emb_col, ...)`.
    pub vector_columns: Vec<VectorConfig>,
    /// Shared tokenizer for all FTS columns. Required iff
    /// `fts_columns` is non-empty.
    pub tokenizer: Option<Arc<dyn Tokenizer>>,
    /// Parquet target row-group size (number of rows).
    pub row_group_size: usize,
    /// Parquet column-chunk compression.
    pub compression: Compression,
    /// Per-column Parquet data-page size limit (uncompressed bytes)
    /// applied to the `id_column` only. Small pages let a point
    /// lookup (`take_by_local_doc_ids`) decompress just the tiny
    /// page holding the requested row instead of the whole
    /// row-group-sized page, which is the dominant `resolve_hits`
    /// cost. Compression stays on; the only cost is a few extra
    /// page headers + offset-index entries for the id column.
    pub id_page_size_limit: usize,
}

/// Default per-column data-page size limit for the id column
/// (uncompressed bytes). At 16 bytes/row (`Decimal128`) this is
/// ~512 rows/page, vs the ~65 536-row single page a default
/// (1 MiB) limit produces for a full row group.
///
/// Non-id columns keep parquet's default page size: shrinking them
/// was measured (320K-doc segments, k=10) to leave full-row resolve
/// flat and regress the `[_id, score]` path 8× — per-hit resolve
/// cost scales with page COUNT (selection planning / offset-index
/// walks), not page decode volume.
pub const DEFAULT_ID_PAGE_SIZE_LIMIT: usize = 8 * 1024;

impl BuilderOptions {
    /// Default `row_group_size = 65_536`, `compression = ZSTD(3)`.
    ///
    /// TODO: expose `row_group_size` and `compression` as
    /// `supertable.parquet.*` fields in `config.yaml` so
    /// operators can tune them per deployment without
    /// recompiling. Follow the existing pattern of
    /// `supertable.commit_threshold_size_mb` →
    /// `SupertableOptions::apply_config` (which already
    /// lives at the config layer with its own default).
    pub fn new(
        schema: Arc<Schema>,
        id_column: impl Into<String>,
        fts_columns: Vec<FtsConfig>,
        vector_columns: Vec<VectorConfig>,
        tokenizer: Option<Arc<dyn Tokenizer>>,
    ) -> Self {
        Self {
            schema,
            id_column: id_column.into(),
            fts_columns,
            vector_columns,
            tokenizer,
            row_group_size: 65_536,
            compression: Compression::ZSTD(
                parquet::basic::ZstdLevel::try_new(3)
                    .expect("zstd level 3 is in the valid 1..=22 range"),
            ),
            id_page_size_limit: DEFAULT_ID_PAGE_SIZE_LIMIT,
        }
    }

    pub fn new_from_reader(reader: &SuperfileReader) -> Self {
        // TODO: Fetch tokenizer from reader. Not possible at the moment because we don't
        // store the tokenizer in the reader. Should work for now because we only have AsciiLowerTokenizer.
        let tokenizer = Arc::new(AsciiLowerTokenizer);
        let fts_columns = if let Some(fts) = &reader.fts() {
            fts.fts_columns_config()
                .map(|c| FtsConfig {
                    column: c.name.clone(),
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let vector_columns = if let Some(vec) = &reader.vec() {
            vec.vector_columns_config()
                .map(|v| {
                    VectorConfig::new(
                        v.name.clone(),
                        v.dim,
                        v.n_cent as usize,
                        v.rot_seed,
                        v.metric,
                    )
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        BuilderOptions::new(
            reader.schema().clone(),
            reader.id_column(),
            fts_columns,
            vector_columns,
            Some(tokenizer),
        )
    }

    fn check_mergeability(
        &self,
        remote_id_col: &str,
        remote_schema: &Arc<Schema>,
        remote_fts_columns: Option<Vec<&str>>,
        remote_vector_columns: Option<Vec<&ColumnReader>>,
    ) -> Result<bool, BuildError> {
        if self.id_column != *remote_id_col {
            return Err(BuildError::IdColumnMismatch(
                self.id_column.clone(),
                remote_id_col.to_string(),
            ));
        }

        if self.schema.fields() != remote_schema.fields() {
            return Err(BuildError::SchemaMismatch {
                mine: self.schema.to_string(),
                other: remote_schema.to_string(),
            });
        }

        if let Some(remote_fts_columns) = remote_fts_columns {
            let self_fts_columns = &self.fts_columns;
            if self_fts_columns.len() != remote_fts_columns.len() {
                return Err(BuildError::FTSSchemaMismatch(format!(
                    "mismatched column len. self {} vs other {}",
                    self_fts_columns.len(),
                    remote_fts_columns.len()
                )));
            }
            for (self_fts_column, remote_fts_column) in
                self_fts_columns.iter().zip(remote_fts_columns.iter())
            {
                if self_fts_column.column != *remote_fts_column {
                    return Err(BuildError::FTSSchemaMismatch(format!(
                        "mismatched column name. self {} vs other {}",
                        self_fts_column.column, remote_fts_column
                    )));
                }
            }
        }

        if let Some(remote_vector_columns) = remote_vector_columns {
            let self_vec_columns = &self.vector_columns;
            if self_vec_columns.len() != remote_vector_columns.len() {
                return Err(BuildError::VectorSchemaMismatch(format!(
                    "mismatched column len. self {} vs other {}",
                    self_vec_columns.len(),
                    remote_vector_columns.len()
                )));
            }

            for (self_vec_column, remote_vector_column) in
                self_vec_columns.iter().zip(remote_vector_columns.iter())
            {
                if self_vec_column.column != remote_vector_column.name {
                    return Err(BuildError::VectorSchemaMismatch(format!(
                        "mismatched column name. self {} vs other {}",
                        self_vec_column.column, remote_vector_column.name
                    )));
                }
                if self_vec_column.dim != remote_vector_column.dim {
                    return Err(BuildError::VectorSchemaMismatch(format!(
                        "mismatched column dim. self {} vs other {}",
                        self_vec_column.dim, remote_vector_column.dim
                    )));
                }
            }
        }

        Ok(true)
    }
}

impl std::fmt::Debug for SuperfileBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SuperfileBuilder")
            .field("id_column", &self.opts.id_column)
            .field("n_fts_columns", &self.opts.fts_columns.len())
            .field("n_vector_columns", &self.opts.vector_columns.len())
            .field("n_batches", &self.batches.len())
            .field("next_local_doc_id", &self.next_local_doc_id)
            .finish()
    }
}

pub struct SuperfileBuilder {
    opts: BuilderOptions,
    /// Cached column indices for FTS columns, parallel to `opts.fts_columns`.
    fts_col_idxs: Vec<usize>,
    /// Accumulated input batches. Drained at `finish()`.
    batches: Vec<RecordBatch>,
    /// FtsBuilder accumulating tokens across every `add_batch`.
    /// `None` if `opts.fts_columns` is empty.
    fts_builder: Option<FtsBuilder>,
    /// VectorBuilder accumulating vectors across every `add_batch`.
    /// `None` if `opts.vector_columns` is empty.
    vec_builder: Option<VectorBuilder>,
    /// Running local doc-id counter, increments with every row in
    /// every `add_batch`.
    next_local_doc_id: u32,
}

impl SuperfileBuilder {
    /// Construct from options. Validates schema + names; returns
    /// `BuildError::*` on any inconsistency.
    pub fn new(opts: BuilderOptions) -> Result<Self, BuildError> {
        // 1. id_column must exist and be `Decimal128(38, 0)`.
        //    Precision 38 + scale 0 carries every 128-bit
        //    signed integer value without truncation; that's
        //    the type the supertable injects via its
        //    snowflake-shaped IdGenerator.
        let id_idx = opts
            .schema
            .index_of(&opts.id_column)
            .map_err(|_| BuildError::MissingIdColumn(opts.id_column.clone()))?;
        let id_field = opts.schema.field(id_idx);
        let expected = DataType::Decimal128(38, 0);
        if id_field.data_type() != &expected {
            return Err(BuildError::IdColumnWrongType(
                opts.id_column.clone(),
                format!("{:?}", id_field.data_type()),
            ));
        }

        // 2. Each FTS column must exist and be LargeUtf8.
        let mut fts_col_idxs = Vec::with_capacity(opts.fts_columns.len());
        for fc in &opts.fts_columns {
            let idx = opts
                .schema
                .index_of(&fc.column)
                .map_err(|_| BuildError::FtsColumnMissing(fc.column.clone()))?;
            let f = opts.schema.field(idx);
            if f.data_type() != &DataType::LargeUtf8 {
                return Err(BuildError::FtsColumnMustBeLargeUtf8 {
                    column: fc.column.clone(),
                    actual: format!("{:?}", f.data_type()),
                });
            }
            fts_col_idxs.push(idx);
        }

        // 3. No reserved separator / prefix / duplication across the
        //    combined logical-name namespace (FTS + vector + any
        //    schema-name-vs-vector collision).
        let mut seen_logical: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for fc in &opts.fts_columns {
            check_user_column_name(&fc.column)?;
            if !seen_logical.insert(fc.column.as_str()) {
                return Err(BuildError::DuplicateLogicalName(fc.column.clone()));
            }
        }
        for vc in &opts.vector_columns {
            check_user_column_name(&vc.column)?;
            if !seen_logical.insert(vc.column.as_str()) {
                return Err(BuildError::DuplicateLogicalName(vc.column.clone()));
            }
            // Vector logical name must not collide with a schema column.
            if opts.schema.index_of(&vc.column).is_ok() {
                return Err(BuildError::DuplicateLogicalName(vc.column.clone()));
            }
        }

        // 4. FTS requires a tokenizer.
        if !opts.fts_columns.is_empty() && opts.tokenizer.is_none() {
            return Err(BuildError::FtsColumnTypeInvalid {
                column: opts.fts_columns[0].column.clone(),
                actual: "missing tokenizer in BuilderOptions".to_string(),
            });
        }

        // 5. Wire up the unified FTS + vector sub-builders.
        let fts_builder = if opts.fts_columns.is_empty() {
            None
        } else {
            let tk = opts
                .tokenizer
                .as_ref()
                .expect("validated non-empty FTS implies Some tokenizer")
                .clone();
            let mut fb = FtsBuilder::new(tk);
            for fc in &opts.fts_columns {
                fb.register_column(fc.column.clone())?;
            }
            Some(fb)
        };

        let vec_builder = if opts.vector_columns.is_empty() {
            None
        } else {
            let mut vb = VectorBuilder::new();
            for vc in &opts.vector_columns {
                // VectorConfig is now the same type at both layers
                // (re-exported from vector::builder), so the manual
                // field-by-field bridge is gone — just clone.
                vb.register_column(vc.clone())?;
            }
            Some(vb)
        };

        Ok(Self {
            opts,
            fts_col_idxs,
            batches: Vec::new(),
            fts_builder,
            vec_builder,
            next_local_doc_id: 0,
        })
    }

    /// Override the FTS builder's in-RAM spill threshold (forwarded
    /// to [`FtsBuilder::set_spill_threshold_bytes`]). No-op if this
    /// `SuperfileBuilder` was constructed without any FTS columns.
    ///
    /// Primarily useful for tests that need to force the spill +
    /// streaming-FST finish path on a corpus too small to cross the
    /// default 256 MiB threshold; production callers should leave
    /// the default in place.
    pub fn set_fts_spill_threshold_bytes(&mut self, threshold: usize) {
        if let Some(fb) = self.fts_builder.as_mut() {
            fb.set_spill_threshold_bytes(threshold);
        }
    }

    /// Append a `RecordBatch`. Its schema must match
    /// `opts.schema` field-for-field. `vectors[i]` is the flat f32
    /// buffer for `opts.vector_columns[i]`, length
    /// `batch.num_rows() * vector_columns[i].dim`.
    pub fn add_batch(&mut self, batch: &RecordBatch, vectors: &[&[f32]]) -> Result<(), BuildError> {
        if batch.schema().fields() != self.opts.schema.fields() {
            return Err(BuildError::BatchSchemaMismatch);
        }
        if vectors.len() != self.opts.vector_columns.len() {
            return Err(BuildError::VectorCountMismatch {
                expected: self.opts.vector_columns.len(),
                actual: vectors.len(),
            });
        }
        let n_rows = batch.num_rows() as u32;

        // Validate vector slice lengths up-front before mutating any state.
        for (i, vc) in self.opts.vector_columns.iter().enumerate() {
            let expected_total = (n_rows as usize) * vc.dim;
            if vectors[i].len() != expected_total {
                return Err(BuildError::VectorDimMismatch {
                    column: vc.column.clone(),
                    expected: expected_total,
                    actual: vectors[i].len(),
                });
            }
        }

        // Route FTS columns. Pull each column's LargeStringArray once.
        if let Some(fb) = self.fts_builder.as_mut() {
            for (col_id, &schema_idx) in self.fts_col_idxs.iter().enumerate() {
                let arr = batch.column(schema_idx);
                let strs = arr
                    .as_any()
                    .downcast_ref::<arrow_array::LargeStringArray>()
                    .expect("schema validated as LargeUtf8");
                for row in 0..(n_rows as usize) {
                    let local_doc_id = self.next_local_doc_id + row as u32;
                    // Null-as-empty: we still index a 0-token doc so doc_lengths
                    // stays in lock-step with Parquet rows.
                    let text = if strs.is_null(row) {
                        ""
                    } else {
                        strs.value(row)
                    };
                    fb.add_doc(col_id as u32, local_doc_id, text)?;
                }
            }
        }

        // Route vectors.
        if let Some(vb) = self.vec_builder.as_mut() {
            for (i, vc) in self.opts.vector_columns.iter().enumerate() {
                let dim = vc.dim;
                for row in 0..(n_rows as usize) {
                    let start = row * dim;
                    vb.add(i as u32, &vectors[i][start..start + dim])?;
                }
            }
        }

        self.next_local_doc_id += n_rows;
        self.batches.push(batch.clone());
        Ok(())
    }

    /// Add all data (Parquet + fts + vectors) from another [`SuperfileReader`] to this builder.
    ///
    /// Extracts the record batch and vectors from the reader and adds them via
    /// [`Self::add_batch`]. This is useful for merging superfiles or copying data
    /// between builders.
    ///
    /// **Requirements:**
    /// - The reader's vector columns must use the **Fp32 codec**. Other codecs
    ///   (Sq8ResidualEpsilon, RabitqOnly) will fail with `BuildError::VectorReadError`.
    /// - Vector column names and dimensions in the reader must match those in
    ///   `self.opts.vector_columns` in the exact same order. Mismatches will
    ///   return `BuildError::VectorDimMismatch` error.
    ///
    /// **Memory:** Loads the reader's entire vector dataset into memory at once.
    /// For very large superfiles, consider the memory overhead.
    ///
    /// # Errors
    ///
    /// Returns `BuildError::BatchReadError` if reading the record batch fails.
    ///
    /// Returns `BuildError::VectorReadError` if reading vectors fails
    /// (e.g., codec is not Fp32).
    ///
    /// Returns `BuildError::VectorDimMismatch` if vector column names or
    /// dimensions don't match the builder's configuration.
    pub fn add_batch_from_reader(
        &mut self,
        reader: &SuperfileReader,
        deleted_docs_bitmap: Option<Arc<RoaringBitmap>>,
    ) -> Result<SuperfileStats, BuildError> {
        self.opts.check_mergeability(
            reader.id_column(),
            reader.schema(),
            reader.fts().map(|f| f.fts_columns().collect::<Vec<_>>()),
            reader
                .vec()
                .map(|v| v.vector_columns_config().collect::<Vec<_>>()),
        )?;
        let record_batch = reader
            .get_record_batch(deleted_docs_bitmap.clone())
            .map_err(|_| BuildError::BatchReadError)?;

        let superfile_stats = SuperfileStats::try_compute_from_record_batch(&record_batch)?;

        let num_rows = record_batch.num_rows();
        let mut vectors: Vec<Vec<f32>> = Vec::new();
        if let Some(v) = reader.vec() {
            let reader_columns: Vec<_> = v.vector_columns_config().collect();

            // Validate that reader's vector columns match builder's configuration
            if reader_columns.len() != self.opts.vector_columns.len() {
                return Err(BuildError::VectorDimMismatch {
                    column: format!(
                        "vector column count mismatch: expected {}, got {}",
                        self.opts.vector_columns.len(),
                        reader_columns.len()
                    ),
                    expected: self.opts.vector_columns.len(),
                    actual: reader_columns.len(),
                });
            }

            for (reader_col, builder_col) in reader_columns.iter().zip(&self.opts.vector_columns) {
                if reader_col.name != builder_col.column || reader_col.dim != builder_col.dim {
                    return Err(BuildError::VectorDimMismatch {
                        column: reader_col.name.clone(),
                        expected: builder_col.dim,
                        actual: reader_col.dim,
                    });
                }

                let mut this_col_vectors = Vec::with_capacity(builder_col.dim * num_rows);
                let result = v
                    .get_vectors_fp32(&reader_col.name)
                    .map_err(|_| BuildError::VectorReadError)?;
                for (row_idx, single_row) in result.iter().enumerate() {
                    // Skip deleted documents: only include rows not in the deleted_docs_bitmap
                    if let Some(ref bitmap) = deleted_docs_bitmap
                        && bitmap.contains(row_idx as u32)
                    {
                        continue;
                    }
                    this_col_vectors.extend_from_slice(single_row.as_slice());
                }
                vectors.push(this_col_vectors);
            }
        }

        let slices: Vec<&[f32]> = vectors.iter().map(|row| row.as_slice()).collect();
        self.add_batch(&record_batch, &slices)?;
        Ok(superfile_stats)
    }

    /// Builds a superfile from the given readers, merging them into one.
    pub fn build_from_readers(
        readers: &[(Arc<SuperfileReader>, Option<Arc<RoaringBitmap>>)],
    ) -> Result<(Vec<u8>, SuperfileStats), BuildError> {
        let first = readers.first().ok_or(BuildError::BatchReadError)?;

        let builder_opts = BuilderOptions::new_from_reader(&first.0);
        let mut superfile_builder = SuperfileBuilder::new(builder_opts)?;

        let mut stats_collector = Vec::with_capacity(readers.len());
        for reader in readers {
            let stats = superfile_builder.add_batch_from_reader(&reader.0, reader.1.clone())?;
            stats_collector.push(stats);
        }

        let bytes = superfile_builder.finish()?;
        let stats = SuperfileStats::from_children(stats_collector.as_slice());

        Ok((bytes, stats))
    }

    /// Consume the builder and emit one self-contained superfile.
    ///
    /// If no `add_batch` calls have landed any rows, returns an
    /// empty `Vec<u8>` — there's no Parquet body to write and no
    /// FTS/vector blobs to embed.
    pub fn finish(mut self) -> Result<Vec<u8>, BuildError> {
        if self.next_local_doc_id == 0 {
            return Ok(Vec::new());
        }
        let n_docs = self.next_local_doc_id as u64;

        let fts_builder = self.fts_builder.take();
        let vec_builder = self.vec_builder.take();

        // Assemble inf.* KV metadata (cheap; do it before the parallel
        // section so the splice has it ready).
        let mut kvs: Vec<(String, String)> = vec![
            (kv::FORMAT.into(), kv::FORMAT_VALUE.into()),
            (kv::FORMAT_VERSION.into(), format::FORMAT_VERSION.into()),
            (kv::ID_COLUMN.into(), self.opts.id_column.clone()),
            (kv::N_DOCS.into(), n_docs.to_string()),
            (kv::BUILDER.into(), crate::BUILDER_ID.to_string()),
        ];
        if !self.opts.fts_columns.is_empty() {
            kvs.push((
                kv::FTS_COLUMNS.into(),
                fts_columns_json(&self.opts.fts_columns),
            ));
        }
        if !self.opts.vector_columns.is_empty() {
            kvs.push((
                kv::VEC_COLUMNS.into(),
                vec_columns_json(&self.opts.vector_columns),
            ));
        }

        // A superfile has three independent build outputs: the scalar /
        // relational Parquet body (the SQL-queryable columns), the FTS
        // blob, and the vector blob. None reads another's bytes — blobs
        // are appended after the last row group, and FTS/vector
        // finalization share no state — so they can run concurrently.
        //
        // But how to overlap them depends on the vector index. The
        // vector finalizer already saturates every core via its own
        // rayon `par_iter` (rotation / encode / quantize), so overlapping
        // the *serial* Parquet body encode with it just steals a core
        // from the bottleneck — a measured regression on vector builds.
        // So: when a vector index is present, finalize the index blobs
        // (FTS ‖ vector) first and encode the body afterward. When it is
        // absent, the FTS finalizer doesn't saturate the pool, so hide
        // the body encode behind it (body ‖ FTS). The final splice (byte
        // appends + footer rewrite) is cheap and stays serial.
        let id_page_limit = [(self.opts.id_column.as_str(), self.opts.id_page_size_limit)];
        let encode_body = || {
            encode_parquet_body(
                &self.opts.schema,
                &self.batches,
                self.opts.compression,
                self.opts.row_group_size,
                &id_page_limit,
            )
        };
        let (body, fts_blob, vec_blob) = if vec_builder.is_some() {
            let (fts_blob, vec_blob) = finish_index_blobs(fts_builder, vec_builder)?;
            let body = encode_body()?;
            (body, fts_blob, vec_blob)
        } else {
            let (body_res, blobs_res) =
                rayon::join(encode_body, || finish_index_blobs(fts_builder, vec_builder));
            let body = body_res?;
            let (fts_blob, vec_blob) = blobs_res?;
            (body, fts_blob, vec_blob)
        };

        let parts = splice_index_blobs(body, &fts_blob, &vec_blob, &kvs)?;
        Ok(parts.bytes)
    }
}

/// Finish the independent embedded index blobs. Once `add_batch` has
/// routed scalar text and vectors into their builders, FTS and vector
/// finalization do not share mutable state, so build them as sibling
/// rayon jobs when both indexes are present.
fn finish_index_blobs(
    fts_builder: Option<FtsBuilder>,
    vec_builder: Option<VectorBuilder>,
) -> Result<(Vec<u8>, Vec<u8>), BuildError> {
    match (fts_builder, vec_builder) {
        (Some(fb), Some(vb)) => {
            let (fts, vec) = rayon::join(|| fb.finish(), || vb.finish());
            Ok((fts?, vec?))
        }
        (Some(fb), None) => Ok((fb.finish()?, Vec::new())),
        (None, Some(vb)) => Ok((Vec::new(), vb.finish()?)),
        (None, None) => Ok((Vec::new(), Vec::new())),
    }
}

/// Reject user-supplied column names that would collide with
/// infino's internal byte-protocol or KV-key conventions:
///
/// - `\x1F` (ASCII Unit Separator) is the FST dictionary's
///   `(column_id, term)` separator. A column name containing
///   it would break the FST decode path that splits on it.
/// - The `inf.` prefix is reserved for the infino-managed
///   Parquet KV metadata keys (`inf.format`, `inf.fts.columns`,
///   etc.). Allowing a user column to start with it would risk
///   collision with future infino-defined keys.
///
/// Called at `SuperfileBuilder::new` for every FTS and vector
/// column. The supertable layer carries the same check (under
/// the same name) on its own column lists so callers see the
/// typed error at the earliest possible construction point.
fn check_user_column_name(name: &str) -> Result<(), BuildError> {
    if name.as_bytes().contains(&format::FST_SEPARATOR) {
        return Err(BuildError::ReservedSeparatorInColumnName(name.to_string()));
    }
    if name.starts_with(format::RESERVED_PREFIX) {
        return Err(BuildError::ReservedPrefixInColumnName(name.to_string()));
    }
    Ok(())
}

/// Serialize `[FtsConfig]` to the JSON form stored in the
/// Parquet KV metadata key `inf.fts.columns`. Hand-rolled
/// because the shape is fixed + small and `serde_derive` on
/// `FtsConfig` would add a derived `Serialize` impl across
/// the format boundary purely to write five characters of
/// JSON per column.
///
/// Output shape per column:
/// `{"name":"<escaped>","tokenizer":"ascii_lower"}`.
/// `ascii_lower` is hardcoded today because that's the only
/// tokenizer the format supports.
fn fts_columns_json(cols: &[FtsConfig]) -> String {
    let mut s = String::from("[");
    for (i, c) in cols.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(r#"{"name":""#);
        s.push_str(&escape_json(&c.column));
        s.push_str(r#"","tokenizer":"ascii_lower"}"#);
    }
    s.push(']');
    s
}

/// Serialize `[VectorConfig]` to the JSON form stored in the
/// Parquet KV metadata key `inf.vec.columns`. Same hand-rolled
/// rationale as `fts_columns_json` — fixed shape, no derived
/// `Serialize` needed.
///
/// Output shape per column:
/// `{"column":"<escaped>","dim":<u>,"n_cent":<u>,"rot_seed":<u>,"metric":"<l2sq|cosine|negdot>"}`.
/// The reader at open time parses this back into
/// `VectorConfig` to drive distance kernels + IVF probing.
fn vec_columns_json(cols: &[VectorConfig]) -> String {
    let mut s = String::from("[");
    for (i, c) in cols.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(r#"{"column":""#);
        s.push_str(&escape_json(&c.column));
        s.push_str(r#"","dim":"#);
        s.push_str(&c.dim.to_string());
        s.push_str(r#","n_cent":"#);
        s.push_str(&c.n_cent.to_string());
        s.push_str(r#","rot_seed":"#);
        s.push_str(&c.rot_seed.to_string());
        s.push_str(r#","metric":""#);
        s.push_str(metric_str(c.metric));
        s.push_str("\"}");
    }
    s.push(']');
    s
}

/// Stable string label for each `Metric` variant — the form
/// stored in `inf.vec.columns` JSON. Matches the strings the
/// reader's parser accepts; do not rename without updating
/// both sides.
fn metric_str(m: Metric) -> &'static str {
    match m {
        Metric::L2Sq => "l2sq",
        Metric::Cosine => "cosine",
        Metric::NegDot => "negdot",
    }
}

/// Minimal JSON string-value escape: quote, backslash, the
/// four whitespace escapes JSON requires, plus the
/// `\u00XX`-encoded form for any other control character
/// (< 0x20). All other characters (including all non-ASCII)
/// pass through unchanged — column names are arbitrary
/// UTF-8 and JSON strings are UTF-8 natively, so escaping
/// non-control non-quote characters would only bloat the
/// output.
fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{decimal128_ids, default_tokenizer, default_vector_config};
    use arrow_array::{Decimal128Array, LargeStringArray};
    use arrow_schema::Field;
    use bytes::Bytes;
    use roaring::RoaringBitmap;
    use std::sync::Arc;

    fn schema_with_fts() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("body", DataType::LargeUtf8, false),
        ]))
    }

    fn opts_minimal() -> BuilderOptions {
        BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            Some(default_tokenizer()),
        )
    }

    #[test]
    fn new_rejects_missing_id_column() {
        let mut opts = opts_minimal();
        opts.id_column = "nope".into();
        let err = SuperfileBuilder::new(opts).expect_err("expected error");
        assert!(matches!(err, BuildError::MissingIdColumn(_)));
    }

    #[test]
    fn new_rejects_id_column_not_decimal128_38_0() {
        // Every type listed here should be rejected with
        // `BuildError::IdColumnWrongType`. Coverage spans:
        //   - UInt64: the historical id type before the supertable
        //     layer's 128-bit Snowflake forced Decimal128. Most
        //     likely real-world miss for a caller migrating from an
        //     older fixture.
        //   - Int64: the previous regression case; kept so this
        //     test still subsumes what the old one covered.
        //   - Decimal128(38, 1) and Decimal128(37, 0): right type
        //     family, wrong scale / precision. These are the cases
        //     a caller *trying* to comply but typo'ing the
        //     parameters would hit — exactly where the rule's
        //     strictness matters.
        let cases = [
            DataType::UInt64,
            DataType::Int64,
            DataType::Decimal128(38, 1),
            DataType::Decimal128(37, 0),
        ];
        for ty in cases {
            let schema = Arc::new(Schema::new(vec![
                Field::new("doc_id", ty.clone(), false),
                Field::new("title", DataType::LargeUtf8, false),
            ]));
            let opts = BuilderOptions::new(
                schema,
                "doc_id",
                vec![FtsConfig {
                    column: "title".into(),
                }],
                vec![],
                Some(default_tokenizer()),
            );
            let err =
                SuperfileBuilder::new(opts).expect_err(&format!("expected rejection for {ty:?}"));
            assert!(
                matches!(err, BuildError::IdColumnWrongType(_, _)),
                "wrong error variant for {ty:?}: {err:?}",
            );
        }
    }

    #[test]
    fn new_rejects_fts_column_missing_from_schema() {
        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![FtsConfig {
                column: "nope".into(),
            }],
            vec![],
            Some(default_tokenizer()),
        );
        let err = SuperfileBuilder::new(opts).expect_err("expected error");
        assert!(matches!(err, BuildError::FtsColumnMissing(_)));
    }

    #[test]
    fn new_rejects_fts_column_wrong_type() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("title", DataType::Utf8, false),
        ]));
        let opts = BuilderOptions::new(
            schema,
            "doc_id",
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            Some(default_tokenizer()),
        );
        let err = SuperfileBuilder::new(opts).expect_err("expected error");
        assert!(matches!(err, BuildError::FtsColumnMustBeLargeUtf8 { .. }));
    }

    #[test]
    fn new_rejects_duplicate_logical_name_across_fts_and_vector() {
        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![default_vector_config("title", 1)],
            Some(default_tokenizer()),
        );
        let err = SuperfileBuilder::new(opts).expect_err("expected error");
        assert!(matches!(err, BuildError::DuplicateLogicalName(_)));
    }

    #[test]
    fn new_rejects_vector_column_collides_with_schema() {
        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![],
            vec![default_vector_config("body", 1)], // same name as a schema column
            None,
        );
        let err = SuperfileBuilder::new(opts).expect_err("expected error");
        assert!(matches!(err, BuildError::DuplicateLogicalName(_)));
    }

    #[test]
    fn new_rejects_reserved_prefix_in_logical_name() {
        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![],
            vec![default_vector_config("inf.bad", 1)],
            None,
        );
        let err = SuperfileBuilder::new(opts).expect_err("expected error");
        assert!(matches!(err, BuildError::ReservedPrefixInColumnName(_)));
    }

    #[test]
    fn new_with_fts_requires_tokenizer() {
        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            None,
        );
        let err = SuperfileBuilder::new(opts).expect_err("expected error");
        assert!(matches!(err, BuildError::FtsColumnTypeInvalid { .. }));
    }

    fn batch_two_rows(schema: &Arc<Schema>) -> RecordBatch {
        let ids = decimal128_ids(vec![10u64, 11]);
        let title = LargeStringArray::from(vec!["hello world", "rust async"]);
        let body = LargeStringArray::from(vec!["foo bar", "baz quux"]);
        RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(ids), Arc::new(title), Arc::new(body)],
        )
        .expect("build RecordBatch")
    }

    #[test]
    fn add_batch_increments_next_local_doc_id() {
        let mut b = SuperfileBuilder::new(opts_minimal()).expect("new SuperfileBuilder");
        let schema = b.opts.schema.clone();
        let batch = batch_two_rows(&schema);
        b.add_batch(&batch, &[]).expect("add_batch");
        assert_eq!(b.next_local_doc_id, 2);
        b.add_batch(&batch, &[]).expect("add_batch");
        assert_eq!(b.next_local_doc_id, 4);
    }

    #[test]
    fn add_batch_rejects_schema_mismatch() {
        let mut b = SuperfileBuilder::new(opts_minimal()).expect("new SuperfileBuilder");
        // Intentionally mismatched: a single-column UInt64 schema
        // whose type doesn't match the builder's
        // Decimal128(38, 0) id column.
        let other = Arc::new(Schema::new(vec![Field::new(
            "doc_id",
            DataType::UInt64,
            false,
        )]));
        let bad = RecordBatch::try_new(
            other,
            vec![Arc::new(arrow_array::UInt64Array::from(vec![1u64]))],
        )
        .expect("build RecordBatch");
        let err = b.add_batch(&bad, &[]).expect_err("expected error");
        assert!(matches!(err, BuildError::BatchSchemaMismatch));
    }

    #[test]
    fn add_batch_rejects_wrong_vector_count() {
        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![],
            vec![default_vector_config("emb", 1)],
            None,
        );
        let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
        let schema = b.opts.schema.clone();
        let batch = batch_two_rows(&schema);
        let err = b.add_batch(&batch, &[]).expect_err("expected error");
        assert!(matches!(err, BuildError::VectorCountMismatch { .. }));
    }

    #[test]
    fn add_batch_rejects_wrong_vector_dim() {
        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![],
            vec![default_vector_config("emb", 1)],
            None,
        );
        let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
        let schema = b.opts.schema.clone();
        let batch = batch_two_rows(&schema);
        // Need 2 rows × 16 dim = 32 floats; pass 30 instead.
        let bad: Vec<f32> = vec![0.0; 30];
        let err = b
            .add_batch(&batch, &[bad.as_slice()])
            .expect_err("expected error");
        assert!(matches!(err, BuildError::VectorDimMismatch { .. }));
    }

    #[test]
    fn finish_with_no_indexes_produces_valid_parquet() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("title", DataType::LargeUtf8, false),
        ]));
        let opts = BuilderOptions::new(schema.clone(), "doc_id", vec![], vec![], None);
        let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
        let ids = decimal128_ids(vec![1u64, 2, 3]);
        let titles = LargeStringArray::from(vec!["a", "b", "c"]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(titles)])
            .expect("build RecordBatch");
        b.add_batch(&batch, &[]).expect("add_batch");
        let bytes = b.finish().expect("finish builder");
        // Must be a valid Parquet file.
        assert_eq!(&bytes[..4], b"PAR1");
        assert_eq!(&bytes[bytes.len() - 4..], b"PAR1");
    }

    #[test]
    fn finish_emits_required_kv_pointers_for_fts() {
        let mut b = SuperfileBuilder::new(opts_minimal()).expect("new SuperfileBuilder");
        let schema = b.opts.schema.clone();
        let batch = batch_two_rows(&schema);
        b.add_batch(&batch, &[]).expect("add_batch");
        let bytes = b.finish().expect("finish builder");
        let kv =
            crate::superfile::format::footer::read_kv_metadata(&bytes).expect("read kv metadata");
        assert_eq!(
            kv.get("inf.format").map(String::as_str),
            Some("infino-superfile")
        );
        assert_eq!(kv.get("inf.id_column").map(String::as_str), Some("doc_id"));
        assert_eq!(kv.get("inf.n_docs").map(String::as_str), Some("2"));
        assert!(kv.contains_key("inf.fts.offset"));
        assert!(kv.contains_key("inf.fts.length"));
        assert!(kv.contains_key("inf.fts.columns"));
        assert!(!kv.contains_key("inf.vec.offset"));
    }

    #[test]
    fn finish_emits_kv_pointers_for_vectors() {
        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![],
            vec![default_vector_config("emb", 7)],
            None,
        );
        let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
        let schema = b.opts.schema.clone();
        let batch = batch_two_rows(&schema);
        // 2 rows × 16 dim, normalized so cosine doesn't NaN — simple
        // unit-axis vectors per row.
        let mut v: Vec<f32> = vec![0.0; 32];
        v[0] = 1.0;
        v[16 + 1] = 1.0;
        b.add_batch(&batch, &[v.as_slice()]).expect("add_batch");
        let bytes = b.finish().expect("finish builder");
        let kv =
            crate::superfile::format::footer::read_kv_metadata(&bytes).expect("read kv metadata");
        assert!(kv.contains_key("inf.vec.offset"));
        assert!(kv.contains_key("inf.vec.length"));
        assert!(kv.contains_key("inf.vec.columns"));
        assert!(!kv.contains_key("inf.fts.offset"));
    }

    #[test]
    fn fts_columns_json_round_trip_shape() {
        let cols = vec![
            FtsConfig {
                column: "title".into(),
            },
            FtsConfig {
                column: "body".into(),
            },
        ];
        let s = fts_columns_json(&cols);
        assert!(s.starts_with('['));
        assert!(s.contains(r#""name":"title""#));
        assert!(s.contains(r#""name":"body""#));
        assert!(s.contains(r#""tokenizer":"ascii_lower""#));
    }

    #[test]
    fn vec_columns_json_round_trip_shape() {
        let cols = vec![VectorConfig {
            column: "emb".into(),
            dim: 384,
            n_cent: 64,
            rot_seed: 99,
            metric: Metric::L2Sq,
            rerank_codec: crate::superfile::vector::rerank_codec::RerankCodec::Fp32,
        }];
        let s = vec_columns_json(&cols);
        assert!(s.contains(r#""column":"emb""#));
        assert!(s.contains(r#""dim":384"#));
        assert!(s.contains(r#""n_cent":64"#));
        assert!(s.contains(r#""rot_seed":99"#));
        assert!(s.contains(r#""metric":"l2sq""#));
    }

    #[test]
    fn escape_json_handles_control_chars() {
        assert_eq!(escape_json(r#"a"b"#), r#"a\"b"#);
        assert_eq!(escape_json("a\\b"), "a\\\\b");
        assert_eq!(escape_json("a\nb"), "a\\nb");
        assert_eq!(escape_json("a\x01b"), "a\\u0001b");
    }

    #[test]
    fn add_batch_from_reader_on_empty_builder_produces_identical_superfile() {
        // Build original superfile with FTS and vectors
        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![default_vector_config("emb", 7)],
            Some(default_tokenizer()),
        );
        let mut b1 = SuperfileBuilder::new(opts.clone()).expect("new SuperfileBuilder");
        let schema = b1.opts.schema.clone();
        let batch = batch_two_rows(&schema);
        let mut v: Vec<f32> = vec![0.0; 32]; // 2 rows × 16 dim
        v[0] = 1.0;
        v[16 + 1] = 1.0;
        b1.add_batch(&batch, &[v.as_slice()]).expect("add_batch");
        let original_bytes = b1.finish().expect("finish builder");

        // Read the superfile
        let reader = SuperfileReader::open(Bytes::from(original_bytes.clone()))
            .expect("open superfile reader");

        // Create a new builder and add from reader
        let mut b2 = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
        let stats = b2
            .add_batch_from_reader(&reader, None)
            .expect("add_batch_from_reader");
        let merged_bytes = b2.finish().expect("finish builder");

        // Verify stats are populated correctly
        assert_eq!(stats.n_docs, 2, "stats should report 2 documents");
        assert_eq!(stats.id_min, 10, "id_min should be 10");
        assert_eq!(stats.id_max, 11, "id_max should be 11");

        // Verify scalar_stats contains entries for all scalar columns
        assert!(
            !stats.scalar_stats.cols.is_empty(),
            "scalar_stats should have column entries"
        );
        assert!(
            stats.scalar_stats.cols.contains_key("doc_id"),
            "scalar_stats should contain id_column"
        );
        assert!(
            stats.scalar_stats.cols.contains_key("title"),
            "scalar_stats should contain FTS column"
        );
        assert!(
            stats.scalar_stats.cols.contains_key("body"),
            "scalar_stats should contain body column"
        );

        // Verify scalar_stats values match expected min/max
        // doc_id: IDs are [10, 11], so min=10, max=11
        let (id_min_arr, id_max_arr) = stats
            .scalar_stats
            .cols
            .get("doc_id")
            .expect("doc_id should have stats");
        let id_min = id_min_arr
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("id min should be Decimal128")
            .value(0);
        let id_max = id_max_arr
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("id max should be Decimal128")
            .value(0);
        assert_eq!(id_min, 10i128, "doc_id min should be 10");
        assert_eq!(id_max, 11i128, "doc_id max should be 11");

        // title: ["hello world", "rust async"], so min="hello world", max="rust async"
        let (title_min_arr, title_max_arr) = stats
            .scalar_stats
            .cols
            .get("title")
            .expect("title should have stats");
        let title_min = title_min_arr
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("title min should be LargeUtf8")
            .value(0);
        let title_max = title_max_arr
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("title max should be LargeUtf8")
            .value(0);
        assert_eq!(
            title_min, "hello world",
            "title min should be 'hello world'"
        );
        assert_eq!(title_max, "rust async", "title max should be 'rust async'");

        // body: ["foo bar", "baz quux"], so min="baz quux", max="foo bar"
        let (body_min_arr, body_max_arr) = stats
            .scalar_stats
            .cols
            .get("body")
            .expect("body should have stats");
        let body_min = body_min_arr
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("body min should be LargeUtf8")
            .value(0);
        let body_max = body_max_arr
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("body max should be LargeUtf8")
            .value(0);
        assert_eq!(body_min, "baz quux", "body min should be 'baz quux'");
        assert_eq!(body_max, "foo bar", "body max should be 'foo bar'");

        // The two superfiles should be identical
        assert_eq!(
            original_bytes, merged_bytes,
            "superfile created from reader should be identical to original"
        );
    }

    #[test]
    fn add_batch_from_reader_adds_parquet_data_correctly() {
        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            Some(default_tokenizer()),
        );
        let mut b1 = SuperfileBuilder::new(opts.clone()).expect("new SuperfileBuilder");
        let schema = b1.opts.schema.clone();
        let batch = batch_two_rows(&schema);
        b1.add_batch(&batch, &[]).expect("add_batch");
        let bytes = b1.finish().expect("finish builder");

        // Read and verify parquet data
        let reader = SuperfileReader::open(Bytes::from(bytes)).expect("open superfile reader");
        let reader_batch = reader
            .get_record_batch(None)
            .expect("get_record_batch from reader");

        // Should have 2 rows
        assert_eq!(reader_batch.num_rows(), 2);

        // Now add to a new builder
        let mut b2 = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
        let stats = b2
            .add_batch_from_reader(&reader, None)
            .expect("add_batch_from_reader");
        assert_eq!(stats.n_docs, 2, "stats should report 2 documents");
        assert_eq!(stats.id_min, 10, "id_min should be 10");
        assert_eq!(stats.id_max, 11, "id_max should be 11");
        assert!(
            !stats.scalar_stats.cols.is_empty(),
            "scalar_stats should have column entries"
        );
        let merged_bytes = b2.finish().expect("finish builder");

        // Read back and verify parquet data is correct
        let reader2 =
            SuperfileReader::open(Bytes::from(merged_bytes)).expect("open merged superfile reader");
        let merged_batch = reader2
            .get_record_batch(None)
            .expect("get_record_batch from merged reader");
        assert_eq!(merged_batch.num_rows(), 2);
    }

    #[test]
    fn add_batch_from_reader_adds_vectors_correctly() {
        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![],
            vec![default_vector_config("emb", 7)],
            None,
        );
        let mut b1 = SuperfileBuilder::new(opts.clone()).expect("new SuperfileBuilder");
        let schema = b1.opts.schema.clone();
        let batch = batch_two_rows(&schema);
        let mut v: Vec<f32> = vec![0.0; 32]; // 2 rows × 16 dim
        v[0] = 1.0;
        v[16 + 1] = 1.0;
        b1.add_batch(&batch, &[v.as_slice()]).expect("add_batch");
        let bytes = b1.finish().expect("finish builder");

        // Read vectors from original superfile
        let reader = SuperfileReader::open(Bytes::from(bytes)).expect("open superfile reader");
        let vectors_before = reader
            .vec()
            .expect("get vector reader")
            .get_vectors_fp32("emb")
            .expect("get vectors fp32");

        let mut b2 = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
        let stats = b2
            .add_batch_from_reader(&reader, None)
            .expect("add_batch_from_reader");
        assert_eq!(stats.n_docs, 2, "stats should report 2 documents");
        assert_eq!(stats.id_min, 10, "id_min should be 10");
        assert_eq!(stats.id_max, 11, "id_max should be 11");
        assert!(
            !stats.scalar_stats.cols.is_empty(),
            "scalar_stats should have column entries"
        );
        let merged_bytes = b2.finish().expect("finish builder");

        // Read vectors from merged superfile
        let reader2 =
            SuperfileReader::open(Bytes::from(merged_bytes)).expect("open merged superfile reader");
        let vectors_after = reader2
            .vec()
            .expect("get vector reader")
            .get_vectors_fp32("emb")
            .expect("get vectors fp32");

        // Vectors should match
        assert_eq!(vectors_before.len(), vectors_after.len());
        for (v1, v2) in vectors_before.iter().zip(vectors_after.iter()) {
            for (val1, val2) in v1.iter().zip(v2.iter()) {
                assert!((val1 - val2).abs() < 1e-6);
            }
        }
    }

    #[tokio::test]
    async fn add_batch_from_reader_adds_fts_correctly() {
        use crate::superfile::fts::reader::BoolMode;

        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            Some(default_tokenizer()),
        );
        let mut b1 = SuperfileBuilder::new(opts.clone()).expect("new SuperfileBuilder");
        let schema = b1.opts.schema.clone();
        let batch = batch_two_rows(&schema);
        b1.add_batch(&batch, &[]).expect("add_batch");
        let bytes = b1.finish().expect("finish builder");

        // Read FTS data from original
        let reader = SuperfileReader::open(Bytes::from(bytes)).expect("open superfile reader");
        let fts_reader = reader.fts().expect("get fts reader");
        let results = fts_reader
            .search("title", &["hello"], 10, BoolMode::Or)
            .await
            .expect("search fts");
        assert_eq!(results.len(), 1);

        // Add to new builder
        let mut b2 = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
        let stats = b2
            .add_batch_from_reader(&reader, None)
            .expect("add_batch_from_reader");
        assert_eq!(stats.n_docs, 2, "stats should report 2 documents");
        assert_eq!(stats.id_min, 10, "id_min should be 10");
        assert_eq!(stats.id_max, 11, "id_max should be 11");
        assert!(
            !stats.scalar_stats.cols.is_empty(),
            "scalar_stats should have column entries"
        );
        let merged_bytes = b2.finish().expect("finish builder");

        // Verify FTS still works after merge
        let reader2 =
            SuperfileReader::open(Bytes::from(merged_bytes)).expect("open merged superfile reader");
        let fts_reader2 = reader2.fts().expect("get fts reader");
        let results2 = fts_reader2
            .search("title", &["hello"], 10, BoolMode::Or)
            .await
            .expect("search fts in merged");
        assert_eq!(results2.len(), 1);
    }

    #[tokio::test]
    async fn add_batch_from_reader_to_non_empty_builder_includes_both_datasets() {
        use crate::superfile::fts::reader::BoolMode;

        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![default_vector_config("emb", 7)],
            Some(default_tokenizer()),
        );

        // Create first superfile
        let mut b1 = SuperfileBuilder::new(opts.clone()).expect("new SuperfileBuilder");
        let schema = b1.opts.schema.clone();
        let batch1 = batch_two_rows(&schema);
        let mut v1: Vec<f32> = vec![0.0; 32];
        v1[0] = 1.0;
        v1[16 + 1] = 1.0;
        b1.add_batch(&batch1, &[v1.as_slice()]).expect("add_batch");
        let bytes1 = b1.finish().expect("finish builder");

        // Create second superfile
        let mut b2 = SuperfileBuilder::new(opts.clone()).expect("new SuperfileBuilder");
        let ids2 = decimal128_ids(vec![20u64, 21]);
        let title2 = LargeStringArray::from(vec!["foo bar", "baz qux"]);
        let body2 = LargeStringArray::from(vec!["quux corge", "grault garply"]);
        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(ids2), Arc::new(title2), Arc::new(body2)],
        )
        .expect("build RecordBatch");
        let mut v2: Vec<f32> = vec![0.0; 32];
        v2[1] = 1.0;
        v2[16] = 1.0;
        b2.add_batch(&batch2, &[v2.as_slice()]).expect("add_batch");
        let _bytes2 = b2.finish().expect("finish builder");

        // Read first superfile
        let reader1 = SuperfileReader::open(Bytes::from(bytes1)).expect("open reader1");

        // Create merged builder - add existing data + reader data
        let mut merged = SuperfileBuilder::new(opts.clone()).expect("new SuperfileBuilder");
        merged
            .add_batch(&batch2, &[v2.as_slice()])
            .expect("add_batch");
        let stats = merged
            .add_batch_from_reader(&reader1, None)
            .expect("add_batch_from_reader");
        assert_eq!(stats.n_docs, 2, "stats should report 2 documents");
        assert_eq!(stats.id_min, 10, "id_min should be 10");
        assert_eq!(stats.id_max, 11, "id_max should be 11");
        assert!(
            !stats.scalar_stats.cols.is_empty(),
            "scalar_stats should have column entries"
        );
        let merged_bytes = merged.finish().expect("finish builder");

        // Verify merged result
        let merged_reader =
            SuperfileReader::open(Bytes::from(merged_bytes)).expect("open merged reader");

        // Should have 4 docs total (2 from batch2 + 2 from reader1)
        let merged_batch = merged_reader
            .get_record_batch(None)
            .expect("get_record_batch");
        assert_eq!(merged_batch.num_rows(), 4);

        // Verify vectors are correct
        let merged_vectors = merged_reader
            .vec()
            .expect("get vector reader")
            .get_vectors_fp32("emb")
            .expect("get vectors");
        assert_eq!(merged_vectors.len(), 4);

        // Verify FTS works and finds both datasets
        let fts_reader = merged_reader.fts().expect("get fts reader");
        let hello_results = fts_reader
            .search("title", &["hello"], 10, BoolMode::Or)
            .await
            .expect("search for hello");
        assert!(
            !hello_results.is_empty(),
            "should find 'hello' from first dataset"
        );

        let foo_results = fts_reader
            .search("title", &["foo"], 10, BoolMode::Or)
            .await
            .expect("search for foo");
        assert!(
            !foo_results.is_empty(),
            "should find 'foo' from second dataset"
        );
    }

    #[test]
    fn add_vector_fp32_returns_correct_vectors() {
        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![],
            vec![default_vector_config("emb", 7)],
            None,
        );
        let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
        let schema = b.opts.schema.clone();
        let batch = batch_two_rows(&schema);
        let mut v: Vec<f32> = vec![0.0; 32]; // 2 rows × 16 dim
        v[0] = 1.0;
        v[16] = 1.0;
        v[17] = 1.0;
        v[31] = 1.0;
        b.add_batch(&batch, &[v.as_slice()]).expect("add_batch");
        let bytes = b.finish().expect("finish builder");

        let reader = SuperfileReader::open(Bytes::from(bytes)).expect("open superfile reader");
        let vectors = reader
            .vec()
            .expect("get vector reader")
            .get_vectors_fp32("emb")
            .expect("get vectors fp32");

        // Verify structure
        assert_eq!(vectors.len(), 2, "should have 2 vectors");
        assert_eq!(
            vectors[0].len(),
            16,
            "first vector should have 16 dimensions"
        );
        assert_eq!(
            vectors[1].len(),
            16,
            "second vector should have 16 dimensions"
        );

        // Verify values
        assert!((vectors[0][0] - 1.0).abs() < 1e-6);
        assert!((vectors[0][1] - 0.0).abs() < 1e-6);
        assert!((vectors[1][0] - 1.0).abs() < 1e-6);
        assert!((vectors[1][1] - 1.0).abs() < 1e-6);
        assert!((vectors[1][15] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn add_vector_fp32_rejects_non_fp32_codec() {
        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![],
            vec![crate::superfile::vector::builder::VectorConfig {
                column: "emb".into(),
                dim: 16,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::L2Sq,
                rerank_codec:
                    crate::superfile::vector::rerank_codec::RerankCodec::Sq8ResidualEpsilon,
            }],
            None,
        );
        let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
        let schema = b.opts.schema.clone();
        let batch = batch_two_rows(&schema);
        let v: Vec<f32> = vec![0.0; 32];
        b.add_batch(&batch, &[v.as_slice()]).expect("add_batch");
        let bytes = b.finish().expect("finish builder");

        let reader = SuperfileReader::open(Bytes::from(bytes)).expect("open superfile reader");
        let result = reader
            .vec()
            .expect("get vector reader")
            .get_vectors_fp32("emb");

        assert!(result.is_err(), "should reject Sq8ResidualEpsilon codec");
    }

    #[tokio::test]
    async fn add_batch_from_reader_queries_work_correctly() {
        use crate::superfile::fts::reader::BoolMode;

        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![default_vector_config("emb", 7)],
            Some(default_tokenizer()),
        );

        // Create original superfile
        let mut b1 = SuperfileBuilder::new(opts.clone()).expect("new SuperfileBuilder");
        let schema = b1.opts.schema.clone();
        let batch = batch_two_rows(&schema);
        let mut v: Vec<f32> = vec![0.0; 32]; // 2 rows × 16 dim
        v[0] = 1.0;
        v[16 + 1] = 1.0;
        b1.add_batch(&batch, &[v.as_slice()]).expect("add_batch");
        let bytes1 = b1.finish().expect("finish builder");

        // Read original superfile
        let reader1 = SuperfileReader::open(Bytes::from(bytes1)).expect("open reader1");

        // Create merged superfile with data from reader
        let mut b_merged = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
        let stats = b_merged
            .add_batch_from_reader(&reader1, None)
            .expect("add_batch_from_reader");
        assert_eq!(stats.n_docs, 2, "stats should report 2 documents");
        assert_eq!(stats.id_min, 10, "id_min should be 10");
        assert_eq!(stats.id_max, 11, "id_max should be 11");
        assert!(
            !stats.scalar_stats.cols.is_empty(),
            "scalar_stats should have column entries"
        );
        let merged_bytes = b_merged.finish().expect("finish builder");

        // Read merged superfile
        let reader_merged =
            SuperfileReader::open(Bytes::from(merged_bytes)).expect("open merged reader");

        // Verify vector search works
        let vec_reader = reader_merged.vec().expect("get vector reader");
        let search_results = vec_reader
            .search(
                "emb",
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                10,
                4,
                100,
            )
            .await
            .expect("vector search");
        assert!(
            !search_results.is_empty(),
            "vector search should return results"
        );

        // Verify FTS search works
        let fts_reader = reader_merged.fts().expect("get fts reader");
        let fts_results = fts_reader
            .search("title", &["hello"], 10, BoolMode::Or)
            .await
            .expect("fts search");
        assert!(!fts_results.is_empty(), "fts search should return results");

        // Verify parquet query works
        let batch = reader_merged
            .get_record_batch(None)
            .expect("get_record_batch");
        assert_eq!(batch.num_rows(), 2);
    }

    #[test]
    fn build_from_readers_rejects_empty_readers_array() {
        let result = SuperfileBuilder::build_from_readers(&[]);
        assert!(result.is_err(), "should reject empty readers array");
    }

    fn empty_bitmap() -> Option<Arc<RoaringBitmap>> {
        None
    }

    #[test]
    fn build_from_readers_single_reader_produces_valid_superfile() {
        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            Some(default_tokenizer()),
        );
        let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
        let schema = b.opts.schema.clone();
        let batch = batch_two_rows(&schema);
        b.add_batch(&batch, &[]).expect("add_batch");
        let original_bytes = b.finish().expect("finish builder");

        let reader = SuperfileReader::open(Bytes::from(original_bytes.clone()))
            .expect("open superfile reader");

        let (merged_bytes, stats) =
            SuperfileBuilder::build_from_readers(&[(Arc::new(reader), empty_bitmap())])
                .expect("build_from_readers");

        // Verify result is a valid superfile
        assert_eq!(&merged_bytes[..4], b"PAR1");
        assert_eq!(&merged_bytes[merged_bytes.len() - 4..], b"PAR1");

        // Verify stats are correct
        assert_eq!(stats.n_docs, 2);
        assert_eq!(stats.id_min, 10);
        assert_eq!(stats.id_max, 11);
        assert!(stats.scalar_stats.cols.contains_key("doc_id"));
        assert!(stats.scalar_stats.cols.contains_key("title"));
        assert!(stats.scalar_stats.cols.contains_key("body"));

        // Verify data is preserved
        let merged_reader =
            SuperfileReader::open(Bytes::from(merged_bytes)).expect("open merged reader");
        let merged_batch = merged_reader
            .get_record_batch(None)
            .expect("get_record_batch");
        assert_eq!(merged_batch.num_rows(), 2);
    }

    #[test]
    fn build_from_readers_merges_multiple_readers_correctly() {
        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            Some(default_tokenizer()),
        );

        // Create first superfile
        let mut b1 = SuperfileBuilder::new(opts.clone()).expect("new SuperfileBuilder");
        let schema = b1.opts.schema.clone();
        let batch1 = batch_two_rows(&schema);
        b1.add_batch(&batch1, &[]).expect("add_batch");
        let bytes1 = b1.finish().expect("finish builder");

        // Create second superfile
        let mut b2 = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
        let ids2 = decimal128_ids(vec![20u64, 21]);
        let title2 = LargeStringArray::from(vec!["foo bar", "baz qux"]);
        let body2 = LargeStringArray::from(vec!["quux corge", "grault garply"]);
        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(ids2), Arc::new(title2), Arc::new(body2)],
        )
        .expect("build RecordBatch");
        b2.add_batch(&batch2, &[]).expect("add_batch");
        let bytes2 = b2.finish().expect("finish builder");

        let reader1 = SuperfileReader::open(Bytes::from(bytes1)).expect("open reader1");
        let reader2 = SuperfileReader::open(Bytes::from(bytes2)).expect("open reader2");

        let (merged_bytes, stats) = SuperfileBuilder::build_from_readers(&[
            (Arc::new(reader1), empty_bitmap()),
            (Arc::new(reader2), empty_bitmap()),
        ])
        .expect("build_from_readers");

        // Verify stats are correct
        assert_eq!(stats.n_docs, 4, "should have 4 total documents");
        assert_eq!(stats.id_min, 10, "id_min should be 10");
        assert_eq!(stats.id_max, 21, "id_max should be 21");
        assert_eq!(stats.scalar_stats.cols.len(), 3, "should have 3 columns");

        // Verify merged superfile
        let merged_reader =
            SuperfileReader::open(Bytes::from(merged_bytes)).expect("open merged reader");
        let merged_batch = merged_reader
            .get_record_batch(None)
            .expect("get_record_batch");

        // Should have 4 rows total (2 + 2)
        assert_eq!(merged_batch.num_rows(), 4);
    }

    #[test]
    fn build_from_readers_preserves_vectors_and_fts() {
        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![default_vector_config("emb", 7)],
            Some(default_tokenizer()),
        );

        // Create superfile with both FTS and vectors
        let mut b1 = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
        let schema = b1.opts.schema.clone();
        let batch = batch_two_rows(&schema);
        let mut v: Vec<f32> = vec![0.0; 32]; // 2 rows × 16 dim
        v[0] = 1.0;
        v[16 + 1] = 1.0;
        b1.add_batch(&batch, &[v.as_slice()]).expect("add_batch");
        let bytes1 = b1.finish().expect("finish builder");

        let reader = SuperfileReader::open(Bytes::from(bytes1)).expect("open reader");

        let (merged_bytes, stats) =
            SuperfileBuilder::build_from_readers(&[(Arc::new(reader), empty_bitmap())])
                .expect("build_from_readers");

        // Verify stats
        assert_eq!(stats.n_docs, 2);
        assert_eq!(stats.id_min, 10);
        assert_eq!(stats.id_max, 11);

        // Verify merged superfile has both FTS and vector indexes
        let merged_reader =
            SuperfileReader::open(Bytes::from(merged_bytes)).expect("open merged reader");

        // FTS should be present
        assert!(merged_reader.fts().is_some(), "FTS index should be present");

        // Vectors should be present
        assert!(
            merged_reader.vec().is_some(),
            "Vector index should be present"
        );
    }

    #[tokio::test]
    async fn build_from_readers_preserves_fts_search_functionality() {
        use crate::superfile::fts::reader::BoolMode;

        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            Some(default_tokenizer()),
        );

        // Create superfile with FTS
        let mut b = SuperfileBuilder::new(opts.clone()).expect("new SuperfileBuilder");
        let schema = b.opts.schema.clone();
        let batch = batch_two_rows(&schema);
        b.add_batch(&batch, &[]).expect("add_batch");
        let bytes = b.finish().expect("finish builder");

        let reader1 = SuperfileReader::open(Bytes::from(bytes)).expect("open reader");

        let mut b2 = SuperfileBuilder::new(opts.clone()).expect("new SuperfileBuilder");
        b2.add_batch(&batch, &[]).expect("add batch");
        let bytes = b2.finish().expect("finish builder");
        let reader2 = SuperfileReader::open(Bytes::from(bytes)).expect("open reader");

        // Build merged superfile
        let (merged_bytes, stats) = SuperfileBuilder::build_from_readers(&[
            (Arc::new(reader1), empty_bitmap()),
            (Arc::new(reader2), empty_bitmap()),
        ])
        .expect("build_from_readers");

        // Verify stats
        assert_eq!(stats.n_docs, 4, "should have 4 documents (2 + 2)");
        assert_eq!(stats.id_min, 10);
        assert_eq!(stats.id_max, 11);

        // Verify FTS search works on merged
        let merged_reader =
            SuperfileReader::open(Bytes::from(merged_bytes)).expect("open merged reader");
        let fts_reader_merged = merged_reader.fts().expect("get fts reader from merged");
        let results_merged = fts_reader_merged
            .search("title", &["hello"], 10, BoolMode::Or)
            .await
            .expect("search merged");
        assert_eq!(results_merged.len(), 2);
    }

    #[test]
    fn build_from_readers_three_superfiles() {
        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            Some(default_tokenizer()),
        );

        // Create three superfiles
        let mut bytes_list = Vec::new();
        for base_id in [10u64, 20u64, 30u64] {
            let mut b = SuperfileBuilder::new(opts.clone()).expect("new SuperfileBuilder");
            let schema = b.opts.schema.clone();
            let ids = decimal128_ids(vec![base_id, base_id + 1]);
            let title = LargeStringArray::from(vec!["foo", "bar"]);
            let body = LargeStringArray::from(vec!["baz", "qux"]);
            let batch =
                RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(title), Arc::new(body)])
                    .expect("build RecordBatch");
            b.add_batch(&batch, &[]).expect("add_batch");
            bytes_list.push(b.finish().expect("finish builder"));
        }

        // Create readers
        let readers: Vec<_> = bytes_list
            .iter()
            .map(|b| {
                (
                    Arc::new(SuperfileReader::open(Bytes::from(b.clone())).expect("open reader")),
                    empty_bitmap(),
                )
            })
            .collect();

        // Merge all three
        let (merged_bytes, stats) =
            SuperfileBuilder::build_from_readers(&readers).expect("build_from_readers");

        // Verify stats
        assert_eq!(stats.n_docs, 6, "should have 6 total documents");
        assert_eq!(stats.id_min, 10, "id_min should be 10");
        assert_eq!(stats.id_max, 31, "id_max should be 31");

        // Verify merged result has all rows
        let merged_reader =
            SuperfileReader::open(Bytes::from(merged_bytes)).expect("open merged reader");
        let merged_batch = merged_reader
            .get_record_batch(None)
            .expect("get_record_batch");

        // Should have 6 rows total (2 + 2 + 2)
        assert_eq!(merged_batch.num_rows(), 6);
    }

    #[tokio::test]
    async fn build_from_readers_with_only_vectors_and_search() {
        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![],
            vec![default_vector_config("emb", 7)],
            None,
        );

        // Create first superfile with only vectors (no FTS)
        let mut b1 = SuperfileBuilder::new(opts.clone()).expect("new SuperfileBuilder");
        let schema = b1.opts.schema.clone();
        let batch1 = batch_two_rows(&schema);
        let mut v1: Vec<f32> = vec![0.0; 32]; // 2 rows × 16 dim
        v1[0] = 1.0;
        v1[16 + 1] = 1.0;
        b1.add_batch(&batch1, &[v1.as_slice()]).expect("add_batch");
        let bytes1 = b1.finish().expect("finish builder");

        // Create second superfile with different vectors
        let mut b2 = SuperfileBuilder::new(opts.clone()).expect("new SuperfileBuilder");
        let ids2 = decimal128_ids(vec![20u64, 21]);
        let title2 = LargeStringArray::from(vec!["foo bar", "baz qux"]);
        let body2 = LargeStringArray::from(vec!["quux corge", "grault garply"]);
        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(ids2), Arc::new(title2), Arc::new(body2)],
        )
        .expect("build RecordBatch");
        let mut v2: Vec<f32> = vec![0.0; 32];
        v2[1] = 1.0;
        v2[16 + 2] = 1.0;
        b2.add_batch(&batch2, &[v2.as_slice()]).expect("add_batch");
        let bytes2 = b2.finish().expect("finish builder");

        let reader1 = SuperfileReader::open(Bytes::from(bytes1)).expect("open reader1");
        let reader2 = SuperfileReader::open(Bytes::from(bytes2)).expect("open reader2");

        // Merge both readers
        let (merged_bytes, stats) = SuperfileBuilder::build_from_readers(&[
            (Arc::new(reader1), empty_bitmap()),
            (Arc::new(reader2), empty_bitmap()),
        ])
        .expect("build_from_readers");

        // Verify stats
        assert_eq!(stats.n_docs, 4, "should have 4 total documents");
        assert_eq!(stats.id_min, 10, "id_min should be 10");
        assert_eq!(stats.id_max, 21, "id_max should be 21");

        // Verify merged superfile
        let merged_reader =
            SuperfileReader::open(Bytes::from(merged_bytes)).expect("open merged reader");

        // Should have vectors but no FTS
        assert!(merged_reader.vec().is_some(), "should have vector index");
        assert!(merged_reader.fts().is_none(), "should not have FTS index");

        let batch = merged_reader
            .get_record_batch(None)
            .expect("get_record_batch");
        assert_eq!(batch.num_rows(), 4, "should have 4 rows (2 + 2)");

        // Perform vector search on merged data
        let vec_reader = merged_reader.vec().expect("get vector reader");
        let query = [
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let search_results = vec_reader
            .search("emb", &query, 10, 4, 100)
            .await
            .expect("vector search");

        // Should return exactly 4 results (all vectors from both superfiles are returned)
        assert_eq!(
            search_results.len(),
            4,
            "vector search should return all 4 vectors from merged superfiles"
        );
    }

    #[test]
    fn build_from_readers_filters_deleted_documents() {
        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            Some(default_tokenizer()),
        );

        // Create first superfile with 2 rows (indices 0, 1)
        let mut b1 = SuperfileBuilder::new(opts.clone()).expect("new SuperfileBuilder");
        let schema = b1.opts.schema.clone();
        let batch1 = batch_two_rows(&schema);
        b1.add_batch(&batch1, &[]).expect("add_batch");
        let bytes1 = b1.finish().expect("finish builder");

        // Create second superfile with 2 rows (indices 0, 1)
        let mut b2 = SuperfileBuilder::new(opts.clone()).expect("new SuperfileBuilder");
        let ids2 = decimal128_ids(vec![20u64, 21]);
        let title2 = LargeStringArray::from(vec!["foo bar", "baz qux"]);
        let body2 = LargeStringArray::from(vec!["quux corge", "grault garply"]);
        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(ids2), Arc::new(title2), Arc::new(body2)],
        )
        .expect("build RecordBatch");
        b2.add_batch(&batch2, &[]).expect("add_batch");
        let bytes2 = b2.finish().expect("finish builder");

        let reader1 = SuperfileReader::open(Bytes::from(bytes1)).expect("open reader1");
        let reader2 = SuperfileReader::open(Bytes::from(bytes2)).expect("open reader2");

        // Create bitmaps to mark deleted rows
        // For reader1: mark row 0 as deleted (keep row 1, id=11)
        let mut bitmap1 = RoaringBitmap::new();
        bitmap1.insert(0);

        // For reader2: mark row 1 as deleted (keep row 0, id=20)
        let mut bitmap2 = RoaringBitmap::new();
        bitmap2.insert(1);

        // Merge with deletion bitmaps
        let (merged_bytes, stats) = SuperfileBuilder::build_from_readers(&[
            (Arc::new(reader1), Some(Arc::new(bitmap1))),
            (Arc::new(reader2), Some(Arc::new(bitmap2))),
        ])
        .expect("build_from_readers");

        // Verify stats: should have 2 rows after deletion (id_min=11 from reader1, id_max=20 from reader2)
        assert_eq!(stats.n_docs, 2, "should have 2 documents after filtering");
        assert_eq!(stats.id_min, 11, "id_min should be 11 (from reader1 row 1)");
        assert_eq!(stats.id_max, 20, "id_max should be 20 (from reader2 row 0)");

        // Verify merged superfile has only 2 rows (1 from each superfile after deletion)
        let merged_reader =
            SuperfileReader::open(Bytes::from(merged_bytes)).expect("open merged reader");
        let merged_batch = merged_reader
            .get_record_batch(None)
            .expect("get_record_batch");

        // Should have exactly 2 rows: row 1 from reader1 + row 0 from reader2
        assert_eq!(
            merged_batch.num_rows(),
            2,
            "merged superfile should have 2 rows after filtering deleted documents"
        );
    }

    #[test]
    fn build_from_readers_validates_scalar_stats_min_max_single_reader() {
        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            Some(default_tokenizer()),
        );
        let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
        let schema = b.opts.schema.clone();
        let batch = batch_two_rows(&schema);
        b.add_batch(&batch, &[]).expect("add_batch");
        let bytes = b.finish().expect("finish builder");

        let reader = SuperfileReader::open(Bytes::from(bytes)).expect("open reader");
        let (_, stats) =
            SuperfileBuilder::build_from_readers(&[(Arc::new(reader), empty_bitmap())])
                .expect("build_from_readers");

        // Verify doc_id min/max (10, 11)
        let (doc_id_min_arr, doc_id_max_arr) = stats
            .scalar_stats
            .cols
            .get("doc_id")
            .expect("doc_id column");
        let doc_id_min = doc_id_min_arr
            .as_ref()
            .as_any()
            .downcast_ref::<arrow_array::Decimal128Array>()
            .expect("downcast to Decimal128")
            .value(0);
        let doc_id_max = doc_id_max_arr
            .as_ref()
            .as_any()
            .downcast_ref::<arrow_array::Decimal128Array>()
            .expect("downcast to Decimal128")
            .value(0);
        assert_eq!(doc_id_min, 10, "doc_id min should be 10");
        assert_eq!(doc_id_max, 11, "doc_id max should be 11");

        // Verify title min/max (from batch_two_rows: ["hello world", "rust async"])
        let (title_min_arr, title_max_arr) =
            stats.scalar_stats.cols.get("title").expect("title column");
        let title_min = title_min_arr
            .as_ref()
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("downcast to LargeStringArray")
            .value(0);
        let title_max = title_max_arr
            .as_ref()
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("downcast to LargeStringArray")
            .value(0);
        assert_eq!(
            title_min, "hello world",
            "title min should be 'hello world'"
        );
        assert_eq!(title_max, "rust async", "title max should be 'rust async'");

        // Verify body min/max (from batch_two_rows: ["foo bar", "baz quux"])
        let (body_min_arr, body_max_arr) =
            stats.scalar_stats.cols.get("body").expect("body column");
        let body_min = body_min_arr
            .as_ref()
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("downcast to LargeStringArray")
            .value(0);
        let body_max = body_max_arr
            .as_ref()
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("downcast to LargeStringArray")
            .value(0);
        assert_eq!(body_min, "baz quux", "body min should be 'baz quux'");
        assert_eq!(body_max, "foo bar", "body max should be 'foo bar'");
    }

    #[test]
    fn build_from_readers_validates_scalar_stats_across_multiple_readers() {
        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            Some(default_tokenizer()),
        );

        // Create first superfile with ids 10, 11, titles ["hello world", "rust async"]
        let mut b1 = SuperfileBuilder::new(opts.clone()).expect("new SuperfileBuilder");
        let schema = b1.opts.schema.clone();
        let batch1 = batch_two_rows(&schema);
        b1.add_batch(&batch1, &[]).expect("add_batch");
        let bytes1 = b1.finish().expect("finish builder");

        // Create second superfile with ids 20, 21, titles ["alpha", "zeta"]
        let mut b2 = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
        let ids2 = decimal128_ids(vec![20u64, 21]);
        let title2 = LargeStringArray::from(vec!["alpha", "zeta"]);
        let body2 = LargeStringArray::from(vec!["aaa", "zzz"]);
        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(ids2), Arc::new(title2), Arc::new(body2)],
        )
        .expect("build RecordBatch");
        b2.add_batch(&batch2, &[]).expect("add_batch");
        let bytes2 = b2.finish().expect("finish builder");

        let reader1 = SuperfileReader::open(Bytes::from(bytes1)).expect("open reader1");
        let reader2 = SuperfileReader::open(Bytes::from(bytes2)).expect("open reader2");

        let (_, stats) = SuperfileBuilder::build_from_readers(&[
            (Arc::new(reader1), empty_bitmap()),
            (Arc::new(reader2), empty_bitmap()),
        ])
        .expect("build_from_readers");

        // Verify doc_id: min should be 10, max should be 21 (merged from both readers)
        let (doc_id_min_arr, doc_id_max_arr) = stats
            .scalar_stats
            .cols
            .get("doc_id")
            .expect("doc_id column");
        let doc_id_min = doc_id_min_arr
            .as_ref()
            .as_any()
            .downcast_ref::<arrow_array::Decimal128Array>()
            .expect("downcast to Decimal128")
            .value(0);
        let doc_id_max = doc_id_max_arr
            .as_ref()
            .as_any()
            .downcast_ref::<arrow_array::Decimal128Array>()
            .expect("downcast to Decimal128")
            .value(0);
        assert_eq!(doc_id_min, 10, "merged doc_id min should be 10");
        assert_eq!(doc_id_max, 21, "merged doc_id max should be 21");

        // Verify title: min should be "alpha", max should be "zeta" (lexicographically from both readers)
        let (title_min_arr, title_max_arr) =
            stats.scalar_stats.cols.get("title").expect("title column");
        let title_min = title_min_arr
            .as_ref()
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("downcast to LargeStringArray")
            .value(0);
        let title_max = title_max_arr
            .as_ref()
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("downcast to LargeStringArray")
            .value(0);
        assert_eq!(title_min, "alpha", "merged title min should be 'alpha'");
        assert_eq!(title_max, "zeta", "merged title max should be 'zeta'");

        // Verify body: min should be "aaa", max should be "zzz" (lexicographically from both readers)
        let (body_min_arr, body_max_arr) =
            stats.scalar_stats.cols.get("body").expect("body column");
        let body_min = body_min_arr
            .as_ref()
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("downcast to LargeStringArray")
            .value(0);
        let body_max = body_max_arr
            .as_ref()
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("downcast to LargeStringArray")
            .value(0);
        assert_eq!(body_min, "aaa", "merged body min should be 'aaa'");
        assert_eq!(body_max, "zzz", "merged body max should be 'zzz'");
    }

    #[test]
    fn build_from_readers_validates_scalar_stats_with_string_columns() {
        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            Some(default_tokenizer()),
        );

        // Create superfile with specific string values to validate min/max ordering
        let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
        let schema = b.opts.schema.clone();
        let ids = decimal128_ids(vec![1u64, 2]);
        let titles = LargeStringArray::from(vec!["zebra", "apple"]);
        let bodies = LargeStringArray::from(vec!["xyz", "abc"]);
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(ids), Arc::new(titles), Arc::new(bodies)],
        )
        .expect("build RecordBatch");
        b.add_batch(&batch, &[]).expect("add_batch");
        let bytes = b.finish().expect("finish builder");

        let reader = SuperfileReader::open(Bytes::from(bytes)).expect("open reader");
        let (_, stats) =
            SuperfileBuilder::build_from_readers(&[(Arc::new(reader), empty_bitmap())])
                .expect("build_from_readers");

        // Verify title min/max (values: ["zebra", "apple"] => min="apple", max="zebra")
        let (title_min_arr, title_max_arr) =
            stats.scalar_stats.cols.get("title").expect("title column");
        let title_min = title_min_arr
            .as_ref()
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("downcast to LargeStringArray")
            .value(0);
        let title_max = title_max_arr
            .as_ref()
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("downcast to LargeStringArray")
            .value(0);
        assert_eq!(title_min, "apple", "title min should be 'apple'");
        assert_eq!(title_max, "zebra", "title max should be 'zebra'");

        // Verify body min/max (values: ["xyz", "abc"] => min="abc", max="xyz")
        let (body_min_arr, body_max_arr) =
            stats.scalar_stats.cols.get("body").expect("body column");
        let body_min = body_min_arr
            .as_ref()
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("downcast to LargeStringArray")
            .value(0);
        let body_max = body_max_arr
            .as_ref()
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("downcast to LargeStringArray")
            .value(0);
        assert_eq!(body_min, "abc", "body min should be 'abc'");
        assert_eq!(body_max, "xyz", "body max should be 'xyz'");
    }
}
