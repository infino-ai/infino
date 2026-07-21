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
use std::{
    collections::{HashMap, HashSet},
    fmt,
    io::{BufReader, BufWriter, Cursor, Error, Seek, SeekFrom, Write},
    sync::Arc,
};

use arrow::compute::{concat_batches, take};
use arrow_array::{Array, ArrayRef, Decimal128Array, LargeStringArray, RecordBatch, UInt32Array};
use arrow_schema::{DataType, Schema};
use parquet::basic::{Compression, ZstdLevel};
use roaring::RoaringBitmap;
use tempfile::tempfile;

pub use crate::superfile::vector::builder::VectorConfig;
use crate::superfile::{
    BuildError, SuperfileReader,
    format::{
        self,
        footer::{encode_parquet_body, splice_index_blobs, splice_index_streams_to},
        kv,
    },
    fts::{
        builder::FtsBuilder,
        tokenize::{AsciiLowerTokenizer, Tokenizer},
    },
    stats::SuperfileStats,
    vector::{
        builder::{
            MultiCellSubsectionSource, VectorBuilder, build_merged_subsection_from_materialized,
            finish_multi_cell_blob_to,
        },
        cell_posting::{CellPostingBuilder, MaterializedIvfRow},
        distance::Metric,
        ivf_merge::{
            MergedIvfSubsection, Sq8IvfMergeInput, merge_sq8_ivf_subsections,
            merge_sq8_ivf_subsections_from_parsed, stable_ids_in_merged_local_order,
        },
        layout::VectorLayout,
        reader::{ColumnReader, VectorReader},
        rerank_codec::RerankCodec,
    },
};

/// Per-column FTS configuration. The `column` must exist in
/// `BuilderOptions.schema` and be `LargeUtf8`.
#[derive(Clone)]
pub struct FtsConfig {
    pub column: String,
    /// Record token positions for this column, enabling exact phrase
    /// queries against it. Off by default: positions roughly double
    /// the column's FTS index footprint, so the cost is a per-column
    /// opt-in. Columns without positions answer phrase queries with a
    /// typed error, never a silent bag-of-words fallback.
    pub positions: bool,
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
    /// At this layer (superfile), a vector entry is a
    /// **logical index name only** — the f32 slices are passed
    /// separately to `add_batch(scalar_batch, &[&[f32]])` and
    /// the name lives in the legacy-named `inf.vec.columns` KV metadata, not
    /// in the Parquet schema. The "must NOT collide with a
    /// column in `schema`" rule is the format-layer
    /// disambiguation that keeps vector names out of the
    /// Parquet column namespace.
    ///
    /// At the supertable ingest boundary the constraint reads
    /// differently: there, vectors arrive as schema fields
    /// (typed `FixedSizeList<Float32, dim>`). The supertable's
    /// `vector_split` strips them at commit time and forwards
    /// `(scalar_only_batch, &[&[f32]])` down to this builder
    /// — so by the time a `BuilderOptions` reaches us, those vectors
    /// have already left the scalar schema and are index payloads. The
    /// supertable enforces the same cross-list uniqueness
    /// against its FTS columns at construction.
    ///
    /// To run both FTS and vector against the same business
    /// concept (e.g. semantic + lexical "description"
    /// search), model it as one stored
    /// `LargeUtf8` text column plus one ingest-time `FixedSizeList<f32>`
    /// vector payload. Hybrid retrieval
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
    /// Embedded vector blob layout. Default IVF.
    pub(crate) vector_layout: VectorLayout,
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
                ZstdLevel::try_new(3).expect("zstd level 3 is in the valid 1..=22 range"),
            ),
            id_page_size_limit: DEFAULT_ID_PAGE_SIZE_LIMIT,
            vector_layout: VectorLayout::Ivf,
        }
    }

    pub(crate) fn with_vector_layout(mut self, layout: VectorLayout) -> Self {
        self.vector_layout = layout;
        self
    }

    /// Stamp caller-supplied global centroids onto every vector column so
    /// the IVF build partitions against them instead of training local
    /// k-means. See [`VectorConfig::provided_centroids`]. `None` is a no-op
    /// (local k-means, the default).
    pub(crate) fn with_vector_centroids(
        mut self,
        centroids: Option<std::sync::Arc<[f32]>>,
    ) -> Self {
        for vc in &mut self.vector_columns {
            vc.provided_centroids = centroids.clone();
        }
        self
    }

    pub fn new_from_reader(reader: &SuperfileReader) -> Self {
        // TODO: Fetch tokenizer from reader. Not possible at the moment because we don't
        // store the tokenizer in the reader. Should work for now because we only have AsciiLowerTokenizer.
        let tokenizer = Arc::new(AsciiLowerTokenizer);
        let fts_columns = if let Some(fts) = &reader.fts() {
            fts.fts_columns_config()
                .map(|c| FtsConfig {
                    column: c.name.clone(),
                    positions: c.positions,
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let (vector_columns, vector_layout) = if let Some(vec) = &reader.vec() {
            if vec.is_multi_cell() {
                // One logical column; cell IVFs live in the v2 cell directory.
                let v = vec
                    .vector_columns_config()
                    .next()
                    .expect("multi-cell reader has at least one cell ColumnReader");
                (
                    vec![
                        VectorConfig::new(
                            v.name.clone(),
                            v.dim,
                            v.n_cent as usize,
                            v.rot_seed,
                            v.metric,
                        )
                        .with_rerank_codec(v.rerank_codec),
                    ],
                    VectorLayout::MultiCellIvf,
                )
            } else {
                (
                    vec.vector_columns_config()
                        .map(|v| {
                            VectorConfig::new(
                                v.name.clone(),
                                v.dim,
                                v.n_cent as usize,
                                v.rot_seed,
                                v.metric,
                            )
                            .with_rerank_codec(v.rerank_codec)
                        })
                        .collect::<Vec<_>>(),
                    VectorLayout::Ivf,
                )
            }
        } else {
            (Vec::new(), VectorLayout::Ivf)
        };

        BuilderOptions::new(
            reader.schema().clone(),
            reader.id_column(),
            fts_columns,
            vector_columns,
            Some(tokenizer),
        )
        .with_vector_layout(vector_layout)
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

impl fmt::Debug for SuperfileBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
    cell_posting_builder: Option<CellPostingBuilder>,
    /// Pre-built cell-IVF subsections for [`VectorLayout::MultiCellIvf`].
    /// When set, `finish` assembles a v2 multi-cell vector blob instead of
    /// running the streaming IVF builder.
    prebuilt_multi_cell: Option<Vec<(u32, MergedIvfSubsection)>>,
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
        let mut seen_logical: HashSet<&str> = HashSet::new();
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
                fb.register_column(fc.column.clone(), fc.positions)?;
            }
            Some(fb)
        };

        let (vec_builder, cell_posting_builder) = if opts.vector_columns.is_empty() {
            (None, None)
        } else if opts.vector_layout == VectorLayout::CellPosting {
            let mut cb = CellPostingBuilder::new();
            for vc in &opts.vector_columns {
                cb.register_column(vc.clone())?;
            }
            (None, Some(cb))
        } else if opts.vector_layout == VectorLayout::MultiCellIvf {
            // Multi-cell blobs are assembled from prebuilt cell IVFs at
            // finish time; no streaming VectorBuilder is needed.
            (None, None)
        } else {
            let mut vb = VectorBuilder::new();
            for vc in &opts.vector_columns {
                vb.register_column(vc.clone())?;
            }
            (Some(vb), None)
        };

        Ok(Self {
            opts,
            fts_col_idxs,
            batches: Vec::new(),
            fts_builder,
            vec_builder,
            cell_posting_builder,
            prebuilt_multi_cell: None,
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
            return Err(BuildError::BatchSchemaMismatch {
                batch: batch.schema().to_string(),
                builder: self.opts.schema.to_string(),
            });
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
        self.index_fts_batch(batch, n_rows)?;

        // Route vectors.
        if let Some(vb) = self.vec_builder.as_mut() {
            for (i, vc) in self.opts.vector_columns.iter().enumerate() {
                let dim = vc.dim;
                for row in 0..(n_rows as usize) {
                    let start = row * dim;
                    vb.add(i as u32, &vectors[i][start..start + dim])?;
                }
            }
        } else if let Some(cb) = self.cell_posting_builder.as_mut() {
            for (i, vc) in self.opts.vector_columns.iter().enumerate() {
                let dim = vc.dim;
                for row in 0..(n_rows as usize) {
                    let start = row * dim;
                    cb.add(i as u32, &vectors[i][start..start + dim])?;
                }
            }
        }

        self.next_local_doc_id += n_rows;
        self.batches.push(batch.clone());
        Ok(())
    }

    /// Append a scalar-only batch (ids without vector payloads). Used when the
    /// vector blob is supplied separately via a prebuilt IVF subsection.
    pub(crate) fn add_batch_ids_only(&mut self, batch: &RecordBatch) -> Result<(), BuildError> {
        if batch.schema().fields() != self.opts.schema.fields() {
            return Err(BuildError::BatchSchemaMismatch {
                batch: batch.schema().to_string(),
                builder: self.opts.schema.to_string(),
            });
        }
        // Sq8 / multi-cell merge paths supply the vector blob out of band, but
        // any FTS columns still need to be indexed from the scalar batch —
        // otherwise `finish` emits an empty FTS blob against a non-empty
        // Parquet body (silent query corruption).
        let n_rows = batch.num_rows() as u32;
        self.index_fts_batch(batch, n_rows)?;
        self.next_local_doc_id += n_rows;
        self.batches.push(batch.clone());
        Ok(())
    }

    /// Index FTS text columns from `batch` starting at `self.next_local_doc_id`.
    /// Null cells index as empty strings so doc_lengths stay aligned with Parquet.
    fn index_fts_batch(&mut self, batch: &RecordBatch, n_rows: u32) -> Result<(), BuildError> {
        let Some(fb) = self.fts_builder.as_mut() else {
            return Ok(());
        };
        for (col_id, &schema_idx) in self.fts_col_idxs.iter().enumerate() {
            let arr = batch.column(schema_idx);
            let strs = arr
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("schema validated as LargeUtf8");
            for row in 0..(n_rows as usize) {
                let local_doc_id = self.next_local_doc_id + row as u32;
                let text = if strs.is_null(row) {
                    ""
                } else {
                    strs.value(row)
                };
                fb.add_doc(col_id as u32, local_doc_id, text)?;
            }
        }
        Ok(())
    }

    /// Inject a byte-spliced IVF subsection for compaction merge.
    pub(crate) fn set_prebuilt_ivf_subsection(
        &mut self,
        column_id: u32,
        subsection: MergedIvfSubsection,
    ) -> Result<(), BuildError> {
        let vb = self
            .vec_builder
            .as_mut()
            .ok_or_else(|| BuildError::VectorSchemaMismatch("no vector builder".into()))?;
        vb.set_prebuilt_subsection(column_id, subsection)?;
        Ok(())
    }

    /// Inject many complete cell-IVF subsections for a multi-cell packed
    /// superfile ([`VectorLayout::MultiCellIvf`]). Cells must be unique and
    /// will be sorted by `cell_id` at finish.
    pub(crate) fn set_prebuilt_multi_cell_ivfs(
        &mut self,
        mut cells: Vec<(u32, MergedIvfSubsection)>,
    ) -> Result<(), BuildError> {
        if self.opts.vector_layout != VectorLayout::MultiCellIvf {
            return Err(BuildError::VectorSchemaMismatch(
                "set_prebuilt_multi_cell_ivfs requires MultiCellIvf layout".into(),
            ));
        }
        if cells.is_empty() {
            return Err(BuildError::VectorSchemaMismatch(
                "multi-cell pack requires at least one cell IVF".into(),
            ));
        }
        let configured_codec = self
            .opts
            .vector_columns
            .first()
            .ok_or(BuildError::VectorReadError)?
            .rerank_codec;
        let expected_codec = if configured_codec.is_sq8_residual_family() {
            configured_codec
        } else {
            RerankCodec::Sq8Residual
        };
        if cells
            .iter()
            .any(|(_, subsection)| subsection.rerank_codec != expected_codec)
        {
            return Err(BuildError::VectorSchemaMismatch(
                "multi-cell subsection codec does not match builder options".into(),
            ));
        }
        cells.sort_unstable_by_key(|(cell, _)| *cell);
        for w in cells.windows(2) {
            if w[0].0 == w[1].0 {
                return Err(BuildError::VectorSchemaMismatch(format!(
                    "duplicate cell_id {} in multi-cell pack",
                    w[0].0
                )));
            }
        }
        self.prebuilt_multi_cell = Some(cells);
        Ok(())
    }

    /// Merge Sq8 IVF superfiles without fp32 corpus decode — byte-splices
    /// per-cluster IVF blocks and remaps doc ids.
    pub fn build_from_sq8_ivf_readers(
        readers: &[(Arc<SuperfileReader>, Option<Arc<RoaringBitmap>>)],
    ) -> Result<(Vec<u8>, SuperfileStats), BuildError> {
        let first = readers.first().ok_or(BuildError::BatchReadError)?;
        let builder_opts = BuilderOptions::new_from_reader(&first.0);
        let mut superfile_builder = SuperfileBuilder::new(builder_opts)?;

        let vec_col = first
            .0
            .vec()
            .and_then(|v| v.vector_columns_config().next())
            .ok_or_else(|| BuildError::VectorReadError)?;
        if !vec_col.rerank_codec.is_sq8_residual_family() {
            return Err(BuildError::VectorReadError);
        }
        let column = vec_col.name.clone();

        let mut stats_collector = Vec::with_capacity(readers.len());
        let mut merge_inputs: Vec<(&VectorReader, String, u32)> = Vec::with_capacity(readers.len());
        let mut local_base = 0u32;

        for (idx, (reader, deleted)) in readers.iter().enumerate() {
            // Compaction opens its inputs eagerly (see
            // `query::dispatch::open_compaction_input`), so `get_record_batch`
            // resolves off resident bytes. A lazy reader here is a caller bug,
            // not something to paper over — surface it with context.
            let record_batch = reader.get_record_batch(deleted.clone()).map_err(|e| {
                BuildError::Io(Error::other(format!(
                    "sq8 merge input {idx}: read RecordBatch failed (n_docs={}, eager={}): {e}",
                    reader.n_docs(),
                    reader.parquet_bytes().is_some(),
                )))
            })?;
            let stats = SuperfileStats::try_compute_from_record_batch(&record_batch)?;
            stats_collector.push(stats);

            let v = reader.vec().ok_or(BuildError::VectorReadError)?;
            merge_inputs.push((v, column.clone(), local_base));

            superfile_builder.add_batch_ids_only(&record_batch)?;
            local_base += record_batch.num_rows() as u32;
        }

        let merge_refs: Vec<(&VectorReader, &str, u32)> = merge_inputs
            .iter()
            .map(|(v, col, off)| (*v, col.as_str(), *off))
            .collect();
        let merged_sub = merge_sq8_ivf_subsections(&merge_refs)?;
        superfile_builder.set_prebuilt_ivf_subsection(0, merged_sub)?;

        let bytes = superfile_builder.finish()?;
        let stats = SuperfileStats::from_children(stats_collector.as_slice());
        Ok((bytes, stats))
    }

    /// Merge multi-cell (v2) Sq8 IVF superfiles **per global cell id**, then
    /// repack into one multi-cell output. Never flattens different cells into
    /// one IVF. Parquet `_id` rows follow cell-directory order (same as drain).
    ///
    /// Tombstones (file-local doc ids) drop rows before the per-cell rebuild;
    /// empty tombstones use the byte-splice path.
    pub fn build_from_multi_cell_sq8_ivf_readers(
        readers: &[(Arc<SuperfileReader>, Option<Arc<RoaringBitmap>>)],
    ) -> Result<(Vec<u8>, SuperfileStats), BuildError> {
        let first = readers.first().ok_or(BuildError::BatchReadError)?;
        let builder_opts = BuilderOptions::new_from_reader(&first.0);
        if builder_opts.vector_layout != VectorLayout::MultiCellIvf {
            return Err(BuildError::VectorSchemaMismatch(
                "build_from_multi_cell_sq8_ivf_readers requires multi-cell inputs".into(),
            ));
        }
        let scalar_schema = builder_opts.schema.clone();
        let id_column = builder_opts.id_column.clone();
        let vec_cfg = builder_opts
            .vector_columns
            .first()
            .cloned()
            .ok_or(BuildError::VectorReadError)?;
        let mut superfile_builder = SuperfileBuilder::new(builder_opts)?;

        let any_tombstones = readers
            .iter()
            .any(|(_, deleted)| deleted.as_ref().is_some_and(|b| !b.is_empty()));

        let mut stats_collector = Vec::with_capacity(readers.len());
        let mut scalar_batches = Vec::with_capacity(readers.len());
        for (idx, (reader, deleted)) in readers.iter().enumerate() {
            let record_batch = reader.get_record_batch(deleted.clone()).map_err(|e| {
                BuildError::Io(Error::other(format!(
                    "multi-cell merge input {idx}: read RecordBatch failed: {e}"
                )))
            })?;
            stats_collector.push(SuperfileStats::try_compute_from_record_batch(
                &record_batch,
            )?);
            let v = reader.vec().ok_or(BuildError::VectorReadError)?;
            if !v.is_multi_cell() {
                return Err(BuildError::VectorSchemaMismatch(
                    "build_from_multi_cell_sq8_ivf_readers requires multi-cell inputs".into(),
                ));
            }
            scalar_batches.push(record_batch);
        }

        let mut packed_cells: Vec<(u32, MergedIvfSubsection)> = Vec::new();
        let mut all_stable_ids: Vec<i128> = Vec::new();

        if any_tombstones {
            // Materialize → filter by file-local tombstone id → rebuild per cell.
            // Also track the max fine-cluster count seen per cell so rebuilds
            // keep the source IVF width (empty clusters stay empty).
            let mut by_cell: HashMap<u32, (usize, Vec<MaterializedIvfRow>)> = HashMap::new();
            for (reader, deleted) in readers {
                let v = reader.vec().ok_or(BuildError::VectorReadError)?;
                let mut file_doc_base = 0u32;
                let cell_cols: Vec<&ColumnReader> = v.vector_columns_config().collect();
                for (ci, &cell_id) in v.packed_cell_ids().iter().enumerate() {
                    let col = cell_cols.get(ci).ok_or(BuildError::VectorReadError)?;
                    let mut rows = v.materialized_cell_rows_at(ci)?;
                    if let Some(deny) = deleted.as_ref() {
                        rows.retain(|r| !deny.contains(file_doc_base + r.local_doc_id));
                    }
                    file_doc_base = file_doc_base.saturating_add(col.n_docs);
                    if rows.is_empty() {
                        continue;
                    }
                    let entry = by_cell.entry(cell_id).or_insert_with(|| (0, Vec::new()));
                    entry.0 = entry.0.max(col.n_cent as usize);
                    entry.1.extend(rows);
                }
            }

            let mut cell_ids: Vec<u32> = by_cell.keys().copied().collect();
            cell_ids.sort_unstable();
            for cell_id in cell_ids {
                let (n_cent, mut rows) = by_cell.remove(&cell_id).expect("cell present");
                for (i, row) in rows.iter_mut().enumerate() {
                    row.local_doc_id = i as u32;
                }
                let stable_ids: Vec<i128> = rows.iter().map(|r| r.stable_id).collect();
                let mut cfg = vec_cfg.clone();
                cfg.n_cent = n_cent.max(1);
                let merged = build_merged_subsection_from_materialized(cfg, rows)?;
                if stable_ids.len() != merged.n_docs as usize {
                    return Err(BuildError::VectorSchemaMismatch(format!(
                        "cell {cell_id}: stable_ids len {} != merged n_docs {}",
                        stable_ids.len(),
                        merged.n_docs
                    )));
                }
                all_stable_ids.extend_from_slice(&stable_ids);
                packed_cells.push((cell_id, merged));
            }
        } else {
            // Track each cell fragment's source `(reader, column-slot)` next to
            // its parsed merge input: fragments that agree on fine `n_cent`
            // byte-splice, disagreeing ones re-materialize from those sources.
            let mut by_cell: HashMap<u32, Vec<(usize, usize, Sq8IvfMergeInput)>> = HashMap::new();
            for (reader_idx, (reader, _)) in readers.iter().enumerate() {
                let v = reader.vec().ok_or(BuildError::VectorReadError)?;
                for (ci, &cell_id) in v.packed_cell_ids().iter().enumerate() {
                    let inp = v.sq8_ivf_merge_input_at(ci, 0)?;
                    by_cell
                        .entry(cell_id)
                        .or_default()
                        .push((reader_idx, ci, inp));
                }
            }

            let mut cell_ids: Vec<u32> = by_cell.keys().copied().collect();
            cell_ids.sort_unstable();
            for cell_id in cell_ids {
                let sources = by_cell.remove(&cell_id).expect("cell present");
                let same_shape = sources
                    .windows(2)
                    .all(|pair| pair[0].2.n_cent == pair[1].2.n_cent);
                if same_shape {
                    let mut inputs: Vec<Sq8IvfMergeInput> =
                        sources.into_iter().map(|(_, _, inp)| inp).collect();
                    let mut doc_base = 0u32;
                    for inp in &mut inputs {
                        inp.doc_id_offset = doc_base;
                        doc_base = doc_base.saturating_add(inp.n_docs);
                    }
                    let merged = merge_sq8_ivf_subsections_from_parsed(&inputs)?;
                    let cell_ids_col = stable_ids_in_merged_local_order(&inputs)?;
                    if cell_ids_col.len() != merged.n_docs as usize {
                        return Err(BuildError::VectorSchemaMismatch(format!(
                            "cell {cell_id}: stable_ids len {} != merged n_docs {}",
                            cell_ids_col.len(),
                            merged.n_docs
                        )));
                    }
                    all_stable_ids.extend_from_slice(&cell_ids_col);
                    packed_cells.push((cell_id, merged));
                    continue;
                }
                // Same cell, different fine `n_cent` (a small delta drain
                // merging into a larger base): byte-splice is positional per
                // cluster, so rebuild this cell from materialized rows at the
                // widest source width — same path the tombstone branch uses.
                let n_cent = sources
                    .iter()
                    .map(|(_, _, inp)| inp.n_cent)
                    .max()
                    .unwrap_or(1);
                let mut rows: Vec<MaterializedIvfRow> = Vec::new();
                for (reader_idx, ci, _) in sources {
                    let v = readers[reader_idx]
                        .0
                        .vec()
                        .ok_or(BuildError::VectorReadError)?;
                    rows.extend(v.materialized_cell_rows_at(ci)?);
                }
                for (i, row) in rows.iter_mut().enumerate() {
                    row.local_doc_id = i as u32;
                }
                let stable_ids: Vec<i128> = rows.iter().map(|r| r.stable_id).collect();
                let mut cfg = vec_cfg.clone();
                cfg.n_cent = n_cent.max(1);
                let merged = build_merged_subsection_from_materialized(cfg, rows)?;
                if stable_ids.len() != merged.n_docs as usize {
                    return Err(BuildError::VectorSchemaMismatch(format!(
                        "cell {cell_id}: stable_ids len {} != merged n_docs {}",
                        stable_ids.len(),
                        merged.n_docs
                    )));
                }
                all_stable_ids.extend_from_slice(&stable_ids);
                packed_cells.push((cell_id, merged));
            }
        }

        if packed_cells.is_empty() {
            return Err(BuildError::VectorSchemaMismatch(
                "multi-cell merge produced no live cells after tombstone filter".into(),
            ));
        }

        // Parquet rows must follow the same cell-directory order as the packed
        // IVF subsections. Hidden index files are `_id`-only; user MultiCell
        // files carry the full scalar schema (title, …) and must be reordered
        // by stable id — not replaced with an id-only batch.
        let scalar_batch = scalar_batch_in_stable_id_order(
            &scalar_schema,
            &id_column,
            &scalar_batches,
            &all_stable_ids,
        )?;
        superfile_builder.add_batch_ids_only(&scalar_batch)?;
        superfile_builder.set_prebuilt_multi_cell_ivfs(packed_cells)?;
        let bytes = superfile_builder.finish()?;
        let stats = SuperfileStats::from_children(stats_collector.as_slice());
        Ok((bytes, stats))
    }

    /// Add all data (Parquet + fts + vectors) from another [`SuperfileReader`] to this builder.
    ///
    /// Extracts the record batch and vectors from the reader and adds them via
    /// [`Self::add_batch`]. This is useful for merging superfiles or copying data
    /// between builders.
    ///
    /// **Requirements:**
    /// - The reader's vector indexes must use the **Fp32 codec**. Other codecs
    ///   (Sq8Residual, RabitqOnly) will fail with `BuildError::VectorReadError`.
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
    /// Returns `BuildError::VectorDimMismatch` if vector index names or
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

            // Validate that reader's vector indexes match builder's configuration
            if reader_columns.len() != self.opts.vector_columns.len() {
                return Err(BuildError::VectorDimMismatch {
                    column: format!(
                        "vector index count mismatch: expected {}, got {}",
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
                    .get_vectors_for_merge(&reader_col.name)
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
        let cell_posting_builder = self.cell_posting_builder.take();
        let prebuilt_multi_cell = self.prebuilt_multi_cell.take();

        // Assemble inf.* KV metadata (cheap; do it before the parallel
        // section so the splice has it ready).
        let cell_ids: Option<Vec<u32>> = prebuilt_multi_cell
            .as_ref()
            .map(|cells| cells.iter().map(|(id, _)| *id).collect());
        let kvs = superfile_kvs(&self.opts, n_docs, cell_ids.as_deref())?;

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
        let has_vector = vec_builder.is_some()
            || cell_posting_builder.is_some()
            || prebuilt_multi_cell.is_some();
        let (body, fts_blob, vec_blob) = if has_vector {
            let (fts_blob, vec_blob) = finish_index_blobs(
                fts_builder,
                vec_builder,
                cell_posting_builder,
                prebuilt_multi_cell,
            )?;
            let body = encode_body()?;
            (body, fts_blob, vec_blob)
        } else {
            let (body_res, blobs_res) = rayon::join(encode_body, || {
                finish_index_blobs(
                    fts_builder,
                    vec_builder,
                    cell_posting_builder,
                    prebuilt_multi_cell,
                )
            });
            let body = body_res?;
            let (fts_blob, vec_blob) = blobs_res?;
            (body, fts_blob, vec_blob)
        };

        let parts = splice_index_blobs(body, &fts_blob, &vec_blob, &kvs)?;
        Ok(parts.bytes)
    }

    /// Consume an ids-only builder and stream one packed MultiCellIvf
    /// superfile to `output`.
    ///
    /// Drain uses disk-backed [`MultiCellSubsectionSource`] implementations,
    /// while commit's ordinary [`finish`](Self::finish) uses in-memory
    /// subsections. Directory/CRC assembly and Parquet footer surgery remain
    /// single implementations shared by both paths.
    pub(crate) fn finish_multi_cell_sources_to<W, S>(
        mut self,
        cells: &[S],
        mut output: W,
    ) -> Result<(), BuildError>
    where
        W: Write,
        S: MultiCellSubsectionSource,
    {
        if self.next_local_doc_id == 0 {
            return Err(BuildError::VectorSchemaMismatch(
                "streamed multi-cell finish requires at least one row".into(),
            ));
        }
        if self.fts_builder.is_some()
            || self.cell_posting_builder.is_some()
            || self.prebuilt_multi_cell.is_some()
        {
            return Err(BuildError::VectorSchemaMismatch(
                "streamed multi-cell finish requires ids-only batches and disk-backed cell IVFs"
                    .into(),
            ));
        }
        if self.opts.vector_layout != VectorLayout::MultiCellIvf {
            return Err(BuildError::VectorSchemaMismatch(
                "streamed multi-cell finish requires MultiCellIvf layout".into(),
            ));
        }
        // `SuperfileBuilder::new` registers the configured vector column, but
        // `add_batch_ids_only` deliberately feeds it no rows. The streamed
        // cell-IVFs are the sole vector source for this finish.
        drop(self.vec_builder.take());

        let n_docs = self.next_local_doc_id as u64;
        let cell_ids: Vec<u32> = cells
            .iter()
            .map(MultiCellSubsectionSource::cell_id)
            .collect();
        let kvs = superfile_kvs(&self.opts, n_docs, Some(&cell_ids))?;
        let id_page_limit = [(self.opts.id_column.as_str(), self.opts.id_page_size_limit)];
        let body = encode_parquet_body(
            &self.opts.schema,
            &self.batches,
            self.opts.compression,
            self.opts.row_group_size,
            &id_page_limit,
        )?;

        let mut vector_file = tempfile().map_err(BuildError::Io)?;
        finish_multi_cell_blob_to(cells, BufWriter::new(&mut vector_file))?;
        let vector_length = vector_file.seek(SeekFrom::End(0)).map_err(BuildError::Io)?;
        vector_file
            .seek(SeekFrom::Start(0))
            .map_err(BuildError::Io)?;
        splice_index_streams_to(
            body,
            BufReader::new(Cursor::new(Vec::<u8>::new())),
            0,
            BufReader::new(vector_file),
            vector_length,
            &kvs,
            &mut output,
        )?;
        output.flush().map_err(BuildError::Io)?;
        Ok(())
    }
}

fn superfile_kvs(
    options: &BuilderOptions,
    n_docs: u64,
    multi_cell_ids: Option<&[u32]>,
) -> Result<Vec<(String, String)>, BuildError> {
    let mut kvs: Vec<(String, String)> = vec![
        (kv::FORMAT.into(), kv::FORMAT_VALUE.into()),
        (kv::FORMAT_VERSION.into(), format::FORMAT_VERSION.into()),
        (kv::ID_COLUMN.into(), options.id_column.clone()),
        (kv::N_DOCS.into(), n_docs.to_string()),
        (kv::BUILDER.into(), crate::BUILDER_ID.to_string()),
    ];
    if !options.fts_columns.is_empty() {
        kvs.push((
            kv::FTS_COLUMNS.into(),
            fts_columns_json(&options.fts_columns),
        ));
    }
    if !options.vector_columns.is_empty() {
        kvs.push((
            kv::VEC_COLUMNS.into(),
            vec_columns_json(&options.vector_columns),
        ));
        if options.vector_layout != VectorLayout::Ivf {
            kvs.push((
                kv::VEC_LAYOUT.into(),
                options.vector_layout.as_kv_value().into(),
            ));
        }
        if let Some(cell_ids) = multi_cell_ids {
            let cells_json = serde_json::to_string(cell_ids).map_err(|error| {
                BuildError::VectorSchemaMismatch(format!("inf.vec.cells JSON: {error}"))
            })?;
            kvs.push((kv::VEC_CELLS.into(), cells_json));
        }
    }
    Ok(kvs)
}

/// Rebuild a scalar `RecordBatch` whose rows follow `ordered_ids`.
///
/// - **Id-only schema** (hidden vector-index packs): synthesize the Decimal128
///   `_id` column from `ordered_ids` directly.
/// - **Full scalar schema** (user MultiCell packs): concat the input batches,
///   look up each stable id's row, and `take` every column into cell order so
///   Parquet stays aligned with the packed IVF directory (and FTS rebuild sees
///   the text columns).
fn scalar_batch_in_stable_id_order(
    schema: &Arc<Schema>,
    id_column: &str,
    batches: &[RecordBatch],
    ordered_ids: &[i128],
) -> Result<RecordBatch, BuildError> {
    if schema.fields().len() == 1 {
        let id_array = Decimal128Array::from_iter_values(ordered_ids.iter().copied())
            .with_precision_and_scale(38, 0)
            .map_err(|e| BuildError::BatchSchemaMismatch {
                batch: format!("id Decimal128(38,0) construct failed: {e}"),
                builder: schema.to_string(),
            })?;
        return RecordBatch::try_new(schema.clone(), vec![Arc::new(id_array) as ArrayRef]).map_err(
            |e| BuildError::BatchSchemaMismatch {
                batch: format!("id-only RecordBatch construct failed: {e}"),
                builder: schema.to_string(),
            },
        );
    }

    if batches.is_empty() {
        return Err(BuildError::BatchReadError);
    }
    let concat = concat_batches(schema, batches).map_err(|e| {
        BuildError::Io(Error::other(format!(
            "multi-cell merge: concat scalar batches failed: {e}"
        )))
    })?;
    let id_idx =
        concat
            .schema()
            .index_of(id_column)
            .map_err(|_| BuildError::BatchSchemaMismatch {
                batch: format!("missing id column {id_column:?} in concatenated scalars"),
                builder: schema.to_string(),
            })?;
    let id_col = concat
        .column(id_idx)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .ok_or_else(|| BuildError::BatchSchemaMismatch {
            batch: format!("id column {id_column:?} is not Decimal128"),
            builder: schema.to_string(),
        })?;

    let mut id_to_row: HashMap<i128, u32> = HashMap::with_capacity(id_col.len());
    for row in 0..id_col.len() {
        let stable_id = id_col.value(row);
        if id_to_row.insert(stable_id, row as u32).is_some() {
            return Err(BuildError::VectorSchemaMismatch(format!(
                "multi-cell merge: duplicate stable_id {stable_id} in scalar batches"
            )));
        }
    }
    if ordered_ids.len() != id_to_row.len() {
        return Err(BuildError::VectorSchemaMismatch(format!(
            "multi-cell merge: {} ordered ids for {} visible scalar rows",
            ordered_ids.len(),
            id_to_row.len()
        )));
    }

    let mut indices = Vec::with_capacity(ordered_ids.len());
    for &stable_id in ordered_ids {
        let row = id_to_row.get(&stable_id).copied().ok_or_else(|| {
            BuildError::VectorSchemaMismatch(format!(
                "multi-cell merge: stable_id {stable_id} missing from scalar batches"
            ))
        })?;
        indices.push(row);
    }
    let index_array = UInt32Array::from(indices);
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(concat.num_columns());
    for col in concat.columns() {
        let taken = take(col.as_ref(), &index_array, None).map_err(|e| {
            BuildError::Io(Error::other(format!(
                "multi-cell merge: take scalar column failed: {e}"
            )))
        })?;
        columns.push(taken);
    }
    RecordBatch::try_new(schema.clone(), columns).map_err(|e| BuildError::BatchSchemaMismatch {
        batch: format!("reordered scalar RecordBatch construct failed: {e}"),
        builder: schema.to_string(),
    })
}

/// Finish the independent embedded index blobs. Once `add_batch` has
/// routed scalar text and vectors into their builders, FTS and vector
/// finalization do not share mutable state, so build them as sibling
/// rayon jobs when both indexes are present.
fn finish_index_blobs(
    fts_builder: Option<FtsBuilder>,
    vec_builder: Option<VectorBuilder>,
    cell_posting_builder: Option<CellPostingBuilder>,
    prebuilt_multi_cell: Option<Vec<(u32, MergedIvfSubsection)>>,
) -> Result<(Vec<u8>, Vec<u8>), BuildError> {
    let vec_blob = if let Some(cells) = prebuilt_multi_cell {
        crate::superfile::vector::builder::finish_multi_cell_blob(&cells)?
    } else {
        Vec::new()
    };
    match (
        fts_builder,
        vec_builder,
        cell_posting_builder,
        vec_blob.is_empty(),
    ) {
        (Some(fb), Some(vb), None, true) => {
            let (fts, vec) = rayon::join(|| fb.finish(), || vb.finish());
            Ok((fts?, vec?))
        }
        (Some(fb), None, Some(cb), true) => Ok((fb.finish()?, cb.finish()?)),
        (Some(fb), None, None, true) => Ok((fb.finish()?, Vec::new())),
        (None, Some(vb), None, true) => Ok((Vec::new(), vb.finish()?)),
        (None, None, Some(cb), true) => Ok((Vec::new(), cb.finish()?)),
        (None, None, None, true) => Ok((Vec::new(), Vec::new())),
        (Some(fb), None, None, false) => Ok((fb.finish()?, vec_blob)),
        (None, None, None, false) => Ok((Vec::new(), vec_blob)),
        _ => Err(BuildError::VectorSchemaMismatch(
            "mixed ivf, cell_posting, and multi-cell builders".into(),
        )),
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
        s.push_str(r#"","tokenizer":"ascii_lower""#);
        // Emitted only when set: a positionless column's JSON stays
        // byte-identical to files written before positions existed
        // (the reader defaults a missing field to false).
        if c.positions {
            s.push_str(r#","positions":true"#);
        }
        s.push('}');
    }
    s.push(']');
    s
}

/// Serialize `[VectorConfig]` to the JSON form stored in the
/// legacy-named Parquet KV metadata key `inf.vec.columns`. Same hand-rolled
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
/// stored in legacy `inf.vec.columns` JSON. Matches the strings the
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
    use std::{collections::HashMap, sync::Arc};

    use arrow_array::{Decimal128Array, Int64Array, LargeStringArray, UInt64Array};
    use arrow_schema::Field;
    use bytes::Bytes;
    use roaring::RoaringBitmap;

    use super::*;
    use crate::{
        runtime_bridge::bridge_sync_to_async,
        superfile::{
            format::footer::read_kv_metadata,
            fts::reader::BoolMode,
            vector::rerank_codec::{RerankCodec, SQ8_FIXED_OFFSET, SQ8_FIXED_SCALE},
        },
        test_helpers::{decimal128_ids, default_tokenizer, default_vector_config},
    };

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
                positions: false,
            }],
            vec![],
            Some(default_tokenizer()),
        )
    }

    /// User column names may not contain the FST separator byte or the
    /// reserved `inf.` prefix.
    #[test]
    fn check_user_column_name_rejects_reserved_names() {
        assert!(check_user_column_name("user_id").is_ok());
        let with_sep = format!("a{}b", format::FST_SEPARATOR as char);
        assert!(matches!(
            check_user_column_name(&with_sep),
            Err(BuildError::ReservedSeparatorInColumnName(_))
        ));
        assert!(matches!(
            check_user_column_name("inf.internal"),
            Err(BuildError::ReservedPrefixInColumnName(_))
        ));
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
                    positions: false,
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
                positions: false,
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
                positions: false,
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
                positions: false,
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
                positions: false,
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
        let bad = RecordBatch::try_new(other, vec![Arc::new(UInt64Array::from(vec![1u64]))])
            .expect("build RecordBatch");
        let err = b.add_batch(&bad, &[]).expect_err("expected error");
        assert!(matches!(err, BuildError::BatchSchemaMismatch { .. }));
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
        let kv = read_kv_metadata(&bytes).expect("read kv metadata");
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
        let kv = read_kv_metadata(&bytes).expect("read kv metadata");
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
                positions: false,
            },
            FtsConfig {
                column: "body".into(),
                positions: false,
            },
        ];
        let s = fts_columns_json(&cols);
        assert!(s.starts_with('['));
        assert!(s.contains(r#""name":"title""#));
        assert!(s.contains(r#""name":"body""#));
        assert!(s.contains(r#""tokenizer":"ascii_lower""#));
        // Positionless columns emit no positions field at all — the
        // JSON stays byte-identical to files written before the flag
        // existed.
        assert!(!s.contains("positions"));
    }

    /// The positions field appears only on the columns that opt in,
    /// and a mixed declaration keeps the positionless column's entry
    /// in the legacy shape.
    #[test]
    fn fts_columns_json_positions_emitted_only_when_true() {
        let cols = vec![
            FtsConfig {
                column: "title".into(),
                positions: true,
            },
            FtsConfig {
                column: "body".into(),
                positions: false,
            },
        ];
        let s = fts_columns_json(&cols);
        assert!(
            s.contains(r#"{"name":"title","tokenizer":"ascii_lower","positions":true}"#),
            "positional column carries the flag: {s}"
        );
        assert!(
            s.contains(r#"{"name":"body","tokenizer":"ascii_lower"}"#),
            "positionless column stays in the legacy shape: {s}"
        );
    }

    #[test]
    fn vec_columns_json_round_trip_shape() {
        let cols = vec![VectorConfig {
            column: "emb".into(),
            dim: 384,
            n_cent: 64,
            rot_seed: 99,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Fp32,
            provided_centroids: None,
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
                positions: false,
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
            !stats.scalar_stats.is_empty(),
            "scalar_stats should have column entries"
        );
        assert!(
            stats.scalar_stats.contains_key("doc_id"),
            "scalar_stats should contain id_column"
        );
        assert!(
            stats.scalar_stats.contains_key("title"),
            "scalar_stats should contain FTS column"
        );
        assert!(
            stats.scalar_stats.contains_key("body"),
            "scalar_stats should contain body column"
        );

        // Verify scalar_stats values match expected min/max
        // doc_id: IDs are [10, 11], so min=10, max=11
        let id_agg = stats
            .scalar_stats
            .get("doc_id")
            .expect("doc_id should have stats");
        let (id_min_arr, id_max_arr) = (&id_agg.min, &id_agg.max);
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
        let title_agg = stats
            .scalar_stats
            .get("title")
            .expect("title should have stats");
        let (title_min_arr, title_max_arr) = (&title_agg.min, &title_agg.max);
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
        let body_agg = stats
            .scalar_stats
            .get("body")
            .expect("body should have stats");
        let (body_min_arr, body_max_arr) = (&body_agg.min, &body_agg.max);
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
                positions: false,
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
            !stats.scalar_stats.is_empty(),
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
            !stats.scalar_stats.is_empty(),
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
        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![FtsConfig {
                column: "title".into(),
                positions: false,
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
            !stats.scalar_stats.is_empty(),
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
        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![FtsConfig {
                column: "title".into(),
                positions: false,
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
            !stats.scalar_stats.is_empty(),
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
            vec![VectorConfig {
                column: "emb".into(),
                dim: 16,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::L2Sq,
                rerank_codec: RerankCodec::Sq8Residual,
                provided_centroids: None,
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

        assert!(result.is_err(), "should reject Sq8Residual codec");
    }

    #[tokio::test]
    async fn add_batch_from_reader_queries_work_correctly() {
        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![FtsConfig {
                column: "title".into(),
                positions: false,
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
            !stats.scalar_stats.is_empty(),
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
                positions: false,
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
        assert!(stats.scalar_stats.contains_key("doc_id"));
        assert!(stats.scalar_stats.contains_key("title"));
        assert!(stats.scalar_stats.contains_key("body"));

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
                positions: false,
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
        assert_eq!(stats.scalar_stats.len(), 3, "should have 3 columns");

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
                positions: false,
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
        let opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![FtsConfig {
                column: "title".into(),
                positions: false,
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
                positions: false,
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
                positions: false,
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
                positions: false,
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
        let doc_id_agg = stats.scalar_stats.get("doc_id").expect("doc_id column");
        let (doc_id_min_arr, doc_id_max_arr) = (&doc_id_agg.min, &doc_id_agg.max);
        let doc_id_min = doc_id_min_arr
            .as_ref()
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("downcast to Decimal128")
            .value(0);
        let doc_id_max = doc_id_max_arr
            .as_ref()
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("downcast to Decimal128")
            .value(0);
        assert_eq!(doc_id_min, 10, "doc_id min should be 10");
        assert_eq!(doc_id_max, 11, "doc_id max should be 11");

        // Verify title min/max (from batch_two_rows: ["hello world", "rust async"])
        let title_agg = stats.scalar_stats.get("title").expect("title column");
        let (title_min_arr, title_max_arr) = (&title_agg.min, &title_agg.max);
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
        let body_agg = stats.scalar_stats.get("body").expect("body column");
        let (body_min_arr, body_max_arr) = (&body_agg.min, &body_agg.max);
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
                positions: false,
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
        let doc_id_agg = stats.scalar_stats.get("doc_id").expect("doc_id column");
        let (doc_id_min_arr, doc_id_max_arr) = (&doc_id_agg.min, &doc_id_agg.max);
        let doc_id_min = doc_id_min_arr
            .as_ref()
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("downcast to Decimal128")
            .value(0);
        let doc_id_max = doc_id_max_arr
            .as_ref()
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("downcast to Decimal128")
            .value(0);
        assert_eq!(doc_id_min, 10, "merged doc_id min should be 10");
        assert_eq!(doc_id_max, 21, "merged doc_id max should be 21");

        // Verify title: min should be "alpha", max should be "zeta" (lexicographically from both readers)
        let title_agg = stats.scalar_stats.get("title").expect("title column");
        let (title_min_arr, title_max_arr) = (&title_agg.min, &title_agg.max);
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
        let body_agg = stats.scalar_stats.get("body").expect("body column");
        let (body_min_arr, body_max_arr) = (&body_agg.min, &body_agg.max);
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
                positions: false,
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
        let title_agg = stats.scalar_stats.get("title").expect("title column");
        let (title_min_arr, title_max_arr) = (&title_agg.min, &title_agg.max);
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
        let body_agg = stats.scalar_stats.get("body").expect("body column");
        let (body_min_arr, body_max_arr) = (&body_agg.min, &body_agg.max);
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

    /// The `Debug` impl reports the builder's shape (column counts and
    /// doc-id cursor) without panicking, and `set_fts_spill_threshold_bytes`
    /// forwards to the live FTS builder.
    // --- Sq8 compaction coverage -------------------------------------------

    #[tokio::test]
    async fn sq8_source_merges_via_ivf_byte_splice() {
        // An Sq8 source superfile must be mergeable, but NOT by decoding it back
        // to fp32 and re-quantizing (lossy, and it would break the recall gate).
        // `add_batch_from_reader` therefore rejects an Sq8 column outright and
        // directs callers to the byte-splice path `build_from_sq8_ivf_readers`,
        // which copies the stored Sq8 IVF bytes without a decode/re-encode round
        // trip. This test pins both halves of that contract.
        let sq8_opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![],
            vec![default_vector_config("emb", 7).with_rerank_codec(RerankCodec::Sq8Residual)],
            None,
        );
        let mut b1 = SuperfileBuilder::new(sq8_opts.clone()).expect("new SuperfileBuilder");
        let schema = b1.opts.schema.clone();
        let batch = batch_two_rows(&schema);
        let mut v: Vec<f32> = vec![0.0; 32]; // 2 rows × 16 dim
        v[0] = 1.0; // doc 0 → axis 0
        v[16 + 1] = 1.0; // doc 1 → axis 1
        b1.add_batch(&batch, &[v.as_slice()]).expect("add_batch");
        let source_bytes = b1.finish().expect("finish builder");

        let reader =
            Arc::new(SuperfileReader::open(Bytes::from(source_bytes)).expect("open source"));

        // The fp32 add-batch merge path must refuse an Sq8 column rather than
        // decode-and-requantize it.
        let mut b2 = SuperfileBuilder::new(sq8_opts).expect("new SuperfileBuilder");
        assert!(
            b2.add_batch_from_reader(&reader, None).is_err(),
            "add_batch_from_reader must reject an Sq8 source (splice path only)"
        );

        // The byte-splice path merges it losslessly.
        let (merged_bytes, stats) =
            SuperfileBuilder::build_from_sq8_ivf_readers(&[(Arc::clone(&reader), None)])
                .expect("build_from_sq8_ivf_readers must merge an Sq8 source");
        assert_eq!(stats.n_docs, 2);

        let merged = SuperfileReader::open(Bytes::from(merged_bytes)).expect("open merged reader");
        assert_eq!(merged.n_docs(), 2);

        // Sq8 codec must be preserved in the merged output.
        let col = merged
            .vec()
            .expect("vector index present")
            .vector_columns_config()
            .next()
            .expect("has column");
        assert_eq!(
            col.rerank_codec,
            RerankCodec::Sq8Residual,
            "merged superfile must carry the Sq8Residual codec"
        );

        // Self-query: axis-0 vector must be top hit.
        let mut query = vec![0.0f32; 16];
        query[0] = 1.0;
        let hits = merged
            .vec()
            .expect("vector reader")
            .search("emb", &query, 1, 4, 100)
            .await
            .expect("vector search on merged Sq8 superfile");
        assert!(!hits.is_empty(), "search should return at least one result");
        assert_eq!(hits[0].0, 0, "top hit for axis-0 query must be doc 0");
    }

    /// SQL-shaped tables carry FTS text columns *and* an Sq8 vector column.
    /// Compaction must take the Sq8 byte-splice path while still rebuilding
    /// the FTS blob from the scalar Parquet rows (regression: optimize on
    /// the SQL bench panicked with BatchSchemaMismatch / empty FTS).
    #[tokio::test]
    async fn sq8_fts_sql_shaped_merge_rebuilds_fts() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("bucket", DataType::LargeUtf8, false),
            Field::new("key", DataType::LargeUtf8, false),
            Field::new("category", DataType::LargeUtf8, false),
            Field::new("rating", DataType::Int64, false),
        ]));
        let fts = vec![
            FtsConfig {
                column: "title".into(),
                positions: false,
            },
            FtsConfig {
                column: "bucket".into(),
                positions: false,
            },
            FtsConfig {
                column: "key".into(),
                positions: false,
            },
            FtsConfig {
                column: "category".into(),
                positions: false,
            },
        ];
        let sq8_opts = BuilderOptions::new(
            schema.clone(),
            "doc_id",
            fts,
            vec![default_vector_config("emb", 7).with_rerank_codec(RerankCodec::Sq8Residual)],
            Some(default_tokenizer()),
        );

        let make_file = |id0: u64, title: &str| {
            let mut b = SuperfileBuilder::new(sq8_opts.clone()).expect("new SuperfileBuilder");
            let ids = decimal128_ids(vec![id0, id0 + 1]);
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(ids),
                    Arc::new(LargeStringArray::from(vec![title, "other"])),
                    Arc::new(LargeStringArray::from(vec!["b0", "b1"])),
                    Arc::new(LargeStringArray::from(vec!["k0", "k1"])),
                    Arc::new(LargeStringArray::from(vec!["cat", "dog"])),
                    Arc::new(Int64Array::from(vec![1i64, 2])),
                ],
            )
            .expect("batch");
            let mut v: Vec<f32> = vec![0.0; 32];
            v[0] = 1.0;
            v[16 + 1] = 1.0;
            b.add_batch(&batch, &[v.as_slice()]).expect("add_batch");
            Bytes::from(b.finish().expect("finish"))
        };

        let r1 = Arc::new(SuperfileReader::open(make_file(10, "hellozzz")).expect("open"));
        let r2 = Arc::new(SuperfileReader::open(make_file(20, "worldzzz")).expect("open"));

        // Source FTS must find the planted term before we blame the merge.
        let src_hits = r1
            .fts()
            .expect("source FTS")
            .search("title", &["hellozzz"], 10, BoolMode::Or)
            .await
            .expect("source bm25");
        assert_eq!(src_hits.len(), 1, "source superfile should index hellozzz");

        // Parquet round-trip must keep reader.schema() aligned with the
        // RecordBatch schema `build_from_sq8_ivf_readers` feeds in.
        let batch = r1.get_record_batch(None).expect("get_record_batch");
        assert_eq!(
            batch.schema().fields(),
            r1.schema().fields(),
            "eager open: batch schema must equal reader.schema()"
        );

        let (merged_bytes, stats) =
            SuperfileBuilder::build_from_sq8_ivf_readers(&[(Arc::clone(&r1), None), (r2, None)])
                .expect("sq8+fts SQL-shaped merge");
        assert_eq!(stats.n_docs, 4);

        let merged = SuperfileReader::open(Bytes::from(merged_bytes)).expect("open merged");
        let fts = merged.fts().expect("merged FTS present");
        let hits = fts
            .search("title", &["hellozzz"], 10, BoolMode::Or)
            .await
            .expect("bm25 after sq8 merge");
        assert!(
            !hits.is_empty(),
            "FTS must be rebuilt during Sq8 merge, got no hits for planted term"
        );
    }

    #[tokio::test]
    async fn build_from_readers_fp32_codec_preserved_by_new_from_reader() {
        // new_from_reader previously omitted .with_rerank_codec, so an Fp32 source
        // produced a Sq8 merged output.  After the fix the codec round-trips exactly.
        let fp32_opts = BuilderOptions::new(
            schema_with_fts(),
            "doc_id",
            vec![],
            vec![default_vector_config("emb", 7)], // Fp32 is the default_vector_config codec
            None,
        );
        let mut b = SuperfileBuilder::new(fp32_opts).expect("new SuperfileBuilder");
        let schema = b.opts.schema.clone();
        let batch = batch_two_rows(&schema);
        let mut v: Vec<f32> = vec![0.0; 32];
        v[0] = 1.0;
        v[16 + 1] = 1.0;
        b.add_batch(&batch, &[v.as_slice()]).expect("add_batch");
        let source_bytes = b.finish().expect("finish builder");

        let reader = SuperfileReader::open(Bytes::from(source_bytes)).expect("open reader");
        let (merged_bytes, stats) =
            SuperfileBuilder::build_from_readers(&[(Arc::new(reader), empty_bitmap())])
                .expect("build_from_readers");
        assert_eq!(stats.n_docs, 2);

        let merged = SuperfileReader::open(Bytes::from(merged_bytes)).expect("open merged reader");

        // Fp32 codec must survive the round-trip through new_from_reader.
        let col = merged
            .vec()
            .expect("vector index")
            .vector_columns_config()
            .next()
            .expect("has column");
        assert_eq!(
            col.rerank_codec,
            RerankCodec::Fp32,
            "build_from_readers must preserve Fp32 codec from source superfile"
        );

        // Search must still work on the Fp32 merged output.
        let mut query = vec![0.0f32; 16];
        query[0] = 1.0;
        let hits = merged
            .vec()
            .expect("vector reader")
            .search("emb", &query, 1, 4, 100)
            .await
            .expect("vector search on merged Fp32 superfile");
        assert!(!hits.is_empty());
        assert_eq!(hits[0].0, 0, "top hit for axis-0 query must be doc 0");
    }

    #[test]
    fn debug_and_set_fts_spill_threshold() {
        const FORCE_SPILL_THRESHOLD: usize = 1;
        let mut b = SuperfileBuilder::new(opts_minimal()).expect("new SuperfileBuilder");
        // A 1-byte threshold forces the FTS column onto the spill path;
        // reaches the `Some(fb)` branch since opts_minimal registers a
        // column. (Zero is rejected by the FtsBuilder.)
        b.set_fts_spill_threshold_bytes(FORCE_SPILL_THRESHOLD);

        let rendered = format!("{b:?}");
        assert!(
            rendered.contains("SuperfileBuilder"),
            "debug output names the struct: {rendered}"
        );
        assert!(
            rendered.contains("n_fts_columns"),
            "debug output lists fts columns: {rendered}"
        );
    }

    /// Build one multi-cell packed superfile for merge tests; each spec is
    /// `(cell_id, n_rows, fine n_cent)`.
    fn pack_cells_superfile_with_codec(
        id_base: i128,
        cells: &[(u32, usize, usize)],
        rerank_codec: RerankCodec,
    ) -> Arc<SuperfileReader> {
        use crate::superfile::vector::{
            builder::build_merged_subsection_from_materialized,
            cell_posting::{EncodedCellRow, MaterializedIvfRow},
        };

        let dim = 16usize;
        let make_rows = |cell: u32, n: usize| -> Vec<MaterializedIvfRow> {
            let (scale, offset): (Arc<[f32]>, Arc<[f32]>) =
                if rerank_codec == RerankCodec::Sq8FixedResidual {
                    (
                        Arc::from(vec![SQ8_FIXED_SCALE; dim]),
                        Arc::from(vec![SQ8_FIXED_OFFSET; dim]),
                    )
                } else {
                    (Arc::from(vec![1.0f32; dim]), Arc::from(vec![0.0f32; dim]))
                };
            (0..n)
                .map(|i| {
                    let local = i as u32;
                    let stable_id = id_base + (cell as i128) * 100 + local as i128;
                    let mut codes = vec![0u8; dim];
                    codes[0] = (cell as u8).wrapping_add(i as u8);
                    MaterializedIvfRow {
                        local_doc_id: local,
                        stable_id,
                        cluster: 0,
                        rabitq_code: vec![0u8; dim.div_ceil(8)],
                        encoded: EncodedCellRow {
                            stable_id,
                            rerank_codec,
                            scale: Arc::clone(&scale),
                            offset: Arc::clone(&offset),
                            codes,
                            residuals: vec![0u8; dim],
                            norm_sq: Some(1.0),
                        },
                    }
                })
                .collect()
        };
        let make_cfg = |n_cent: usize| VectorConfig {
            column: "emb".into(),
            dim,
            n_cent,
            rot_seed: 1,
            metric: if rerank_codec == RerankCodec::Sq8FixedResidual {
                Metric::Cosine
            } else {
                Metric::L2Sq
            },
            rerank_codec,
            provided_centroids: None,
        };
        let mut ids: Vec<i128> = Vec::new();
        let mut packed = Vec::with_capacity(cells.len());
        for &(cell_id, n_rows, n_cent) in cells {
            let rows = make_rows(cell_id, n_rows);
            ids.extend(rows.iter().map(|r| r.stable_id));
            let sub = build_merged_subsection_from_materialized(make_cfg(n_cent), rows)
                .expect("cell subsection");
            packed.push((cell_id, sub));
        }

        let schema = Arc::new(Schema::new(vec![Field::new(
            "doc_id",
            DataType::Decimal128(38, 0),
            false,
        )]));
        let id_array = Decimal128Array::from_iter_values(ids.iter().copied())
            .with_precision_and_scale(38, 0)
            .expect("decimal");
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(id_array) as Arc<dyn Array>])
                .expect("batch");
        let first_n_cent = cells.first().map(|&(_, _, n)| n).unwrap_or(1);
        let opts =
            BuilderOptions::new(schema, "doc_id", vec![], vec![make_cfg(first_n_cent)], None)
                .with_vector_layout(VectorLayout::MultiCellIvf);
        let mut b = SuperfileBuilder::new(opts).expect("builder");
        b.add_batch_ids_only(&batch).expect("ids");
        b.set_prebuilt_multi_cell_ivfs(packed).expect("pack");
        let bytes = b.finish().expect("finish");
        Arc::new(SuperfileReader::open(Bytes::from(bytes)).expect("open"))
    }

    fn pack_cells_superfile(id_base: i128, cells: &[(u32, usize, usize)]) -> Arc<SuperfileReader> {
        pack_cells_superfile_with_codec(id_base, cells, RerankCodec::Sq8Residual)
    }

    /// Two cells (3 + 2 rows), both at fine width 2 — the common shape.
    fn pack_two_cell_superfile(id_base: i128) -> Arc<SuperfileReader> {
        pack_cells_superfile(id_base, &[(0, 3, 2), (1, 2, 2)])
    }

    fn rerank_payloads(reader: &SuperfileReader) -> HashMap<i128, Vec<u8>> {
        let rows = bridge_sync_to_async(
            reader
                .vec()
                .expect("vector reader")
                .materialized_index_rows_async("emb"),
        )
        .expect("materialized rows");
        rows.into_iter()
            .map(|row| {
                let mut payload = row.encoded.codes;
                payload.extend_from_slice(&row.encoded.residuals);
                (row.stable_id, payload)
            })
            .collect()
    }

    /// ManifestSnapshot / prepare path must publish the concatenated flat centroid
    /// directory (sum of per-cell `n_cent`), not only the first packed cell.
    /// Otherwise global nprobe only ever scores one cell per shard.
    #[test]
    fn packed_superfile_cluster_summary_covers_all_cells() {
        let sf = pack_two_cell_superfile(1_000);
        let v = sf.vec().expect("vec");
        assert_eq!(v.packed_cell_ids(), &[0, 1]);
        let per_cell: Vec<u32> = v.vector_columns_config().map(|c| c.n_cent).collect();
        assert_eq!(per_cell.len(), 2);
        let (flat_n_cent, dim, centroids, counts) =
            v.cluster_centroids("emb").expect("flat centroids");
        assert_eq!(dim, 16);
        assert_eq!(
            flat_n_cent,
            per_cell.iter().sum::<u32>(),
            "flat n_cent must equal sum of packed cell n_cent ({per_cell:?})"
        );
        assert_eq!(counts.len(), flat_n_cent as usize);
        assert_eq!(centroids.len(), (flat_n_cent as usize) * 16);
        // First-cell-only would equal per_cell[0]; that is the recall cliff.
        assert!(
            flat_n_cent > per_cell[0],
            "flat n_cent={flat_n_cent} collapsed to first cell n_cent={}",
            per_cell[0]
        );
    }

    #[test]
    fn scalar_batch_in_stable_id_order_rejects_duplicate_ids() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("title", DataType::LargeUtf8, false),
        ]));
        let ids = Decimal128Array::from_iter_values([10i128, 10])
            .with_precision_and_scale(38, 0)
            .expect("decimal");
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(ids),
                Arc::new(LargeStringArray::from(vec!["a", "b"])),
            ],
        )
        .expect("batch");
        let err = scalar_batch_in_stable_id_order(&schema, "doc_id", &[batch], &[10, 11])
            .expect_err("duplicate stable_id must fail");
        assert!(
            matches!(err, BuildError::VectorSchemaMismatch(ref m) if m.contains("duplicate")),
            "got {err:?}"
        );
    }

    #[test]
    fn scalar_batch_in_stable_id_order_rejects_row_count_mismatch() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("title", DataType::LargeUtf8, false),
        ]));
        let ids = Decimal128Array::from_iter_values([10i128, 11])
            .with_precision_and_scale(38, 0)
            .expect("decimal");
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(ids),
                Arc::new(LargeStringArray::from(vec!["a", "b"])),
            ],
        )
        .expect("batch");
        // Two visible rows but only one ordered id — must not silently drop a row.
        let err = scalar_batch_in_stable_id_order(&schema, "doc_id", &[batch], &[10])
            .expect_err("ordered_ids/scalar len mismatch must fail");
        assert!(
            matches!(err, BuildError::VectorSchemaMismatch(ref m) if m.contains("ordered ids")),
            "got {err:?}"
        );
    }

    #[test]
    fn multi_cell_merge_preserves_cell_directory() {
        let a = pack_two_cell_superfile(1_000);
        let b = pack_two_cell_superfile(2_000);
        assert_eq!(a.vec().expect("vec").packed_cell_ids(), &[0, 1]);
        assert_eq!(a.n_docs(), 5);

        let (merged_bytes, stats) =
            SuperfileBuilder::build_from_multi_cell_sq8_ivf_readers(&[(a, None), (b, None)])
                .expect("merge");
        assert_eq!(stats.n_docs, 10);

        let merged = SuperfileReader::open(Bytes::from(merged_bytes)).expect("open merged");
        let v = merged.vec().expect("vec");
        assert!(v.is_multi_cell());
        assert_eq!(v.packed_cell_ids(), &[0, 1]);
        assert_eq!(merged.n_docs(), 10);
        // Each cell merged 3+3 and 2+2 rows respectively.
        let cols: Vec<_> = v.vector_columns_config().collect();
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0].n_docs, 6);
        assert_eq!(cols[1].n_docs, 4);
    }

    /// Base drain and a small delta drain legitimately pack the same global
    /// cell at different fine widths (fine `n_cent` is sized by packed bytes).
    /// The merge must rebuild such cells from materialized rows instead of
    /// failing the byte-splice `n_cent` equality check.
    #[test]
    fn multi_cell_merge_rebuilds_cells_with_mismatched_fine_width() {
        // Cell 0 disagrees on width (4 vs 1); cell 1 agrees (splice path).
        let a = pack_cells_superfile(1_000, &[(0, 6, 4), (1, 2, 2)]);
        let b = pack_cells_superfile(2_000, &[(0, 2, 1), (1, 3, 2)]);

        let (merged_bytes, stats) =
            SuperfileBuilder::build_from_multi_cell_sq8_ivf_readers(&[(a, None), (b, None)])
                .expect("merge with mismatched fine n_cent");
        assert_eq!(stats.n_docs, 13);

        let merged = SuperfileReader::open(Bytes::from(merged_bytes)).expect("open merged");
        assert_eq!(merged.n_docs(), 13);
        let v = merged.vec().expect("vec");
        assert_eq!(v.packed_cell_ids(), &[0, 1]);
        let cols: Vec<_> = v.vector_columns_config().collect();
        assert_eq!(cols[0].n_docs, 8); // 6 + 2 rebuilt at the widest width
        assert_eq!(cols[0].n_cent, 4);
        assert_eq!(cols[1].n_docs, 5); // 2 + 3 byte-spliced
        assert_eq!(cols[1].n_cent, 2);
    }

    #[test]
    fn fixed_multi_cell_mismatched_width_merge_preserves_payloads() {
        let codec = RerankCodec::Sq8FixedResidual;
        let a = pack_cells_superfile_with_codec(1_000, &[(0, 6, 4), (1, 2, 2)], codec);
        let b = pack_cells_superfile_with_codec(2_000, &[(0, 2, 1), (1, 3, 2)], codec);
        let mut expected = rerank_payloads(&a);
        expected.extend(rerank_payloads(&b));
        let (merged_bytes, _) =
            SuperfileBuilder::build_from_multi_cell_sq8_ivf_readers(&[(a, None), (b, None)])
                .expect("fixed mismatch merge");
        let merged = SuperfileReader::open(Bytes::from(merged_bytes)).expect("open fixed merge");
        assert_eq!(rerank_payloads(&merged), expected);
        assert!(
            merged
                .vec()
                .expect("vector reader")
                .vector_columns_config()
                .all(|column| column.rerank_codec == codec)
        );
    }

    #[test]
    fn multi_cell_merge_drops_tombstoned_local_docs() {
        let a = pack_two_cell_superfile(1_000);
        // File-local doc ids: cell0 → 0,1,2; cell1 → 3,4. Drop local 1 and 3.
        let mut deny = RoaringBitmap::new();
        deny.insert(1);
        deny.insert(3);

        let (merged_bytes, _stats) =
            SuperfileBuilder::build_from_multi_cell_sq8_ivf_readers(&[(a, Some(Arc::new(deny)))])
                .expect("merge with tombstones");

        let merged = SuperfileReader::open(Bytes::from(merged_bytes)).expect("open");
        assert_eq!(merged.n_docs(), 3);
        let v = merged.vec().expect("vec");
        assert_eq!(v.packed_cell_ids(), &[0, 1]);
        let cols: Vec<_> = v.vector_columns_config().collect();
        assert_eq!(cols[0].n_docs, 2); // kept locals 0,2 from cell0
        assert_eq!(cols[1].n_docs, 1); // kept local 4 from cell1
    }

    #[test]
    fn fixed_multi_cell_tombstone_rebuild_preserves_survivor_payloads() {
        let codec = RerankCodec::Sq8FixedResidual;
        let source = pack_cells_superfile_with_codec(1_000, &[(0, 3, 2), (1, 2, 2)], codec);
        let before = rerank_payloads(&source);
        let mut deny = RoaringBitmap::new();
        deny.insert(1);
        deny.insert(3);
        let (merged_bytes, _) = SuperfileBuilder::build_from_multi_cell_sq8_ivf_readers(&[(
            source,
            Some(Arc::new(deny)),
        )])
        .expect("fixed tombstone merge");
        let merged = SuperfileReader::open(Bytes::from(merged_bytes)).expect("open");
        let after = rerank_payloads(&merged);
        assert_eq!(after.len(), 3);
        for (stable_id, payload) in after {
            assert_eq!(
                before.get(&stable_id),
                Some(&payload),
                "survivor payload changed"
            );
        }
    }
}
