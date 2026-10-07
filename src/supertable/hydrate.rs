// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Hydrate: bulk-load a stream of record batches into a few big no-blob
//! superfiles, for SQL-only tables.
//!
//! `append` builds one superfile per call plus FTS/vector indexes, then needs
//! `optimize` + `gc` to merge the small files. A SQL-only load needs none of it:
//! hydrate packs batches into `~target_rows`-row superfiles and commits them
//! with `append`'s own commit code.

use std::sync::Arc;

use arrow_array::{ArrayRef, Decimal128Array, RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, Schema};
use bytes::Bytes;
use rayon::prelude::*;
use tracing::debug;

use crate::{
    runtime_bridge::bridge_on_runtime,
    runtime_metrics::{ingest::visible_array_bytes, op_stats, rss::available_memory_bytes},
    superfile::builder::{BuilderOptions, SuperfileBuilder},
    supertable::{
        error::BuildError,
        handle::{Supertable, SupertableInner},
        manifest::ScalarStatsAgg,
        options::{DECIMAL128_PRECISION, DECIMAL128_SCALE},
        utils::vector_split::split_vectors,
        writer::{
            CommitListMetadata, ShardOutput, commit_output_stats,
            persist_superfile_publish_batch_async, planned_data_objects,
            prepare_user_superfile_batch, schedule_background_storage_reclaim,
        },
    },
    utils::schema::compare_schema,
};

/// Share of the memory free to the process that one wave may use, 2/5 = 40%.
/// The rest is headroom for the caller's decode, the commit and the allocator.
const BUDGET_NUMER: u64 = 2;
const BUDGET_DENOM: u64 = 5;
/// A chunk being built also holds its growing compressed output and encode
/// scratch: about 1.3x the chunk, 13/10.
const SLOT_NUMER: u64 = 13;
const SLOT_DENOM: u64 = 10;
/// Bytes per row the build adds to the user data: the 16 B `_id` column plus
/// the 16 B id sidecar.
const ID_BYTES_PER_ROW: u64 = 32;
/// Most rows one superfile holds: its doc counter is a `u32`.
const MAX_SUPERFILE_ROWS: u32 = u32::MAX;

/// Bulk-load `reader` into `handle` as a few no-blob superfiles and return the
/// rows committed. The engine side of
/// [`Supertable::hydrate`](crate::Supertable::hydrate), which states the contract.
pub(crate) fn hydrate_from_reader(
    handle: &Supertable,
    reader: &mut dyn RecordBatchReader,
    target_rows: usize,
) -> Result<usize, BuildError> {
    let cores = handle.inner().options.reader_pool.current_num_threads();
    hydrate_with_budget(handle, reader, target_rows, cores, &mut || {
        available_memory_bytes().map(|a| a / BUDGET_DENOM * BUDGET_NUMER)
    })
}

/// [`hydrate_from_reader`] with the core count and the memory budget passed in,
/// so tests can force small chunks and many waves.
fn hydrate_with_budget(
    handle: &Supertable,
    reader: &mut dyn RecordBatchReader,
    target_rows: usize,
    cores: usize,
    memory_budget: &mut dyn FnMut() -> Option<u64>,
) -> Result<usize, BuildError> {
    let inner = handle.inner();
    let options = &inner.options;

    if target_rows == 0 {
        return Err(BuildError::HydrateZeroTargetRows);
    }
    // Reject an indexed table: hydrate writes empty FTS/vector blobs, so its
    // rows would be invisible to search. Fail loudly, don't drop the index.
    let fts = options.fts_columns.len();
    let vector = options.vector_columns.len();
    if fts != 0 || vector != 0 {
        return Err(BuildError::HydrateRequiresNoIndex { fts, vector });
    }
    // Check the reader's schema before reading anything, so a wrong file fails
    // with nothing committed. Each batch is checked again as it is built.
    if !compare_schema(&reader.schema(), &options.schema) {
        return Err(BuildError::BatchSchemaMismatch);
    }

    // Take the writer slot for the whole load, as `append` does. Without a storage
    // backend a commit has no CAS, so only this slot stops a concurrent commit
    // from overwriting ours. Released on drop.
    let _writer = handle.writer()?;

    // No-blob build options: same as a normal build, but with the FTS and vector
    // columns cleared so each superfile is a plain Parquet body, empty blobs.
    let scalar_schema = options.scalar_schema();
    let mut base_opts = options.builder_options();
    base_opts.fts_columns = Vec::new();
    base_opts.vector_columns = Vec::new();

    // Adaptive sizing, re-read every wave from what the process actually has:
    //
    //   budget = min(40% of free memory, what the connection budget has left)
    //   slot   = (visible bytes + 32 B/row of ids) * 1.3   (chunk + encoder output)
    //   chunk  <= budget / 1.3                             (one chunk fits a slot)
    //   wave   = add chunks while one more of the biggest seen still fits,
    //            checked before it is read; at most one per core
    //
    // A big box hits `target_rows` first; a starved box hits the byte cap and
    // builds smaller chunks. Neither budget known: one chunk at a time, no cap.
    let build_pool = &options.reader_pool;
    let cores = cores.max(1);
    let op_stats = op_stats::current();
    let mut chunks = CoalesceChunks::new(reader, target_rows.min(MAX_SUPERFILE_ROWS as usize));
    let mut total_rows: u64 = 0;
    let mut n_superfiles: usize = 0;
    loop {
        let connection_left = options.connection_memory_budget.remaining();
        let budget = [memory_budget(), connection_left.map(|b| b as u64)]
            .into_iter()
            .flatten()
            .min();
        chunks.max_bytes = budget.map_or(u64::MAX, |b| (b * SLOT_DENOM / SLOT_NUMER).max(1));

        let Some(first) = next_chunk(&mut chunks)? else {
            break;
        };
        let mut largest = first.slot();
        let mut wave_bytes = largest;
        let mut wave: Vec<Chunk> = vec![first];
        if let Some(budget) = budget {
            while wave.len() < cores && wave_bytes + largest <= budget {
                let Some(next) = next_chunk(&mut chunks)? else {
                    break;
                };
                largest = largest.max(next.slot());
                wave_bytes += next.slot();
                wave.push(next);
            }
        }

        // Hold the wave against the connection budget through build and commit,
        // as `append` holds its build. A measure-only budget just counts it.
        let _reservation = options
            .connection_memory_budget
            .try_reserve(wave_bytes as usize)
            .map_err(|e| BuildError::OverBudget(format!("during hydrate, {e}")))?;

        // Build the wave's superfiles in parallel and commit them together.
        // Committing per wave keeps only one wave in memory.
        let shards: Vec<ShardOutput> = build_pool.install(|| {
            wave.par_iter()
                .map(|chunk| build_hydrate_shard(inner, &base_opts, &scalar_schema, chunk))
                .collect::<Result<_, BuildError>>()
        })?;
        let rows: u64 = wave.iter().map(|chunk| chunk.rows as u64).sum();
        let payload_bytes: u64 = wave.iter().map(|chunk| chunk.data_bytes).sum();
        let built = shards.len();
        // The input batches aren't needed once built; free them before the commit.
        drop(wave);

        let hints = vec![None; built];
        let mut batch = prepare_user_superfile_batch(inner, shards, hints, None)?;
        // With a storage backend the bytes are durable once committed, so skip
        // the in-memory reader cache, which would hold the whole load in RAM.
        // Without one, that cache is the only copy, so it stays.
        if options.storage.is_some() {
            batch.skip_memory_fill();
        }
        let output_stats = op_stats.as_ref().map(|_| commit_output_stats(&batch));
        bridge_on_runtime(
            persist_superfile_publish_batch_async(inner, batch, CommitListMetadata::empty()),
            &inner.query_runtime(),
        )?;
        // Counted only once the commit lands, as in `append`.
        if let (Some(stats), Some((superfiles, bytes, fts_terms))) = (&op_stats, output_stats) {
            stats.add_ingested_write(rows, payload_bytes, 0, 0);
            stats.add_commit_outputs(superfiles, bytes, fts_terms);
            stats.add_planned_commit_requests(planned_data_objects(payload_bytes));
        }
        total_rows += rows;
        n_superfiles += built;
    }

    // Hand the manifest-list parts the commits replaced to the deferred reclaim,
    // as `append` does after its commits.
    schedule_background_storage_reclaim(Arc::clone(inner));
    debug!(
        superfiles = n_superfiles,
        rows = total_rows,
        "hydrate committed"
    );
    Ok(total_rows as usize)
}

/// The next chunk, turning a read error from the caller's reader into a build error.
fn next_chunk<I>(chunks: &mut CoalesceChunks<I>) -> Result<Option<Chunk>, BuildError>
where
    I: Iterator<Item = Result<RecordBatch, ArrowError>>,
{
    chunks
        .next()
        .transpose()
        .map_err(BuildError::HydrateInputRead)
}

/// Bytes a batch's rows actually use. Visible bytes, not buffer capacity, so a
/// sliced batch doesn't count its whole parent buffer.
fn batch_data_bytes(batch: &RecordBatch) -> u64 {
    batch
        .columns()
        .iter()
        .map(|c| visible_array_bytes(c.as_ref()))
        .sum()
}

/// Input batches that become one superfile, measured once as they are read.
struct Chunk {
    batches: Vec<RecordBatch>,
    rows: usize,
    /// Visible bytes of the user data, without the ids.
    data_bytes: u64,
}

impl Chunk {
    /// Bytes the chunk takes in a build: its data plus the per-row id bytes.
    fn footprint(&self) -> u64 {
        self.data_bytes + self.rows as u64 * ID_BYTES_PER_ROW
    }

    /// Bytes the chunk holds while it is built, encoder output included.
    fn slot(&self) -> u64 {
        (self.footprint() * SLOT_NUMER / SLOT_DENOM).max(1)
    }
}

/// Build one no-blob superfile from a chunk: mint and prepend `_id` (as `append`
/// does), stream through [`SuperfileBuilder::build_no_blob_from_batches_to`], and
/// wrap the bytes with the manifest metadata a publish needs.
///
/// It runs on many rayon workers at once, and the id generator is their only
/// shared state. So it reserves the chunk's whole id range under one short lock
/// and builds the arrays outside it, and the workers don't queue on the lock.
fn build_hydrate_shard(
    inner: &SupertableInner,
    base_opts: &BuilderOptions,
    scalar_schema: &Arc<Schema>,
    chunk: &Chunk,
) -> Result<ShardOutput, BuildError> {
    // `target_rows` is clamped to this, so only a single oversized input batch
    // can get here.
    let n_ids = u32::try_from(chunk.rows).map_err(|_| BuildError::HydrateChunkTooLarge {
        rows: chunk.rows,
        max: MAX_SUPERFILE_ROWS,
    })?;

    // Reserve all ids for the chunk under one short lock. Ids are monotonic, so
    // the min/max are just the first and last of the reserved spans.
    let id_spans = {
        let generator = inner
            .id_generator
            .lock()
            .expect("id_generator mutex poisoned");
        generator.reserve_range(n_ids)
    };
    let id_min = id_spans.first().map_or(0, |&(first, _)| first);
    let id_max = id_spans.last().map_or(0, |&(_, last)| last);
    // Flatten the spans into one id per row, in row order (total == rows).
    let mut ids = id_spans.iter().flat_map(|&(first, last)| first..=last);

    let mut ided: Vec<RecordBatch> = Vec::with_capacity(chunk.batches.len());
    for batch in &chunk.batches {
        // Check the batch against the table schema by name, type and nullability,
        // and put its columns in the table's order: the same check `append` runs.
        // A permuted batch loads correctly; a misnamed or mistyped one is rejected.
        let (scalar, _) = split_vectors(batch, &inner.options)?;
        let id_array = Decimal128Array::from_iter_values((&mut ids).take(scalar.num_rows()))
            .with_precision_and_scale(DECIMAL128_PRECISION, DECIMAL128_SCALE)
            .expect("invariant: precision 38 + scale 0 is valid for any i128 payload");
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(scalar.num_columns() + 1);
        columns.push(Arc::new(id_array));
        columns.extend(scalar.columns().iter().cloned());
        ided.push(
            RecordBatch::try_new(Arc::clone(scalar_schema), columns)
                .map_err(|_| BuildError::BatchSchemaMismatch)?,
        );
    }

    // Stream into one no-blob superfile. Size the sink to the chunk's footprint:
    // the compressed output is smaller, so this just avoids reallocating mid-encode.
    let mut bytes: Vec<u8> = Vec::with_capacity(chunk.footprint() as usize);
    SuperfileBuilder::build_no_blob_from_batches_to(base_opts.clone(), &ided, &mut bytes)?;

    // Per-scalar-column min/max for skip pruning, over the id-prepended batches.
    let scalar_refs: Vec<&RecordBatch> = ided.iter().collect();
    let scalar_stats = ScalarStatsAgg::from_batches(scalar_schema, &scalar_refs);

    Ok(ShardOutput::new_with_params(
        Bytes::from(bytes),
        chunk.rows as u64,
        id_min,
        id_max,
        scalar_stats,
    ))
}

/// Groups a batch stream into [`Chunk`]s, ending a chunk once it reaches
/// `target_rows` rows or the next batch would push its footprint past
/// `max_bytes`. Empty batches are skipped.
///
/// - The byte cap is strict: a batch that would cross it is held and starts the
///   next chunk. Only a single batch bigger than the cap goes over, on its own.
/// - The row target is not: a chunk ends on the batch that reaches it.
/// - A read error ends the stream with that error; the partly filled chunk is
///   dropped, never built.
///
/// At most one held batch sits outside a wave, so hydrate reads ahead of its
/// budget by at most one input batch.
struct CoalesceChunks<I> {
    inner: I,
    target_rows: usize,
    max_bytes: u64,
    /// A batch that didn't fit the last chunk, with its measured data bytes.
    held: Option<(RecordBatch, u64)>,
}

impl<I> CoalesceChunks<I> {
    /// No byte cap until the caller sets `max_bytes`.
    fn new(inner: I, target_rows: usize) -> Self {
        Self {
            inner,
            target_rows,
            max_bytes: u64::MAX,
            held: None,
        }
    }
}

impl<I: Iterator<Item = Result<RecordBatch, ArrowError>>> Iterator for CoalesceChunks<I> {
    type Item = Result<Chunk, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut chunk = Chunk {
            batches: Vec::new(),
            rows: 0,
            data_bytes: 0,
        };
        while chunk.rows < self.target_rows {
            let (batch, data_bytes) = match self.held.take() {
                Some(held) => held,
                None => match self.inner.next() {
                    Some(Ok(batch)) if batch.num_rows() == 0 => continue,
                    Some(Ok(batch)) => {
                        let data_bytes = batch_data_bytes(&batch);
                        (batch, data_bytes)
                    }
                    Some(Err(e)) => return Some(Err(e)),
                    None => break,
                },
            };
            let footprint = data_bytes + batch.num_rows() as u64 * ID_BYTES_PER_ROW;
            if !chunk.batches.is_empty() && chunk.footprint() + footprint > self.max_bytes {
                self.held = Some((batch, data_bytes));
                break;
            }
            chunk.rows += batch.num_rows();
            chunk.data_bytes += data_bytes;
            chunk.batches.push(batch);
        }
        (!chunk.batches.is_empty()).then_some(Ok(chunk))
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use arrow_array::{
        Array, Int64Array, RecordBatch, RecordBatchIterator, RecordBatchReader, StringArray,
    };
    use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
    use tempfile::TempDir;

    use crate::{
        ConnectOptions, Connection, IndexSpec, InfinoError, Metric, OptimizeOptions, connect,
        connect_with,
        runtime_metrics::op_stats::with_op_stats,
        supertable::{
            Supertable,
            error::BuildError,
            hydrate::{
                Chunk, CoalesceChunks, batch_data_bytes, hydrate_from_reader, hydrate_with_budget,
            },
        },
    };

    /// Total rows across the correctness corpus.
    const N_ROWS: i64 = 5_000;
    /// Rows per input batch fed to both paths.
    const BATCH_ROWS: i64 = 1_000;
    /// Coalescing target: two 1_000-row batches per superfile.
    const HYDRATE_TARGET_ROWS: usize = 2_000;
    /// Cores for the sizing tests: more than any wave needs, so only the budget
    /// limits a wave.
    const MANY_CORES: usize = 64;
    /// Smallest vector dim a table accepts.
    const EMB_DIM: usize = 16;

    fn user_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("n", DataType::Int64, false),
            Field::new("s", DataType::Utf8, false),
        ]))
    }

    /// A batch of `n = lo..=hi` and `s = "r<n>"`.
    fn rows_batch(lo: i64, hi: i64) -> RecordBatch {
        let n = Int64Array::from((lo..=hi).collect::<Vec<_>>());
        let s = StringArray::from((lo..=hi).map(|i| format!("r{i}")).collect::<Vec<_>>());
        RecordBatch::try_new(user_schema(), vec![Arc::new(n), Arc::new(s)]).expect("valid batch")
    }

    /// One-column `(n Int64)` schema: every batch of the same row count has the
    /// same footprint, so the sizing tests can compute budgets exactly.
    fn ints_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, false)]))
    }

    /// A batch of `n = lo..=hi` over [`ints_schema`].
    fn ints_batch(lo: i64, hi: i64) -> RecordBatch {
        let n = Int64Array::from((lo..=hi).collect::<Vec<_>>());
        RecordBatch::try_new(ints_schema(), vec![Arc::new(n)]).expect("valid batch")
    }

    /// `batch` alone, measured as a chunk.
    fn as_chunk(batch: &RecordBatch) -> Chunk {
        Chunk {
            batches: Vec::new(),
            rows: batch.num_rows(),
            data_bytes: batch_data_bytes(batch),
        }
    }

    /// The chunks a coalescer cuts from `batches`.
    fn coalesce(batches: Vec<RecordBatch>, target_rows: usize, max_bytes: u64) -> Vec<Chunk> {
        let mut chunks = CoalesceChunks::new(batches.into_iter().map(Ok), target_rows);
        chunks.max_bytes = max_bytes;
        chunks.collect::<Result<_, _>>().expect("no read error")
    }

    /// `batches` as the reader `hydrate` takes, declaring `schema`.
    fn reader(schema: SchemaRef, batches: Vec<RecordBatch>) -> impl RecordBatchReader {
        RecordBatchIterator::new(batches.into_iter().map(Ok), schema)
    }

    fn hydrate(
        table: &Supertable,
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
        target_rows: usize,
    ) -> Result<usize, BuildError> {
        hydrate_from_reader(table, &mut reader(schema, batches), target_rows)
    }

    /// A fresh database with one no-index table `t` over `schema`.
    fn table_with(schema: SchemaRef) -> (TempDir, Connection, Supertable) {
        table_on(schema, ConnectOptions::default())
    }

    /// [`table_with`] on a connection opened with `options`.
    fn table_on(schema: SchemaRef, options: ConnectOptions) -> (TempDir, Connection, Supertable) {
        let dir = TempDir::new().expect("tempdir");
        let uri = dir.path().join("db");
        let db = connect_with(uri.to_str().expect("utf-8 path"), options).expect("connect");
        db.create_table("t", schema, IndexSpec::new())
            .expect("create_table");
        let table = db.open_table_handle("t").expect("core handle");
        (dir, db, table)
    }

    /// Rows across every superfile `table` has committed.
    fn committed_rows(table: &Supertable) -> u64 {
        table
            .inner()
            .manifest
            .load()
            .get_all_superfiles()
            .iter()
            .map(|entry| entry.n_docs)
            .sum()
    }

    /// One scalar cell rendered as text so two results compare exactly.
    fn render(batches: &[RecordBatch]) -> Vec<String> {
        let mut rows = Vec::new();
        for batch in batches {
            for r in 0..batch.num_rows() {
                let cells: Vec<String> = (0..batch.num_columns())
                    .map(|c| {
                        let col = batch.column(c);
                        if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
                            if a.is_null(r) {
                                "NULL".into()
                            } else {
                                a.value(r).to_string()
                            }
                        } else if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
                            if a.is_null(r) {
                                "NULL".into()
                            } else {
                                a.value(r).to_string()
                            }
                        } else {
                            panic!("unhandled result column type: {:?}", col.data_type())
                        }
                    })
                    .collect();
                rows.push(cells.join("|"));
            }
        }
        rows
    }

    fn query(db: &Connection, sql: &str) -> Vec<String> {
        render(&db.query_sql(sql).expect("query_sql"))
    }

    /// The byte cap cuts a chunk before the row target, and it is strict.
    ///  - cap of 2.5 batches: chunks of 2, 2, 1 batches; a third batch would
    ///    cross the cap, so it starts the next chunk.
    ///  - no cap: grouped by the 2_000-row target, 2, 2, 1 batches.
    ///  - cap smaller than any batch: one whole batch per chunk, never split.
    /// Every input row survives each way.
    #[test]
    fn coalesce_cuts_on_byte_cap() {
        let batches: Vec<RecordBatch> = (0..5)
            .map(|i| ints_batch(i * 1_000 + 1, (i + 1) * 1_000))
            .collect();
        let one = as_chunk(&batches[0]).footprint();
        let shape = |chunks: &[Chunk]| chunks.iter().map(|c| c.batches.len()).collect::<Vec<_>>();
        let rows = |chunks: &[Chunk]| chunks.iter().map(|c| c.rows).sum::<usize>();

        let capped = coalesce(batches.clone(), usize::MAX, one * 5 / 2);
        assert_eq!(shape(&capped), [2, 2, 1]);
        assert!(capped.iter().all(|c| c.footprint() <= one * 5 / 2));
        assert_eq!(rows(&capped), 5_000);

        let by_rows = coalesce(batches.clone(), 2_000, u64::MAX);
        assert_eq!(shape(&by_rows), [2, 2, 1]);
        assert_eq!(rows(&by_rows), 5_000);

        let floored = coalesce(batches, usize::MAX, 1);
        assert_eq!(shape(&floored), [1, 1, 1, 1, 1]);
        assert_eq!(rows(&floored), 5_000);
    }

    /// A read error ends the stream with that error, and the chunk it was filling
    /// is dropped rather than handed out half full.
    #[test]
    fn coalesce_stops_at_a_read_error() {
        let mut chunks = CoalesceChunks::new(
            vec![
                Ok(rows_batch(1, 10)),
                Ok(rows_batch(11, 20)),
                Err(ArrowError::ParseError("bad row group".into())),
            ]
            .into_iter(),
            usize::MAX,
        );
        assert!(matches!(
            chunks.next(),
            Some(Err(ArrowError::ParseError(_)))
        ));
    }

    /// Hydrate and append + optimize + gc give the same answers over the same rows,
    /// and hydrate packs 5 batches into 3 superfiles.
    #[test]
    fn hydrate_matches_normal_ingest_and_coalesces() {
        let dir = TempDir::new().expect("tempdir");
        let db = connect(dir.path().join("db").to_str().expect("utf-8 path")).expect("connect");
        let batches: Vec<RecordBatch> = (0..N_ROWS / BATCH_ROWS)
            .map(|i| rows_batch(i * BATCH_ROWS + 1, (i + 1) * BATCH_ROWS))
            .collect();

        // Normal ingest: one append per batch, then optimize + gc.
        let ingested = db
            .create_table("ingested", user_schema(), IndexSpec::new())
            .expect("create_table (ingest)");
        for batch in &batches {
            ingested.append(batch).expect("append");
        }
        ingested
            .optimize(&OptimizeOptions::default())
            .expect("optimize");
        ingested.gc(Duration::ZERO).expect("gc");

        // Hydrate the same rows through the core handle `query_sql` reads.
        db.create_table("hydrated", user_schema(), IndexSpec::new())
            .expect("create_table (hydrate)");
        let hydrated = db.open_table_handle("hydrated").expect("core table handle");
        let committed =
            hydrate(&hydrated, user_schema(), batches, HYDRATE_TARGET_ROWS).expect("hydrate");
        assert_eq!(committed, N_ROWS as usize);

        // Same answers. `_id` is excluded: it is minted per path and differs.
        for q in [
            "SELECT COUNT(*) FROM {t}",
            "SELECT SUM(n) FROM {t}",
            "SELECT COUNT(*) FROM {t} WHERE n > 3000",
            "SELECT MIN(n), MAX(n) FROM {t}",
        ] {
            let a = query(&db, &q.replace("{t}", "hydrated"));
            let b = query(&db, &q.replace("{t}", "ingested"));
            assert_eq!(a, b, "mismatch for query: {q}");
        }

        // 5 batches of 1_000 at a 2_000-row target pack into 3 superfiles.
        let n_superfiles = hydrated.inner().manifest.load().get_all_superfiles().len();
        assert_eq!(n_superfiles, 3);

        // An empty reader commits nothing.
        let none = hydrate(&hydrated, user_schema(), Vec::new(), HYDRATE_TARGET_ROWS)
            .expect("hydrate empty");
        assert_eq!(none, 0);
    }

    /// Hydrate on an indexed table is rejected, not run with the index dropped:
    /// its rows would otherwise be invisible to BM25 and vector search.
    #[test]
    fn hydrate_rejects_indexed_table() {
        let dir = TempDir::new().expect("tempdir");
        let db = connect(dir.path().join("db").to_str().expect("utf-8 path")).expect("connect");
        let fts_schema = Arc::new(Schema::new(vec![
            Field::new("n", DataType::Int64, false),
            Field::new("body", DataType::LargeUtf8, false),
        ]));
        let emb = DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, false)),
            EMB_DIM as i32,
        );
        let vector_schema = Arc::new(Schema::new(vec![
            Field::new("n", DataType::Int64, false),
            Field::new("emb", emb, false),
        ]));
        db.create_table(
            "docs",
            Arc::clone(&fts_schema),
            IndexSpec::new().fts("body"),
        )
        .expect("create fts table");
        db.create_table(
            "vecs",
            Arc::clone(&vector_schema),
            IndexSpec::new().vector("emb", EMB_DIM, Metric::L2Sq),
        )
        .expect("create vector table");

        let docs = db.open_table_handle("docs").expect("core handle");
        let err = hydrate(&docs, fts_schema, Vec::new(), 1_000).expect_err("fts table");
        assert!(
            matches!(
                err,
                BuildError::HydrateRequiresNoIndex { fts: 1, vector: 0 }
            ),
            "got {err:?}"
        );
        let vecs = db.open_table_handle("vecs").expect("core handle");
        let err = hydrate(&vecs, vector_schema, Vec::new(), 1_000).expect_err("vector table");
        assert!(
            matches!(
                err,
                BuildError::HydrateRequiresNoIndex { fts: 0, vector: 1 }
            ),
            "got {err:?}"
        );
    }

    /// A zero `target_rows` is rejected as a bad setting, not treated as 1.
    #[test]
    fn hydrate_rejects_zero_target_rows() {
        let (_dir, _db, table) = table_with(user_schema());
        let err = hydrate(&table, user_schema(), vec![rows_batch(1, 10)], 0).expect_err("zero");
        assert!(
            matches!(err, BuildError::HydrateZeroTargetRows),
            "got {err:?}"
        );
        assert!(matches!(InfinoError::from(err), InfinoError::Config(_)));
    }

    /// Nulls round-trip through the id-prepend, the build, the scalar stats and
    /// the scan: hydrate and normal ingest give the same `COUNT`, `SUM` (which
    /// skips nulls), `IS NULL` and `MIN`/`MAX`.
    #[test]
    fn hydrate_matches_normal_ingest_with_nulls() {
        // Every third row's `v` is null.
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Int64, true),
        ]));
        let batch = |lo: i64, hi: i64| {
            let k = Int64Array::from((lo..=hi).collect::<Vec<_>>());
            let v = Int64Array::from(
                (lo..=hi)
                    .map(|i| if i % 3 == 0 { None } else { Some(i * 2) })
                    .collect::<Vec<_>>(),
            );
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(k), Arc::new(v)])
                .expect("valid batch")
        };

        let dir = TempDir::new().expect("tempdir");
        let db = connect(dir.path().join("db").to_str().expect("utf-8 path")).expect("connect");
        let ingested = db
            .create_table("ingested", Arc::clone(&schema), IndexSpec::new())
            .expect("create_table");
        ingested.append(&batch(1, 1_500)).expect("append");
        ingested.append(&batch(1_501, 3_000)).expect("append");
        ingested
            .optimize(&OptimizeOptions::default())
            .expect("optimize");
        ingested.gc(Duration::ZERO).expect("gc");

        db.create_table("hydrated", Arc::clone(&schema), IndexSpec::new())
            .expect("create_table");
        let hydrated = db.open_table_handle("hydrated").expect("core handle");
        let committed = hydrate(
            &hydrated,
            Arc::clone(&schema),
            vec![batch(1, 1_500), batch(1_501, 3_000)],
            1_000,
        )
        .expect("hydrate");
        assert_eq!(committed, 3_000);

        for q in [
            "SELECT COUNT(*) FROM {t}",
            "SELECT COUNT(v) FROM {t}",
            "SELECT SUM(v) FROM {t}",
            "SELECT COUNT(*) FROM {t} WHERE v IS NULL",
            "SELECT COUNT(*) FROM {t} WHERE v IS NOT NULL",
            "SELECT MIN(v), MAX(v) FROM {t}",
        ] {
            let a = query(&db, &q.replace("{t}", "hydrated"));
            let b = query(&db, &q.replace("{t}", "ingested"));
            assert_eq!(a, b, "mismatch for query: {q}");
        }
    }

    /// Hydrate holds the writer slot like `append`, so it can't race another writer.
    ///  - while another writer is held, hydrate fails with `SupertableInUse`.
    ///  - once that writer drops, hydrate runs, then releases the slot.
    #[test]
    fn hydrate_takes_the_writer_slot() {
        let (_dir, _db, table) = table_with(user_schema());

        let held = table.writer().expect("writer");
        let err = hydrate(&table, user_schema(), vec![rows_batch(1, 10)], 1_000)
            .expect_err("hydrate must not run while another writer holds the slot");
        assert!(matches!(err, BuildError::SupertableInUse), "got {err:?}");
        drop(held);

        let committed =
            hydrate(&table, user_schema(), vec![rows_batch(1, 10)], 1_000).expect("hydrate");
        assert_eq!(committed, 10);
        table.writer().expect("hydrate released the slot");
    }

    /// Columns are matched by name, not position, exactly as in `append`.
    ///  - the table is `(price, qty)`; the reader yields `(qty, price)`.
    ///  - both are Int64, so a positional load would swap them silently.
    /// Each column's SUM must match what was written.
    #[test]
    fn hydrate_matches_columns_by_name() {
        let table_schema = Arc::new(Schema::new(vec![
            Field::new("price", DataType::Int64, false),
            Field::new("qty", DataType::Int64, false),
        ]));
        let permuted = Arc::new(Schema::new(vec![
            Field::new("qty", DataType::Int64, false),
            Field::new("price", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&permuted),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(Int64Array::from(vec![100, 200, 300])),
            ],
        )
        .expect("valid batch");

        let (_dir, db, table) = table_with(table_schema);
        hydrate(&table, permuted, vec![batch], 1_000).expect("hydrate");
        assert_eq!(
            query(&db, "SELECT SUM(price), SUM(qty) FROM t"),
            vec!["600|6"]
        );
    }

    /// A schema that doesn't match the table is rejected, as in `append`, and
    /// nothing commits.
    ///  - the reader declares a wrong schema: rejected before anything is read.
    ///  - the reader declares the right schema but a batch differs (a misnamed
    ///    column, or a nullable `n` into the non-nullable `n`): rejected when
    ///    that batch is built.
    /// The table stays empty and the writer slot is released either way.
    #[test]
    fn hydrate_rejects_schema_mismatch_before_committing() {
        let (_dir, db, table) = table_with(user_schema());
        let with_fields = |fields: Vec<Field>| {
            RecordBatch::try_new(
                Arc::new(Schema::new(fields)),
                vec![
                    Arc::new(Int64Array::from(vec![1, 2])),
                    Arc::new(StringArray::from(vec!["a", "b"])),
                ],
            )
            .expect("valid batch")
        };
        let misnamed = with_fields(vec![
            Field::new("x", DataType::Int64, false),
            Field::new("s", DataType::Utf8, false),
        ]);
        let nullable = with_fields(vec![
            Field::new("n", DataType::Int64, true),
            Field::new("s", DataType::Utf8, false),
        ]);

        for bad in [misnamed, nullable] {
            let declared_wrong = hydrate(&table, bad.schema(), vec![bad.clone()], 1_000);
            let declared_right = hydrate(&table, user_schema(), vec![bad], 1_000);
            for err in [declared_wrong, declared_right] {
                let err = err.expect_err("a mismatched schema must be rejected");
                assert!(
                    matches!(err, BuildError::BatchSchemaMismatch),
                    "got {err:?}"
                );
            }
        }

        // The declared schema is checked before anything is read: this reader's
        // first item is a read error, yet the schema error comes back.
        let wrong = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
        let mut unread = RecordBatchIterator::new(
            vec![Err(ArrowError::ParseError("never read".into()))],
            wrong,
        );
        let err = hydrate_from_reader(&table, &mut unread, 1_000).expect_err("wrong schema");
        assert!(
            matches!(err, BuildError::BatchSchemaMismatch),
            "got {err:?}"
        );

        assert_eq!(query(&db, "SELECT COUNT(*) FROM t"), vec!["0"]);
        table.writer().expect("slot released on the error path");
    }

    /// A read error from the reader is returned, not swallowed into a partial
    /// table that reports `Ok`.
    ///  - one chunk per wave (1 core), so the first batch commits on its own.
    ///  - the second item is a read error: hydrate returns it as an I/O error.
    /// The rows committed before the error stay: hydrate is not atomic.
    #[test]
    fn hydrate_returns_a_read_error() {
        let (_dir, db, table) = table_with(user_schema());
        let mut failing = RecordBatchIterator::new(
            vec![
                Ok(rows_batch(1, 1_000)),
                Err(ArrowError::ParseError("bad row group".into())),
            ],
            user_schema(),
        );
        let err = hydrate_with_budget(&table, &mut failing, 1_000, 1, &mut || None)
            .expect_err("a read error must be returned");
        assert!(
            matches!(err, BuildError::HydrateInputRead(_)),
            "got {err:?}"
        );
        assert!(matches!(InfinoError::from(err), InfinoError::Io(_)));
        assert_eq!(query(&db, "SELECT COUNT(*) FROM t"), vec!["1000"]);
    }

    /// Each wave stays inside the budget, and no chunk is lost between waves.
    ///  - 6 chunks of 1_000 rows, each one slot `s` (row target binds first).
    ///  - budget `2.5 s`: room for two chunks but not three, so 3 waves of 2.
    /// Every row lands, and the largest reservation is two chunks, under the budget.
    #[test]
    fn hydrate_waves_stay_within_the_budget() {
        let (_dir, db, table) = table_with(ints_schema());
        let batches: Vec<RecordBatch> = (0..6)
            .map(|i| ints_batch(i * 1_000 + 1, (i + 1) * 1_000))
            .collect();
        let slot = as_chunk(&batches[0]).slot();
        let budget = slot * 5 / 2;

        let committed = hydrate_with_budget(
            &table,
            &mut reader(ints_schema(), batches),
            1_000,
            MANY_CORES,
            &mut || Some(budget),
        )
        .expect("hydrate");
        assert_eq!(committed, 6_000);
        assert_eq!(
            query(&db, "SELECT COUNT(*), SUM(n) FROM t"),
            vec!["6000|18003000"]
        );
        assert_eq!(table.inner().manifest.load().get_all_superfiles().len(), 6);
        let peak = table.inner().options.connection_memory_budget.peak() as u64;
        assert_eq!(peak, 2 * slot, "a wave held two chunks, no more");
        assert!(peak <= budget);
    }

    /// A small budget caps chunk size below `target_rows`.
    ///  - `target_rows` is effectively unlimited, so rows alone would make one chunk.
    ///  - the budget fits one batch per slot, so the byte cap cuts every batch.
    /// 6 batches land as 6 superfiles, one wave each.
    #[test]
    fn hydrate_byte_cap_shrinks_chunks() {
        let (_dir, db, table) = table_with(ints_schema());
        let batches: Vec<RecordBatch> = (0..6)
            .map(|i| ints_batch(i * 1_000 + 1, (i + 1) * 1_000))
            .collect();
        let budget = as_chunk(&batches[0]).slot();

        let committed = hydrate_with_budget(
            &table,
            &mut reader(ints_schema(), batches),
            usize::MAX,
            MANY_CORES,
            &mut || Some(budget),
        )
        .expect("hydrate");
        assert_eq!(committed, 6_000);
        assert_eq!(query(&db, "SELECT COUNT(*) FROM t"), vec!["6000"]);
        assert_eq!(table.inner().manifest.load().get_all_superfiles().len(), 6);
    }

    /// On a storage-backed table, hydrate doesn't fill the in-memory reader cache.
    ///  - hydrate: the cache stays empty, and queries read from storage.
    ///  - `append` on the same setup does fill it, so the check means something.
    #[test]
    fn hydrate_skips_the_memory_cache_with_storage() {
        let (_dir, db, table) = table_with(user_schema());
        let options = &table.inner().options;
        assert!(options.storage.is_some() && options.disk_cache.is_none());

        hydrate(&table, user_schema(), vec![rows_batch(1, 1_000)], 1_000).expect("hydrate");
        assert_eq!(options.store.resident_bytes(), 0);
        assert_eq!(query(&db, "SELECT COUNT(*) FROM t"), vec!["1000"]);

        let (_dir2, _db2, appended) = table_with(user_schema());
        let mut writer = appended.writer().expect("writer");
        writer.append(&rows_batch(1, 1_000)).expect("append");
        writer.commit().expect("commit");
        assert!(appended.inner().options.store.resident_bytes() > 0);
    }

    /// Hydrate records the same per-op stats `append` does, so a load shows up as
    /// rows ingested and superfiles written instead of zero.
    #[test]
    fn hydrate_records_op_stats() {
        let (_dir, _db, table) = table_with(user_schema());
        let (committed, stats) = with_op_stats(|| {
            hydrate(
                &table,
                user_schema(),
                vec![rows_batch(1, 1_000), rows_batch(1_001, 2_000)],
                1_000,
            )
            .expect("hydrate")
        });
        assert_eq!(committed, 2_000);
        assert_eq!(stats.rows_written, 2_000);
        assert_eq!(stats.superfiles_written, 2);
    }

    /// On a bounded connection budget, chunks are capped to fit it and every wave
    /// reserves inside it, so the load succeeds instead of hitting `OverBudget`.
    ///  - budget ~2.85 batch footprints: the cap fits 2 batches per chunk.
    ///  - a 3rd batch would cross the cap, so it starts the next chunk; letting it
    ///    in would make a slot (~3.9 footprints) bigger than the whole budget.
    /// 6 batches land as 3 superfiles, and the peak reservation stays inside.
    #[test]
    fn hydrate_fits_a_bounded_connection_budget() {
        let batches: Vec<RecordBatch> = (0..6)
            .map(|i| ints_batch(i * 1_000 + 1, (i + 1) * 1_000))
            .collect();
        let footprint = as_chunk(&batches[0]).footprint();
        // The connection keeps 9/10 of what is configured as its ceiling.
        let configured = footprint * 285 / 100 * 10 / 9;
        let options = ConnectOptions::default().with_connection_memory_budget_bytes(configured);
        let (_dir, _db, table) = table_on(ints_schema(), options);
        let budget = &table.inner().options.connection_memory_budget;
        let limit = budget.remaining().expect("bounded budget") as u64;

        let committed = hydrate(&table, ints_schema(), batches, usize::MAX).expect("hydrate");
        assert_eq!(committed, 6_000);
        assert_eq!(committed_rows(&table), 6_000);
        assert_eq!(table.inner().manifest.load().get_all_superfiles().len(), 3);
        assert!(budget.peak() as u64 <= limit);
    }

    /// A batch too big for the whole connection budget is refused as over budget,
    /// with nothing committed and the writer slot released.
    #[test]
    fn hydrate_is_over_budget_when_one_batch_cannot_fit() {
        let batch = ints_batch(1, 1_000);
        let configured = as_chunk(&batch).footprint() / 2;
        let options = ConnectOptions::default().with_connection_memory_budget_bytes(configured);
        let (_dir, _db, table) = table_on(ints_schema(), options);

        let err = hydrate(&table, ints_schema(), vec![batch], usize::MAX).expect_err("too big");
        assert!(matches!(err, BuildError::OverBudget(_)), "got {err:?}");
        assert!(matches!(InfinoError::from(err), InfinoError::OverBudget(_)));
        assert_eq!(committed_rows(&table), 0);
        table.writer().expect("slot released on the error path");
    }
}
