// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Hydrate: bulk-load a batch stream straight into a few big no-blob
//! superfiles, SQL-only.
//!
//! `append` builds one superfile per call and relies on `optimize` + `gc` to
//! merge them. For data already in columnar form and queried with SQL only,
//! the index builds and that merge pass are wasted. [`hydrate_from_batches`]
//! coalesces the batches into `~target_rows`-row chunks, builds each as a
//! no-blob superfile, and commits them through the same primitives `append`'s
//! commit uses, with no optimize or GC.

use std::sync::Arc;

use arrow_array::{ArrayRef, Decimal128Array, RecordBatch};
use arrow_schema::Schema;
use bytes::Bytes;
use rayon::prelude::*;
use tracing::debug;

use crate::{
    runtime_bridge::bridge_on_runtime,
    runtime_metrics::rss::available_memory_bytes,
    superfile::builder::{BuilderOptions, SuperfileBuilder},
    supertable::{
        error::BuildError,
        handle::{Supertable, SupertableInner},
        manifest::ScalarStatsAgg,
        options::{DECIMAL128_PRECISION, DECIMAL128_SCALE},
        writer::{
            CommitListMetadata, ShardOutput, persist_superfile_publish_batch_async,
            prepare_user_superfile_batch,
        },
    },
};

/// Bulk-load `user_batches` (the user schema, no `_id`) into `handle` as a few
/// no-blob superfiles and return the rows committed. Batches are coalesced into
/// `~target_rows`-row chunks, one superfile each; `_id` is minted and prepended
/// per row exactly as `append` does. No optimize or GC.
///
/// - SQL-only: rejects a table with FTS or vector columns, since hydrate writes
///   empty index blobs and those rows would never be found by search.
/// - Not atomic: superfiles commit per wave, so a failure part way leaves the
///   committed waves in place. Recover by dropping the table and re-hydrating,
///   not by re-calling hydrate (that re-mints ids and duplicates rows).
pub(crate) fn hydrate_from_batches(
    handle: &Supertable,
    user_batches: impl IntoIterator<Item = RecordBatch>,
    target_rows: usize,
) -> Result<usize, BuildError> {
    let inner = handle.inner();
    let scalar_schema = inner.options.scalar_schema();

    // Reject an indexed table: hydrate writes empty FTS/vector blobs, so its
    // rows would be invisible to search. Fail loudly, don't drop the index.
    let fts = inner.options.fts_columns.len();
    let vector = inner.options.vector_columns.len();
    if fts != 0 || vector != 0 {
        return Err(BuildError::HydrateRequiresNoIndex { fts, vector });
    }

    // No-blob build options: same as a normal build, but with the FTS and vector
    // columns cleared so each superfile is a plain Parquet body, empty blobs.
    let mut base_opts = inner.options.builder_options();
    base_opts.fts_columns = Vec::new();
    base_opts.vector_columns = Vec::new();

    // Adaptive sizing: one binary from a 2 GiB container to a 192-core box, no
    // OOM, no idle cores. How many superfiles build at once, and how big each
    // chunk is, both come from what the process actually has:
    //
    //   budget   = 40% of available memory    (cgroup limit, or host MemAvailable)
    //   per_slot = chunk bytes * 1.3          (raw chunk + its compressed output)
    //   wave     = as many chunks as fit the budget, capped at one per core
    //   cap      = budget / 1.3               (so one chunk always fits a slot)
    //
    // The cap makes `target_rows` a max: a big box hits the row target first, a
    // starved box hits the byte cap first and builds smaller chunks. `None`
    // budget (can't measure memory) means one chunk at a time, no cap.
    //
    // Build runs on the reader pool: the work is CPU (Parquet encode), and its
    // thread count already follows a cgroup CPU quota.
    const BUDGET_NUMER: u64 = 2;
    const BUDGET_DENOM: u64 = 5; // 40%
    const SLOT_NUMER: u64 = 13;
    const SLOT_DENOM: u64 = 10; // 1.3x
    let build_pool = &inner.options.reader_pool;
    let cores = build_pool.current_num_threads().max(1) as u64;
    let build_budget = available_memory_bytes().map(|a| a / BUDGET_DENOM * BUDGET_NUMER);
    let chunk_byte_cap = build_budget
        .map(|b| (b * SLOT_DENOM / SLOT_NUMER).max(1))
        .unwrap_or(u64::MAX);

    let mut chunks = CoalesceChunks {
        inner: user_batches.into_iter(),
        target: target_rows.max(1),
        max_bytes: chunk_byte_cap,
    };

    // Build each wave in parallel, commit it, free its bytes, then the next.
    // Committing per wave (not once at the end) keeps only one wave resident. A
    // fresh load has no readers, so partial visibility between waves is fine.
    let slot_bytes =
        |chunk: &[RecordBatch]| (chunk_arrow_bytes(chunk) * SLOT_NUMER / SLOT_DENOM).max(1);
    let mut total_rows: u64 = 0;
    let mut n_superfiles: usize = 0;
    let mut carry: Option<Vec<RecordBatch>> = None;
    while let Some(first) = carry.take().or_else(|| chunks.next()) {
        let mut wave_bytes = slot_bytes(&first);
        let mut wave: Vec<Vec<RecordBatch>> = vec![first];
        // Add chunks while they fit the budget, up to one per core. A chunk that
        // doesn't fit is held in `carry` for the next wave (`chunks` can't peek).
        // The first chunk always goes in, even if it alone is over budget.
        if let Some(budget) = build_budget {
            while (wave.len() as u64) < cores {
                let Some(next) = chunks.next() else { break };
                let next_bytes = slot_bytes(&next);
                if wave_bytes + next_bytes > budget {
                    carry = Some(next);
                    break;
                }
                wave_bytes += next_bytes;
                wave.push(next);
            }
        }

        let shards: Vec<ShardOutput> = build_pool
            .install(|| {
                wave.par_iter()
                    .map(|chunk| build_hydrate_shard(inner, &base_opts, &scalar_schema, chunk))
                    .collect::<Result<Vec<_>, BuildError>>()
            })?
            .into_iter()
            .flatten()
            .collect();
        if shards.is_empty() {
            continue;
        }

        total_rows += shards.iter().map(ShardOutput::n_docs).sum::<u64>();
        n_superfiles += shards.len();

        let hints = vec![None; shards.len()];
        let batch = prepare_user_superfile_batch(inner, shards, hints, None)?;
        bridge_on_runtime(
            persist_superfile_publish_batch_async(inner, batch, CommitListMetadata::empty()),
            &inner.query_runtime(),
        )?;
    }

    debug!(
        superfiles = n_superfiles,
        rows = total_rows,
        "hydrate committed"
    );
    Ok(total_rows as usize)
}

/// Total in-memory Arrow footprint of a coalesced chunk, used to size a wave
/// against the memory budget.
fn chunk_arrow_bytes(chunk: &[RecordBatch]) -> u64 {
    chunk
        .iter()
        .map(|b| b.get_array_memory_size() as u64)
        .sum::<u64>()
        .max(1)
}

/// Build one no-blob superfile from a coalesced chunk: mint and prepend `_id`
/// (as `append` does), stream through
/// [`SuperfileBuilder::build_no_blob_from_batches_to`], and wrap the bytes with
/// the manifest metadata a publish needs. `None` for an empty chunk.
///
/// This runs on many rayon workers at once, and the id generator is the only
/// shared state. So reserve the chunk's whole id range under one short lock, then
/// build the arrays outside it, so the workers don't serialize on the generator.
fn build_hydrate_shard(
    inner: &SupertableInner,
    base_opts: &BuilderOptions,
    scalar_schema: &Arc<Schema>,
    user_chunk: &[RecordBatch],
) -> Result<Option<ShardOutput>, BuildError> {
    let n_docs: usize = user_chunk.iter().map(RecordBatch::num_rows).sum();
    if n_docs == 0 {
        return Ok(None);
    }
    // A superfile's doc counter is `u32`, so a chunk with more rows than that
    // can't be built; fail here, at the id reservation, with a clear message.
    let n_ids: u32 = n_docs
        .try_into()
        .expect("chunk row count exceeds u32; a superfile cannot hold this many rows");

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
    // Flatten the spans into one id per row, in row order (total == n_docs).
    let mut ids = id_spans.iter().flat_map(|&(first, last)| first..=last);

    // Prepend the minted `_id` column to each batch, matching the append path.
    let mut ided: Vec<RecordBatch> = Vec::with_capacity(user_chunk.len());
    for user_batch in user_chunk {
        let n_rows = user_batch.num_rows();
        if n_rows == 0 {
            continue;
        }
        let id_array = Decimal128Array::from_iter_values((&mut ids).take(n_rows))
            .with_precision_and_scale(DECIMAL128_PRECISION, DECIMAL128_SCALE)
            .expect("invariant: precision 38 + scale 0 is valid for any i128 payload");
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(user_batch.num_columns() + 1);
        columns.push(Arc::new(id_array));
        columns.extend(user_batch.columns().iter().cloned());
        ided.push(
            RecordBatch::try_new(Arc::clone(scalar_schema), columns)
                .map_err(|_| BuildError::BatchSchemaMismatch)?,
        );
    }

    // Stream into one no-blob superfile. Size the sink to the raw Arrow bytes:
    // the compressed output is smaller, so this just avoids reallocating mid-encode.
    let mut bytes: Vec<u8> = Vec::with_capacity(chunk_arrow_bytes(user_chunk) as usize);
    SuperfileBuilder::build_no_blob_from_batches_to(base_opts.clone(), &ided, &mut bytes)?;

    // Per-scalar-column min/max for skip pruning, over the id-prepended batches.
    let scalar_refs: Vec<&RecordBatch> = ided.iter().collect();
    let scalar_stats = ScalarStatsAgg::from_batches(scalar_schema, &scalar_refs);

    Ok(Some(ShardOutput::new_with_params(
        Bytes::from(bytes),
        n_docs as u64,
        id_min,
        id_max,
        scalar_stats,
    )))
}

/// Groups a `RecordBatch` stream into chunks, cutting at whichever comes first:
/// `target` rows or `max_bytes` Arrow bytes (use `u64::MAX` for rows only). Empty
/// batches are skipped. A batch bigger than `max_bytes` is its own chunk, so the
/// input batch size is the floor. Yields owned `Vec`s for rayon to build.
struct CoalesceChunks<I> {
    inner: I,
    target: usize,
    max_bytes: u64,
}

impl<I: Iterator<Item = RecordBatch>> Iterator for CoalesceChunks<I> {
    type Item = Vec<RecordBatch>;

    fn next(&mut self) -> Option<Vec<RecordBatch>> {
        let mut chunk: Vec<RecordBatch> = Vec::new();
        let mut rows = 0usize;
        let mut bytes = 0u64;
        while rows < self.target && bytes < self.max_bytes {
            match self.inner.next() {
                Some(batch) => {
                    let n = batch.num_rows();
                    if n == 0 {
                        continue;
                    }
                    rows += n;
                    bytes += batch.get_array_memory_size() as u64;
                    chunk.push(batch);
                }
                None => break,
            }
        }
        if chunk.is_empty() { None } else { Some(chunk) }
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use arrow_array::{Array, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use tempfile::TempDir;

    use crate::{IndexSpec, OptimizeOptions, connect, supertable::hydrate::hydrate_from_batches};

    /// Total rows across the correctness corpus.
    const N_ROWS: i64 = 5_000;
    /// Rows per input batch fed to both paths.
    const BATCH_ROWS: i64 = 1_000;
    /// Coalescing target: two 1_000-row batches per superfile.
    const HYDRATE_TARGET_ROWS: usize = 2_000;

    fn user_schema() -> Arc<Schema> {
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

    /// The byte cap cuts a chunk before the row target when the Arrow bytes reach
    /// it; `u64::MAX` groups by rows only. Either way, every input row survives.
    #[test]
    fn coalesce_cuts_on_byte_cap() {
        use crate::supertable::hydrate::CoalesceChunks;

        let batches: Vec<RecordBatch> = (0..5)
            .map(|i| rows_batch(i * 1000 + 1, i * 1000 + 1000))
            .collect();
        let one_batch_bytes = batches[0].get_array_memory_size() as u64;
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();

        // Byte cap below two batches but at/above one: each chunk holds exactly one
        // batch even though the row target (huge) is never reached.
        let capped: Vec<Vec<RecordBatch>> = CoalesceChunks {
            inner: batches.clone().into_iter(),
            target: usize::MAX,
            max_bytes: one_batch_bytes,
        }
        .collect();
        assert_eq!(capped.len(), 5, "byte cap should cut one batch per chunk");
        assert!(capped.iter().all(|c| c.len() == 1));
        assert_eq!(
            capped.iter().flatten().map(|b| b.num_rows()).sum::<usize>(),
            total_rows
        );

        // No byte cap: group purely by the 2_000-row target (two 1_000-row batches
        // per chunk).
        let by_rows: Vec<Vec<RecordBatch>> = CoalesceChunks {
            inner: batches.into_iter(),
            target: 2_000,
            max_bytes: u64::MAX,
        }
        .collect();
        assert_eq!(
            by_rows.len(),
            3,
            "5 batches of 1000 into 2000-row chunks = 3,2 -> 3 chunks"
        );
        assert_eq!(
            by_rows
                .iter()
                .flatten()
                .map(|b| b.num_rows())
                .sum::<usize>(),
            total_rows
        );

        // Floor: a cap smaller than a single batch must not drop rows or yield an
        // empty chunk. The input batch is the smallest unit, so each chunk is still
        // exactly one batch.
        let floored: Vec<Vec<RecordBatch>> = CoalesceChunks {
            inner: (0..3)
                .map(|i| rows_batch(i * 10 + 1, i * 10 + 10))
                .collect::<Vec<_>>()
                .into_iter(),
            target: usize::MAX,
            max_bytes: 1,
        }
        .collect();
        assert_eq!(
            floored.len(),
            3,
            "a sub-batch cap still cuts one batch per chunk"
        );
        assert!(
            floored.iter().all(|c| c.len() == 1),
            "never splits a batch, never empty"
        );
        assert_eq!(
            floored
                .iter()
                .flatten()
                .map(|b| b.num_rows())
                .sum::<usize>(),
            30
        );
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

    fn query(db: &crate::Connection, sql: &str) -> Vec<String> {
        render(&db.query_sql(sql).expect("query_sql"))
    }

    /// The hydrate path and the normal append + optimize + gc path must answer
    /// every SQL query identically over the same rows, and the hydrate path must
    /// coalesce into a small, bounded number of superfiles.
    #[test]
    fn hydrate_matches_normal_ingest_and_coalesces() {
        let dir = TempDir::new().expect("tempdir");
        let db = connect(dir.path().join("db").to_str().expect("utf-8 path")).expect("connect");

        // (A) Normal ingest: append in BATCH_ROWS-row batches, then optimize + gc.
        let ingested = db
            .create_table("ingested", user_schema(), IndexSpec::new())
            .expect("create_table (ingest)");
        let mut lo = 1;
        while lo <= N_ROWS {
            let hi = (lo + BATCH_ROWS - 1).min(N_ROWS);
            ingested.append(&rows_batch(lo, hi)).expect("append");
            lo = hi + 1;
        }
        ingested
            .optimize(&OptimizeOptions::default())
            .expect("optimize");
        ingested.gc(Duration::ZERO).expect("gc");

        // (B) Hydrate: same rows, coalesced into ~HYDRATE_TARGET_ROWS chunks.
        // Reach the engine's core table handle (what `hydrate_from_batches`
        // operates on) through the catalog: the same shared handle `query_sql`
        // reads, so the commit is visible to queries.
        db.create_table("hydrated", user_schema(), IndexSpec::new())
            .expect("create_table (hydrate)");
        let hydrated = db.open_table_handle("hydrated").expect("core table handle");
        let mut batches = Vec::new();
        let mut lo = 1;
        while lo <= N_ROWS {
            let hi = (lo + BATCH_ROWS - 1).min(N_ROWS);
            batches.push(rows_batch(lo, hi));
            lo = hi + 1;
        }
        let committed =
            hydrate_from_batches(&hydrated, batches, HYDRATE_TARGET_ROWS).expect("hydrate");
        assert_eq!(committed, N_ROWS as usize, "all rows committed");

        // Identical answers for count, sum, filter, and min/max. `_id` is
        // excluded: it is minted per path and differs.
        let queries = [
            "SELECT COUNT(*) FROM {t}",
            "SELECT SUM(n) FROM {t}",
            "SELECT COUNT(*) FROM {t} WHERE n > 3000",
            "SELECT MIN(n), MAX(n) FROM {t}",
        ];
        for q in queries {
            let a = query(&db, &q.replace("{t}", "hydrated"));
            let b = query(&db, &q.replace("{t}", "ingested"));
            assert_eq!(a, b, "mismatch for query: {q}");
        }

        // Coalescing worked: 5 batches of 1_000 at a 2_000-row target pack into
        // 3 superfiles, far fewer than the 5 a per-batch append would produce
        // (and nowhere near the row count).
        let n_superfiles = hydrated.inner().manifest.load().get_all_superfiles().len();
        assert!(
            (1..=3).contains(&n_superfiles),
            "hydrate coalesced into a bounded superfile count, got {n_superfiles}"
        );
        assert!(
            n_superfiles < N_ROWS as usize,
            "far fewer superfiles than rows"
        );

        // Sanity: the empty-input path commits nothing and returns 0.
        let empty: Vec<RecordBatch> = Vec::new();
        let none =
            hydrate_from_batches(&hydrated, empty, HYDRATE_TARGET_ROWS).expect("hydrate empty");
        assert_eq!(none, 0, "empty input commits nothing");
    }

    /// Hydrate on an indexed table is rejected, not silently run with the index
    /// dropped: a table with an FTS (or vector) column would otherwise commit rows
    /// that BM25/vector search can never find.
    #[test]
    fn hydrate_rejects_indexed_table() {
        use crate::supertable::error::BuildError;

        let schema = Arc::new(Schema::new(vec![
            Field::new("n", DataType::Int64, false),
            Field::new("body", DataType::LargeUtf8, false),
        ]));
        let dir = TempDir::new().expect("tempdir");
        let db = connect(dir.path().join("db").to_str().expect("utf-8 path")).expect("connect");
        db.create_table("docs", Arc::clone(&schema), IndexSpec::new().fts("body"))
            .expect("create_table");
        let table = db.open_table_handle("docs").expect("core handle");

        let err = hydrate_from_batches(&table, Vec::<RecordBatch>::new(), 1_000)
            .expect_err("hydrate on an FTS table must error");
        assert!(
            matches!(
                err,
                BuildError::HydrateRequiresNoIndex { fts: 1, vector: 0 }
            ),
            "expected HydrateRequiresNoIndex, got {err:?}"
        );
    }

    /// Nulls must round-trip through the id-prepend, the build, the scalar-stats
    /// aggregation, and the scan: hydrate and normal ingest answer `COUNT`, `SUM`
    /// (which skips nulls), and `IS NULL` / `IS NOT NULL` identically.
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
        let committed =
            hydrate_from_batches(&hydrated, vec![batch(1, 1_500), batch(1_501, 3_000)], 1_000)
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
}
