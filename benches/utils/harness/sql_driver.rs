// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Engine-generic SQL driver.
//!
//! Builds one canonical 1-writer queryable artifact, optionally measures
//! an N-writer build-throughput row, and times SQL queries against the
//! canonical artifact. `run_sql_with_index` returns the artifact so
//! in-tree benches can run additional correctness/warm/cold checks before
//! calling `close`/`delete`.

use std::{
    any::Any,
    panic::{self, AssertUnwindSafe},
    time::{Duration, Instant},
};

use arrow_array::RecordBatch;

use super::{SchemaDrivenSqlEngine, SqlCorpusSpec, SqlEngine, SqlRow};
use crate::{
    cpu,
    markdown::fmt_count,
    rss::{PeakSampler, RssStats},
};

#[derive(Clone, Copy, Debug)]
pub struct SqlQuery {
    pub name: &'static str,
    pub sql: &'static str,
}

#[derive(Clone, Copy, Debug)]
pub struct SqlRunConfig {
    pub iters: usize,
    pub parallel: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct SqlBuildStat {
    pub writers: usize,
    pub wall: Duration,
    pub rss: RssStats,
    /// Measured on-CPU seconds of the build (all-thread schedstat delta),
    /// when sampled — prices the build compute instead of a NOT-METERED gap.
    pub cpu_s: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct SqlQueryStats {
    pub name: &'static str,
    pub p50: Duration,
    pub rss: RssStats,
    pub rows: usize,
    /// Amortized on-CPU seconds of one timed iteration (all-thread schedstat
    /// delta over the loop), when sampled. The cost model prices compute only
    /// from measured CPU, so `None` makes the query report NOT METERED rather
    /// than a wall-clock guess.
    pub cpu_s: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct EngineSqlResult {
    pub engine: &'static str,
    pub builds: Vec<SqlBuildStat>,
    pub queries: Vec<SqlQueryStats>,
    /// Queries whose read panicked (name, panic message) — dropped from
    /// `queries` rather than fabricating a timing, but never silently: this
    /// is the machine-readable record of what the panic already printed.
    pub skipped: Vec<(&'static str, String)>,
}

pub fn run_sql<E: SqlEngine>(
    cfg: SqlRunConfig,
    rows: &[SqlRow<'_>],
    queries: &[SqlQuery],
) -> EngineSqlResult {
    let (result, mut index) = run_sql_with_index::<E>(cfg, rows, queries);
    E::close(&mut index);
    E::delete(index);
    result
}

/// Shared measurement skeleton: 1-writer ingest, optional N-writer probe,
/// then the query battery. Ingest is supplied as closures so the row path
/// and the schema-driven batch path share one copy of the timing logic.
fn measure_sql<E: SqlEngine>(
    cfg: SqlRunConfig,
    index: E::Index,
    n_rows: usize,
    ingest_1w: impl FnOnce(&mut E::Index),
    ingest_nw: impl FnOnce(usize),
    queries: &[SqlQuery],
) -> (EngineSqlResult, E::Index) {
    eprintln!(
        "[harness/sql] {}: building 1-writer table over {} rows...",
        E::name(),
        fmt_count(n_rows),
    );
    let mut index = index;
    let sampler = PeakSampler::start_default();
    let ((), wall, cpu_s) = cpu::timed(|| ingest_1w(&mut index));
    let rss = sampler.stop_stats();
    let mut builds = vec![SqlBuildStat {
        writers: 1,
        wall,
        rss,
        cpu_s,
    }];

    if cfg.parallel > 1 {
        eprintln!(
            "[harness/sql] {}: parallel build probe ({} writers)...",
            E::name(),
            cfg.parallel,
        );
        let sampler = PeakSampler::start_default();
        let ((), wall, cpu_s) = cpu::timed(|| ingest_nw(cfg.parallel));
        let rss = sampler.stop_stats();
        builds.push(SqlBuildStat {
            writers: cfg.parallel,
            wall,
            rss,
            cpu_s,
        });
    }

    // One battery-level progress line; per-query results land in the
    // report table, so per-query progress lines are just noise.
    if !queries.is_empty() {
        eprintln!(
            "[harness/sql] {}: warm query battery ({} queries × {} timed iters)...",
            E::name(),
            queries.len(),
            cfg.iters,
        );
    }
    let mut queries_out = Vec::with_capacity(queries.len());
    let mut skipped: Vec<(&'static str, String)> = Vec::new();
    for q in queries {
        let sampler = PeakSampler::start_default();
        // A panicking read — whether a schema the query can't be planned
        // against, or a genuine engine bug — must not take down every other
        // query's timing. The default panic hook still prints the message
        // (that IS the diagnostic); this only stops it from being fatal.
        let read = || panic::catch_unwind(AssertUnwindSafe(|| E::read(&index, q.sql)));
        let warm = match read() {
            Ok(out) => out,
            Err(payload) => {
                sampler.stop_stats();
                skipped.push((q.name, panic_payload_message(&*payload)));
                continue;
            }
        };
        let mut samples = Vec::with_capacity(cfg.iters.max(1));
        let mut broke_mid_loop = false;
        // Sampled across the timed loop only, so the untimed warmup read
        // above is excluded from the per-query compute the cost model bills.
        let cpu0 = cpu::process_cpu_ns();
        for _ in 0..cfg.iters.max(1) {
            let t0 = Instant::now();
            match read() {
                Ok(out) => {
                    samples.push(t0.elapsed());
                    std::hint::black_box(out);
                }
                Err(payload) => {
                    skipped.push((q.name, panic_payload_message(&*payload)));
                    broke_mid_loop = true;
                    break;
                }
            }
        }
        let cpu_s = cpu::cpu_seconds_since(cpu0);
        let rss = sampler.stop_stats();
        if broke_mid_loop {
            continue; // no stable timing for a query that panicked partway through
        }
        queries_out.push(SqlQueryStats {
            name: q.name,
            p50: percentile_duration(&mut samples, 50),
            rss,
            rows: warm.rows,
            cpu_s: cpu_s.map(|s| s / samples.len() as f64),
        });
    }
    if !skipped.is_empty() {
        eprintln!(
            "[harness/sql] {}: {} of {} queries panicked and were skipped (see the panic \
             output above for each)",
            E::name(),
            skipped.len(),
            queries.len(),
        );
    }

    (
        EngineSqlResult {
            engine: E::name(),
            builds,
            queries: queries_out,
            skipped,
        },
        index,
    )
}

pub fn run_sql_with_index<E: SqlEngine>(
    cfg: SqlRunConfig,
    rows: &[SqlRow<'_>],
    queries: &[SqlQuery],
) -> (EngineSqlResult, E::Index) {
    measure_sql::<E>(
        cfg,
        E::open(),
        rows.len(),
        |index| E::write(index, rows),
        |writers| E::parallel_write(rows, writers),
        queries,
    )
}

/// Schema-driven counterpart to [`run_sql_with_index`]: same measurement
/// skeleton, batches instead of the fixed row fixture.
pub fn run_sql_batches_with_index<E: SchemaDrivenSqlEngine>(
    cfg: SqlRunConfig,
    spec: &SqlCorpusSpec,
    batches: &[RecordBatch],
    queries: &[SqlQuery],
) -> (EngineSqlResult, E::Index) {
    let n_rows = batches.iter().map(RecordBatch::num_rows).sum();
    measure_sql::<E>(
        cfg,
        E::create_with_spec(spec),
        n_rows,
        |index| E::write_batches(index, batches),
        |writers| E::parallel_write_batches(spec, batches, writers),
        queries,
    )
}

/// Best-effort text for a caught panic payload — `catch_unwind` only
/// guarantees `Any`, and the two payload shapes `panic!`/`.expect()` use
/// cover it in practice. Callers must pass `&*payload`, not `&payload`: the
/// latter coerces to the boxed pointer's own (uninformative) `Any` impl
/// instead of the panic value it points to.
fn panic_payload_message(payload: &(dyn Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_string())
}

fn percentile_duration(samples: &mut [Duration], percentile: usize) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    samples.sort_unstable();
    let rank = ((percentile as f64 / 100.0) * samples.len() as f64).ceil() as usize;
    samples[rank.saturating_sub(1).min(samples.len() - 1)]
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use arrow_array::{ArrayRef, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};

    use super::*;
    use crate::harness::{
        Capabilities, SchemaDrivenSqlEngine, SqlCorpusSpec, SqlEngine, SqlOutput, SqlRow,
    };

    struct StubEngine;
    #[derive(Default)]
    struct StubIndex {
        rows_written: usize,
    }

    impl SqlEngine for StubEngine {
        type Index = StubIndex;
        fn name() -> &'static str {
            "stub"
        }
        fn capabilities() -> Capabilities {
            Capabilities {
                sql: true,
                ..Default::default()
            }
        }
        fn create() -> Self::Index {
            StubIndex::default()
        }
        fn write(index: &mut Self::Index, rows: &[SqlRow<'_>]) {
            index.rows_written = rows.len();
        }
        fn parallel_write(_rows: &[SqlRow<'_>], _writers: usize) {}
        fn read(_index: &Self::Index, _sql: &str) -> SqlOutput {
            SqlOutput { rows: 7 }
        }
        fn close(_index: &mut Self::Index) {}
        fn delete(_index: Self::Index) {}
    }

    impl SchemaDrivenSqlEngine for StubEngine {
        fn create_with_spec(_spec: &SqlCorpusSpec) -> Self::Index {
            StubIndex::default()
        }
        fn write_batches(index: &mut Self::Index, batches: &[RecordBatch]) {
            index.rows_written = batches.iter().map(RecordBatch::num_rows).sum();
        }
        fn parallel_write_batches(
            _spec: &SqlCorpusSpec,
            _batches: &[RecordBatch],
            _writers: usize,
        ) {
        }
    }

    fn stub_batch(rows: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, false)]));
        let col = Int64Array::from((0..rows as i64).collect::<Vec<_>>());
        RecordBatch::try_new(schema, vec![Arc::new(col) as ArrayRef]).expect("stub batch")
    }

    /// The batch path reports the summed row count of its batches and
    /// reuses the same skeleton as the row path.
    #[test]
    fn batch_path_counts_rows_across_batches() {
        let batches = [stub_batch(3), stub_batch(5)];
        let spec = SqlCorpusSpec {
            schema: batches[0].schema(),
            fts_columns: Vec::new(),
            vector: None,
        };
        let (result, index) = run_sql_batches_with_index::<StubEngine>(
            SqlRunConfig {
                iters: 1,
                parallel: 1,
            },
            &spec,
            &batches,
            &[],
        );
        assert_eq!(index.rows_written, 8);
        assert_eq!(result.builds.len(), 1);
    }

    /// Compile-time proof that the public row-path signature is unchanged:
    /// this is exactly how an out-of-repo engine calls it.
    #[test]
    fn run_sql_with_index_keeps_its_row_signature() {
        let rows = [SqlRow {
            doc_id: 0,
            title: "a",
            category: "c",
            score: 1,
        }];
        let queries = [SqlQuery {
            name: "q",
            sql: "SELECT 1",
        }];
        let (result, index) = run_sql_with_index::<StubEngine>(
            SqlRunConfig {
                iters: 1,
                parallel: 1,
            },
            &rows,
            &queries,
        );
        assert_eq!(index.rows_written, 1);
        assert_eq!(result.engine, "stub");
        assert_eq!(
            result.builds.len(),
            1,
            "parallel=1 must not add a build row"
        );
        assert_eq!(result.queries.len(), 1);
        assert_eq!(result.queries[0].rows, 7);
    }

    /// Every measured query carries its own on-CPU seconds: the cost model
    /// prices compute only from a sampled figure, so a `None` here silently
    /// drops the query from the serving table instead of pricing it.
    #[test]
    fn timed_queries_record_measured_cpu_seconds() {
        let queries = [SqlQuery {
            name: "q",
            sql: "SELECT 1",
        }];
        let (result, _index) = run_sql_with_index::<StubEngine>(
            SqlRunConfig {
                iters: 4,
                parallel: 1,
            },
            &[],
            &queries,
        );
        // Unsampled on a host without per-thread CPU accounting; the sampler
        // is the thing under test only where it can report at all.
        if cpu::process_cpu_ns().is_some() {
            assert!(
                result.queries[0].cpu_s.is_some(),
                "timed loop must sample on-CPU seconds"
            );
        }
    }

    struct FlakyEngine;

    impl SqlEngine for FlakyEngine {
        type Index = StubIndex;
        fn name() -> &'static str {
            "flaky"
        }
        fn capabilities() -> Capabilities {
            Capabilities {
                sql: true,
                ..Default::default()
            }
        }
        fn create() -> Self::Index {
            StubIndex::default()
        }
        fn write(_index: &mut Self::Index, _rows: &[SqlRow<'_>]) {}
        fn parallel_write(_rows: &[SqlRow<'_>], _writers: usize) {}
        fn read(_index: &Self::Index, sql: &str) -> SqlOutput {
            assert_ne!(
                sql, "BAD",
                "a query DataFusion can't plan panics, like a real engine"
            );
            SqlOutput { rows: 1 }
        }
        fn close(_index: &mut Self::Index) {}
        fn delete(_index: Self::Index) {}
    }

    /// A panicking query FIRST in the battery must not take down the two
    /// that follow it: they still get measured with real timings, and the
    /// panic is recorded rather than dropped on the floor.
    #[test]
    fn unplannable_query_first_does_not_stop_the_rest() {
        let queries = [
            SqlQuery {
                name: "unplannable",
                sql: "BAD",
            },
            SqlQuery {
                name: "ok1",
                sql: "GOOD1",
            },
            SqlQuery {
                name: "ok2",
                sql: "GOOD2",
            },
        ];
        let (result, _index) = run_sql_with_index::<FlakyEngine>(
            SqlRunConfig {
                iters: 2,
                parallel: 1,
            },
            &[],
            &queries,
        );
        assert_eq!(
            result.queries.len(),
            2,
            "both surviving queries are measured, not just the ones after the panic ceases"
        );
        assert_eq!(result.queries[0].name, "ok1");
        assert_eq!(result.queries[1].name, "ok2");
        assert!(
            result.queries.iter().all(|q| q.rows == 1),
            "survivors carry the engine's real row count, not a placeholder"
        );
        assert_eq!(
            result.skipped.len(),
            1,
            "the panic is counted, not silently dropped"
        );
        assert_eq!(result.skipped[0].0, "unplannable");
        assert!(
            result.skipped[0]
                .1
                .contains("a query DataFusion can't plan"),
            "the panic message is preserved: {:?}",
            result.skipped[0].1
        );
    }

    struct MidLoopFlakyEngine;
    struct MidLoopFlakyIndex {
        calls: AtomicUsize,
    }

    /// Panics on the third call (the warm read plus the first timed
    /// iteration succeed; the second timed iteration panics) — the shape a
    /// state-poisoning bug takes: fine at first, then breaks partway
    /// through the timed loop rather than on the very first read.
    impl SqlEngine for MidLoopFlakyEngine {
        type Index = MidLoopFlakyIndex;
        fn name() -> &'static str {
            "mid-loop-flaky"
        }
        fn capabilities() -> Capabilities {
            Capabilities {
                sql: true,
                ..Default::default()
            }
        }
        fn create() -> Self::Index {
            MidLoopFlakyIndex {
                calls: AtomicUsize::new(0),
            }
        }
        fn write(_index: &mut Self::Index, _rows: &[SqlRow<'_>]) {}
        fn parallel_write(_rows: &[SqlRow<'_>], _writers: usize) {}
        fn read(index: &Self::Index, _sql: &str) -> SqlOutput {
            const PANICS_ON_CALL: usize = 3;
            let call = index.calls.fetch_add(1, Ordering::SeqCst) + 1;
            assert_ne!(call, PANICS_ON_CALL, "panics after the warm read succeeded");
            SqlOutput { rows: 1 }
        }
        fn close(_index: &mut Self::Index) {}
        fn delete(_index: Self::Index) {}
    }

    /// [`sql_driver::measure_sql`] must catch a panic from the timed loop's
    /// `E::read`, not only the warm read before it — otherwise a query that
    /// is fine on its first call but breaks on a later one kills the whole
    /// battery with zero diagnostics, worse than never catching panics at
    /// all.
    #[test]
    fn panic_mid_timed_loop_is_caught_not_fatal() {
        let queries = [SqlQuery {
            name: "mid-loop",
            sql: "Q",
        }];
        let (result, _index) = run_sql_with_index::<MidLoopFlakyEngine>(
            SqlRunConfig {
                iters: 3,
                parallel: 1,
            },
            &[],
            &queries,
        );
        assert!(
            result.queries.is_empty(),
            "a query that panics partway through has no stable timing to report"
        );
        assert_eq!(result.skipped.len(), 1);
        assert_eq!(result.skipped[0].0, "mid-loop");
    }

    /// The N-writer probe runs only above parallel=1, and receives the
    /// configured writer count.
    #[test]
    fn parallel_probe_runs_only_above_one_writer() {
        let rows = [SqlRow {
            doc_id: 0,
            title: "a",
            category: "c",
            score: 1,
        }];
        let (result, _index) = run_sql_with_index::<StubEngine>(
            SqlRunConfig {
                iters: 1,
                parallel: 4,
            },
            &rows,
            &[],
        );
        assert_eq!(result.builds.len(), 2);
        assert_eq!(result.builds[1].writers, 4);
    }
}
