// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Shared cold-store measurement recipe for FTS / SQL / vector.
//!
//! One open / first / steady-second / repeat windowing path — modalities
//! only supply the query closures. Open is **consumer construct only**
//! (no `open_all_superfiles` inside the open window).

use std::time::Instant;

use infino::storage::io_counters;

use crate::{
    cpu,
    storage_meter::{ColdStoreSplit, MeteredStorage, ObjectStoreMeter, background_fill_meter},
};

/// Distinct steady-cold samples; the wall-median sample's wall / CPU / I/O
/// are reported together (same constant for every modality).
pub const STEADY_COLD_SAMPLES: usize = 3;

/// Timed + metered cold-store windows shared across modalities.
#[derive(Debug, Clone)]
pub struct ColdStoreMeasurement {
    pub split: ColdStoreSplit,
    pub open_wall_s: f64,
    pub open_cpu_s: Option<f64>,
    pub first_wall_s: f64,
    pub first_cpu_s: Option<f64>,
    pub second_wall_s: f64,
    pub second_cpu_s: Option<f64>,
}

/// Run the shared cold recipe.
///
/// - `open_consumer`: build the cold consumer (counted in the open window).
/// - `run_first`: first cold query (metadata warmup).
/// - `run_steady`: iterator of distinct steady queries (up to
///   [`STEADY_COLD_SAMPLES`]); each is timed separately.
/// - `run_repeat`: first query repeated (fill-lag probe).
/// - `settle_fills`: wait for every background fill the run started, so the fill row counts them
///   all, not only those that ran inside a query window.
pub fn measure_cold_store<C>(
    meter: &MeteredStorage,
    open_consumer: impl FnOnce() -> C,
    run_first: impl FnOnce(&C),
    mut run_steady: impl FnMut(&C, usize),
    n_steady: usize,
    run_repeat: impl FnOnce(&C),
    settle_fills: impl FnOnce(&C),
) -> ColdStoreMeasurement {
    // Every window is a delta from the connection meter. Never stash an
    // absolute snapshot in `ColdStoreSplit.open` — that used to pull ingest
    // PUTs into the "cold table open" cost row.
    let before_open = meter.snapshot();
    let open_cpu0 = cpu::process_cpu_ns();
    let open_started = Instant::now();
    let consumer = open_consumer();
    let open_wall_s = open_started.elapsed().as_secs_f64();
    let open_cpu_s = cpu::cpu_seconds_since(open_cpu0);
    let after_open = meter.snapshot();
    let open = after_open.since(&before_open);

    io_counters::timeline_reset();
    let first_cpu0 = cpu::process_cpu_ns();
    let first_started = Instant::now();
    run_first(&consumer);
    let first_wall_s = first_started.elapsed().as_secs_f64();
    let first_cpu_s = cpu::cpu_seconds_since(first_cpu0);
    for span in io_counters::timeline_take() {
        eprintln!(
            "[cold-timeline] first {} {} off={} len={} {}us{}",
            span.op,
            span.uri,
            span.off,
            span.len,
            span.end_us.saturating_sub(span.start_us),
            if span.background { " (bg)" } else { "" },
        );
    }
    let mut window_start = meter.snapshot();
    let first_query = window_start.since(&after_open);

    let n = n_steady.clamp(1, STEADY_COLD_SAMPLES);
    let mut steady: Vec<(f64, Option<f64>, ObjectStoreMeter)> = Vec::with_capacity(n);
    for i in 0..n {
        // Per-op timeline for the steady windows (INFINO_IO_TIMELINE=1):
        // dumps every object-store op's uri/offset/len so a GET-count
        // regression can be attributed to the exact reads, not inferred.
        io_counters::timeline_reset();
        let cpu0 = cpu::process_cpu_ns();
        let started = Instant::now();
        run_steady(&consumer, i);
        let wall_s = started.elapsed().as_secs_f64();
        let cpu_s = cpu::cpu_seconds_since(cpu0);
        for span in io_counters::timeline_take() {
            eprintln!(
                "[cold-timeline] steady[{i}] {} {} off={} len={} {}us{}",
                span.op,
                span.uri,
                span.off,
                span.len,
                span.end_us.saturating_sub(span.start_us),
                if span.background { " (bg)" } else { "" },
            );
        }
        let now = meter.snapshot();
        steady.push((wall_s, cpu_s, now.since(&window_start)));
        window_start = now;
    }
    let after_steady = window_start;

    run_repeat(&consumer);
    let after_repeat = meter.snapshot();
    // Fills outlive the queries that started them, so the last snapshot waits for them. The fills
    // keep running while it waits; the wait adds none of its own. A fill still in its defer window
    // goes at once instead of after it, which changes when it runs, not what it fetches.
    let settle_started = Instant::now();
    settle_fills(&consumer);
    let after_settle = meter.snapshot();
    eprintln!(
        "[cold-settle] background fills settled {:.2} s after the last query",
        settle_started.elapsed().as_secs_f64()
    );

    // One median sample owns wall, CPU, and I/O together so the cost model
    // never pairs one run's latency with another's GET count.
    steady.sort_unstable_by(|(wall_a, _, _), (wall_b, _, _)| f64::total_cmp(wall_a, wall_b));
    let (median_wall_s, median_cpu_s, median_io) = {
        let (wall_s, cpu_s, io) = &steady[steady.len() / 2];
        (*wall_s, *cpu_s, *io)
    };

    ColdStoreMeasurement {
        split: ColdStoreSplit {
            open,
            first_query,
            second_query: median_io,
            repeat_query: after_repeat.since(&after_steady),
            fill: background_fill_meter(&after_settle.since(&before_open)),
            fill_after_queries: background_fill_meter(&after_settle.since(&after_repeat)),
        },
        open_wall_s,
        open_cpu_s,
        first_wall_s,
        first_cpu_s,
        second_wall_s: median_wall_s,
        second_cpu_s: median_cpu_s,
    }
}

#[cfg(test)]
mod tests {
    use std::{ops::Range, sync::Arc};

    use bytes::Bytes;
    use infino::{
        runtime_metrics::{UsageMeter, scope_background},
        storage::{LocalFsStorageProvider, StorageProvider},
    };
    use tokio::runtime::Runtime;

    use crate::{cold_store::measure_cold_store, storage_meter::wrap};

    /// The object the fake cold run reads.
    const OBJECT: &str = "seg/x.bin";
    /// Its bytes.
    const OBJECT_BYTES: &[u8] = b"0123456789";
    /// The range a query reads.
    const QUERY_RANGE: Range<u64> = 0..4;
    /// The range a fill reads.
    const FILL_RANGE: Range<u64> = 4..10;
    /// Fills in the run: one during the first query, one after the last.
    const FILLS: u64 = 2;

    /// A cold run's fill row counts the fills that run after its last query, not only those inside
    /// a query window, and the query rows count only the queries' own GETs.
    #[test]
    fn the_fill_row_counts_fills_until_they_settle() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let runtime = Runtime::new().expect("runtime");
        let provider: Arc<dyn StorageProvider> = Arc::new(
            LocalFsStorageProvider::new_with_meter(dir.path(), UsageMeter::new()).expect("localfs"),
        );
        runtime
            .block_on(provider.put_atomic(OBJECT, Bytes::from_static(OBJECT_BYTES)))
            .expect("put");
        let meter = wrap(Arc::clone(&provider));
        // A query's own GET; a fill's GET, which runs as background I/O.
        let query = || {
            runtime
                .block_on(provider.get_range(OBJECT, QUERY_RANGE))
                .expect("query GET");
        };
        let fill = || {
            runtime
                .block_on(scope_background(provider.get_range(OBJECT, FILL_RANGE)))
                .expect("fill GET");
        };

        // One fill runs during the first query; one more is still running when the repeat ends.
        let measured = measure_cold_store(
            &meter,
            || (),
            |_| {
                query();
                fill();
            },
            |_, _| query(),
            1,
            |_| query(),
            |_| fill(),
        );

        assert_eq!(
            measured.split.first_query.get_count, 1,
            "the query's own GET"
        );
        assert_eq!(
            measured.split.repeat_query.get_count, 1,
            "the repeat's own GET"
        );
        assert_eq!(
            measured.split.fill.get_count, FILLS,
            "both fills, until they settled"
        );
        assert_eq!(
            measured.split.fill.get_bytes,
            FILLS * (FILL_RANGE.end - FILL_RANGE.start),
            "and their bytes"
        );
        assert_eq!(
            measured.split.fill_after_queries.get_count,
            FILLS - 1,
            "the one that ran after the last query"
        );
    }
}
