// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Shared cold-store measurement recipe for FTS / SQL / vector.
//!
//! One open / first / steady-second / repeat windowing path — modalities
//! only supply the query closures. Open is **consumer construct only**
//! (no `open_all_superfiles` inside the open window).

use std::time::Instant;

use crate::{
    cpu,
    storage_meter::{ColdStoreSplit, MeteredStorage, ObjectStoreMeter},
};

/// Distinct steady-cold samples; median wall and median-GET I/O sample
/// are reported (same constant for every modality).
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
pub fn measure_cold_store<C>(
    meter: &MeteredStorage,
    open_consumer: impl FnOnce() -> C,
    run_first: impl FnOnce(&C),
    mut run_steady: impl FnMut(&C, usize),
    n_steady: usize,
    run_repeat: impl FnOnce(&C),
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

    let first_cpu0 = cpu::process_cpu_ns();
    let first_started = Instant::now();
    run_first(&consumer);
    let first_wall_s = first_started.elapsed().as_secs_f64();
    let first_cpu_s = cpu::cpu_seconds_since(first_cpu0);
    let mut window_start = meter.snapshot();
    let first_query = window_start.since(&after_open);

    let n = n_steady.max(1).min(STEADY_COLD_SAMPLES);
    let mut steady: Vec<(f64, Option<f64>, ObjectStoreMeter)> = Vec::with_capacity(n);
    for i in 0..n {
        let cpu0 = cpu::process_cpu_ns();
        let started = Instant::now();
        run_steady(&consumer, i);
        let wall_s = started.elapsed().as_secs_f64();
        let cpu_s = cpu::cpu_seconds_since(cpu0);
        let now = meter.snapshot();
        steady.push((wall_s, cpu_s, now.since(&window_start)));
        window_start = now;
    }
    let after_steady = window_start;

    run_repeat(&consumer);
    let after_repeat = meter.snapshot();

    let median_wall_s = {
        let mut walls: Vec<f64> = steady.iter().map(|(wall, _, _)| *wall).collect();
        walls.sort_unstable_by(f64::total_cmp);
        walls[walls.len() / 2]
    };
    let (_, median_cpu_s, median_io) = {
        steady.sort_unstable_by_key(|(_, _, io)| io.get_count);
        steady[steady.len() / 2]
    };

    ColdStoreMeasurement {
        split: ColdStoreSplit {
            open,
            first_query,
            second_query: median_io,
            repeat_query: after_repeat.since(&after_steady),
        },
        open_wall_s,
        open_cpu_s,
        first_wall_s,
        first_cpu_s,
        second_wall_s: median_wall_s,
        second_cpu_s: median_cpu_s,
    }
}
