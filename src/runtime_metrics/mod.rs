// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! In-engine metering signals for benches and `features = ["metering"]`.
//!
//! Three resource families — parallel names, one ownership home:
//! - [`io`] — object-store request/byte ledger (+ background attribution)
//! - [`cpu`] — process on-CPU time
//! - [`rss`] — process resident set / peak sampler
//!
//! Storage providers `record_*` into [`UsageMeter`]; they do not own it.
//! Benches and platform must use these APIs rather than reimplement
//! `/proc` parsers or parallel I/O counters.

pub mod cpu;
pub mod io;
pub mod rss;

pub use io::{
    ClassIo, N_URI_CLASSES, TraceEntry, UriClass, UsageMeter, UsageSnapshot, io_is_background,
    scope_background,
};
