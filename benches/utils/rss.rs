// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Thin re-export of the engine RSS / peak sampler.
//!
//! Implementation lives in [`infino::runtime_metrics::rss`] — benches must
//! not keep a parallel `/proc` parser.

pub use infino::runtime_metrics::rss::*;
