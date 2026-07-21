// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Process CPU and RSS sampling used by benches and optional usage flush.
//!
//! Process-wide attribution only — not a Stripe dimension. Benches must call
//! these APIs rather than reimplement `/proc` parsers.

pub mod cpu;
pub mod rss;
