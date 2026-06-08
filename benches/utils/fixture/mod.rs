// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Process-wide shared bench state (build once per `cargo bench` process).
//!
//! Not Infino product terminology — just "expensive setup reused by every
//! search group" (one object-storage ingest + one search consumer).

pub mod supertable;
