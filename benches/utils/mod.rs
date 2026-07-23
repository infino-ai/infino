// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

// Shared bench library:
//
// - `corpus/` — synthetic data (stream + optional grading cache)
// - `ingest/` — stream corpus → append → commit → object storage
// - `fixture/` — one shared 10M ingest per process (`supertable_all`)
// - `superfile`, `supertable` — tier-specific bench runners by modality
// - `tiers`, `markdown`, `rss` — storage backends + reporting

pub mod cold_store;
pub mod corpus;
pub mod cost;
pub mod cpu;
pub mod dataset;
pub mod executors;
pub mod fixture;
pub mod harness;
pub mod ingest;
pub mod markdown;
pub mod report;
pub mod rss;
pub mod rustfs_server;
pub mod storage_meter;
pub mod storage_options;
pub mod tiers;

pub mod superfile;
pub mod supertable;

pub mod concurrent;
pub mod diag_common;
pub mod fts_diag;
pub mod scale;
pub mod sql_diag;
pub mod supertable_update;
pub mod tombstone_overhead;
pub mod unified_object_store;
