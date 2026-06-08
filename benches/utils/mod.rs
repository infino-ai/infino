// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

// Shared bench library:
//
// - `corpus/` — synthetic data (stream + optional grading cache)
// - `ingest/` — stream corpus → append → commit → object storage
// - `fixture/` — one shared 10M ingest per process (`supertable_all`)
// - `fts_superfile`, `vector_superfile` — 1M superfile bench bodies
// - `tiers`, `markdown`, `rss` — storage backends + reporting

pub mod corpus;
pub mod fixture;
pub mod harness;
pub mod ingest;
pub mod markdown;
pub mod report;
pub mod rss;
pub mod tiers;

pub mod fts_superfile;
pub mod sql_bench;
pub mod supertable_bench;
pub mod unified_object_store;
pub mod vector_superfile;
