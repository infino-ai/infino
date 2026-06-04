// Shared bench library:
//
// - `corpus/` — synthetic data (stream + optional grading cache)
// - `ingest/` — stream corpus → append → commit → object storage
// - `fixture/` — one shared 10M ingest per process (`supertable_all`)
// - `bench/` — criterion groups only (ingest timing, FTS search, vector search)
// - `fts_superfile`, `vector_superfile` — 1M superfile bench bodies
// - `tiers`, `markdown`, `rss` — storage backends + reporting

pub mod bench;
pub mod corpus;
pub mod fixture;
pub mod ingest;
pub mod markdown;
pub mod rss;
pub mod tiers;

pub mod fts_superfile;
pub mod unified_object_store;
pub mod vector_superfile;
