//! Combined 10M FTS + vector supertable benchmarks (`supertable_all`).

pub mod query;

pub mod fts_search;
pub mod ingest;
pub mod vector_search;

use criterion::criterion_group;

criterion_group!(
    benches,
    ingest::bench,
    vector_search::bench,
    fts_search::bench
);
