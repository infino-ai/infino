// Shared bench library. Holds:
//
// - Helpers (corpus, markdown, rss) consumed by every bench body.
// - The bench bodies themselves (fts_*, vector_*), so each
//   `[[bench]]` binary under benches/{fts,vector}/main*.rs is a
//   2-line shell that imports the body via `use` and hands the
//   criterion group to `criterion::criterion_main!`.
//
// File layout for the bench bodies follows `<topic>_<layer>.rs`
// so the module name matches the criterion-binary name (`fts`,
// `fts-superfile`, `fts-supertable`, `vector`) verbatim.

pub mod corpus;
pub mod markdown;
pub mod rss;

pub mod fts_superfile;
pub mod fts_supertable;
pub mod vector_superfile;
pub mod vector_supertable;
