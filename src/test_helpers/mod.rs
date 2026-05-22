//! Convenience builders for test fixtures.
//!
//! Three test contexts share these helpers:
//!
//! - **Unit tests** (`#[cfg(test)] mod tests` inside `src/`)
//!   reach this module via `crate::test_helpers::...` —
//!   `cfg(test)` always enables it.
//! - **Integration tests** (`tests/...`) reach it via
//!   `infino::test_helpers::...` — the `test-helpers` Cargo
//!   feature is auto-enabled by the `dev-dependencies` self-
//!   reference in `Cargo.toml`.
//! - **Benches** (`benches/...`) reach it the same way.
//!
//! Scope: small atomic idioms that repeat across dozens of
//! test / bench fixtures (Decimal128 id construction, default
//! tokenizer, default vector config). Higher-level "build a
//! test corpus" / "build a full test superfile" stays in the
//! test files themselves — those vary too much per scenario
//! to share usefully.
//!
//! [`brute_force_bm25`] is the textbook BM25 reference impl
//! used as the FTS correctness oracle.

pub mod brute_force_bm25;

use std::sync::Arc;

use arrow_array::{Decimal128Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

use crate::superfile::builder::{FtsConfig, VectorConfig};
use crate::superfile::fts::tokenize::{AsciiLowerTokenizer, Tokenizer};
use crate::superfile::vector::distance::Metric;
use crate::supertable::SupertableOptions;

/// Build a `Decimal128Array(38, 0)` from `u64` ids.
///
/// Centralizes the verbose three-step construction that
/// every test fixture reinvents:
///
/// ```ignore
/// Decimal128Array::from(ids.into_iter().map(|v| v as i128).collect::<Vec<_>>())
///     .with_precision_and_scale(38, 0)
///     .expect("decimal128")
/// ```
pub fn decimal128_ids<I: IntoIterator<Item = u64>>(ids: I) -> Decimal128Array {
    Decimal128Array::from(ids.into_iter().map(|v| v as i128).collect::<Vec<_>>())
        .with_precision_and_scale(38, 0)
        .expect("Decimal128(38, 0) is a valid precision/scale pair")
}

/// `Field` for the primary-key id column — `Decimal128(38, 0)`,
/// non-nullable. Caller supplies the column name (typically
/// `"_id"` at the supertable layer or `"doc_id"` in lower-level
/// superfile fixtures).
pub fn decimal128_id_field(name: &str) -> Field {
    Field::new(name, DataType::Decimal128(38, 0), false)
}

/// The default tokenizer used in tests + benches:
/// `AsciiLowerTokenizer` wrapped in `Arc<dyn Tokenizer>`.
///
/// Callers passing this into `BuilderOptions::new` wrap in
/// `Some(...)` at the call site:
///
/// ```ignore
/// BuilderOptions::new(schema, "doc_id", fts_cols, vec_cols,
///                     Some(default_tokenizer()));
/// ```
pub fn default_tokenizer() -> Arc<dyn Tokenizer> {
    Arc::new(AsciiLowerTokenizer)
}

/// Default `VectorConfig` for test fixtures: `dim=16`,
/// `n_cent=4`, `metric=Cosine`. Caller supplies the column
/// name and `rot_seed` — the only fields tests vary.
///
/// For realistic-scale vectors (e.g. `dim=384` in benches),
/// callers construct `VectorConfig` directly with their own
/// values.
pub fn default_vector_config(column: &str, rot_seed: u64) -> VectorConfig {
    VectorConfig {
        column: column.into(),
        dim: 16,
        n_cent: 4,
        rot_seed,
        metric: Metric::Cosine,
    }
}

/// Single-column user schema with `title: LargeUtf8`.
///
/// Mirrors the supertable's auto-`_id` model: the supertable
/// layer prepends `_id: Decimal128(38, 0)` automatically at
/// append time, so the user-facing schema only declares the
/// payload columns. Dozens of supertable tests reconstruct
/// this exact schema; centralizing keeps the
/// supertable-auto-injects-id contract in one place.
pub fn schema_id_title() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new(
        "title",
        DataType::LargeUtf8,
        false,
    )]))
}

/// Build a single-column `RecordBatch` of titles matching
/// [`schema_id_title`]. Caller supplies the title strings;
/// the rest is fixed.
pub fn build_title_batch(titles: &[&str]) -> RecordBatch {
    let titles_arr = LargeStringArray::from(titles.to_vec());
    RecordBatch::try_new(schema_id_title(), vec![Arc::new(titles_arr)])
        .expect("RecordBatch shape matches schema_id_title")
}

/// `SupertableOptions` with the test-fixture defaults:
/// [`schema_id_title`] schema, a single FTS column `title`,
/// no vector columns, and a 1-thread rayon writer pool.
///
/// Caller chains `.with_storage(...)` / `.with_disk_cache(...)`
/// / `.with_*(...)` for whatever the specific test needs.
/// Returning the un-storage-d shape lets each test decide
/// explicitly whether to attach storage.
pub fn default_supertable_options() -> SupertableOptions {
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("rayon ThreadPoolBuilder with num_threads(1) builds"),
    );
    SupertableOptions::new(
        schema_id_title(),
        vec![FtsConfig {
            column: "title".into(),
        }],
        vec![],
        Some(default_tokenizer()),
    )
    .expect("SupertableOptions::new with default test fixture args")
    .with_writer_pool(pool)
}
