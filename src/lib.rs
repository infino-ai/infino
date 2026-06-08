// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! infino — search-optimized lakehouse format.
//!
//! A superfile is a valid Apache Parquet file with embedded BM25 + vector
//! indexes. The Parquet portion is readable by vanilla DataFusion / DuckDB
//! / pyarrow; the embedded blobs are accessible via [`SuperfileReader`].

// `coverage_nightly` is set by `cargo +nightly llvm-cov`. Under it we opt
// into `#[coverage(off)]` annotations on stable-uncoverable error paths
// (OOM handlers, overflow guards). On stable the feature flag is inert
// and the annotations become no-ops.
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
// No `.unwrap()` anywhere — including tests and benches. Production
// code uses `?` for fallible operations or
// `.expect("invariant: ...")` for paths that are infallible by
// construction. Test/bench code uses `.expect("description")` so a
// failing test panic message tells you which step broke without
// having to count line numbers in the source. The integration tests
// in `tests/` and benches in `benches/` are separate crates; the
// lint is reasserted there via inner attributes.
#![deny(clippy::unwrap_used)]
// `doc_lazy_continuation` fires across a lot of existing doc comments
// where a paragraph wraps a leading punctuation token (`+`, `-`) and
// rustdoc's Markdown parser treats it as a list-item start. The
// rendered docs are fine; rewording each site would distort prose.
// Allowed crate-wide as a style decision.
#![allow(clippy::doc_lazy_continuation)]
// `type_complexity` flags reader-cache and manifest-aggregate state
// types that are intentionally nested. Factoring them into aliases
// adds indirection without clarity at the call sites. Allowed
// crate-wide; revisit when the underlying state shapes stabilize.
#![allow(clippy::type_complexity)]
// `too_many_arguments` flags `disk.rs::finalize_to_mmap` which has 8
// parameters by design (each captures a distinct stage hand-off).
// Restructuring into a builder adds boilerplate without clarity.
#![allow(clippy::too_many_arguments)]

// `mimalloc` calls into a C runtime; miri can't execute foreign
// functions, so we fall back to the system allocator under miri.
// Production builds and tests not under miri keep mimalloc.
#[cfg(not(miri))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Compile-time-baked writer identification, written to `inf.builder` KV.
/// Format: `infino/<crate-version>+<git-short-hash>[-dirty]`, or `…+unknown`
/// when built outside a git checkout (e.g. crates.io). Captured at build time
/// by `build.rs`; not user-overridable.
pub const BUILDER_ID: &str = concat!(
    "infino/",
    env!("CARGO_PKG_VERSION"),
    "+",
    env!("INFINO_GIT_HASH")
);

pub mod config;
mod runtime_bridge;
pub mod storage;
pub mod superfile;
pub mod supertable;

/// Convenience builders for test fixtures. Visible to:
///   - Unit tests (via `cfg(test)` — always on for `cargo test`)
///   - Integration tests + benches (via the `test-helpers`
///     feature, auto-enabled by the `dev-dependencies` self-
///     reference in `Cargo.toml`)
///
/// NOT part of infino's stable API. Signatures may change.
#[cfg(any(test, feature = "test-helpers"))]
pub mod test_helpers;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_id_starts_with_crate_name_and_version() {
        assert!(BUILDER_ID.starts_with("infino/"));
        let crate_ver = env!("CARGO_PKG_VERSION");
        assert!(BUILDER_ID.starts_with(&format!("infino/{crate_ver}+")));
    }

    #[test]
    fn builder_id_contains_git_hash_or_unknown() {
        // Either a real short hash, "unknown", or those plus "-dirty".
        let after_plus = BUILDER_ID.split('+').nth(1).expect("has +<hash>");
        assert!(!after_plus.is_empty());
    }
}
