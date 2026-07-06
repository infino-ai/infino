// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

// The crate-level docs are a Rust-focused page (Rust quickstart + API map),
// separate from the multi-language project README. The Rust quick example
// runs as a `cargo test --doc` doctest so it can't drift from the API. The
// Python/Node guides live in their own bindings and on the docs site.
#![doc = include_str!("../crate-docs.md")]
#![doc(
    html_logo_url = "https://infino.ai/docs/logo/infino-logo.png",
    html_favicon_url = "https://infino.ai/docs/favicon.png"
)]
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
// In a normal (non-`test-helpers`) build the internal layers (`config`,
// `storage`, the manifest + WAL + reader-cache + query stack) are
// `pub(crate)`. The curated public surface reaches a large part of them
// (`Connection` builds storage + creates/opens tables; `append` commits;
// the search methods query), but not all of it — the WAL lease/heartbeat
// machinery, cold-fetch cache tiers, config-file loading, and assorted
// deeper query/format helpers are only driven from paths the minimal
// public API doesn't exercise yet, so they read as dead here, and some
// test-facing re-exports go unused. Allow that *only* in this build mode:
// the `test-helpers` build — which CI compiles with `-D warnings` and
// which runs every test/bench — exercises those paths, so genuinely dead
// code (dead even under `test-helpers`) is still caught. Narrow or drop
// this as more of the surface (SQL, cache config) lands.
#![cfg_attr(not(feature = "test-helpers"), allow(dead_code, unused_imports))]

// `mimalloc` calls into a C runtime; miri can't execute foreign
// functions, so we fall back to the system allocator under miri.
// Production builds and tests not under miri keep mimalloc. Gated on the
// default-on `mimalloc` feature so an embedding loaded into a host
// process with its own allocator (the Python extension) can opt out — a
// second global allocator dlopened into a live process segfaults.
#[cfg(all(not(miri), feature = "mimalloc"))]
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

/// Visibility shim for items the layer-isolated integration tests and
/// benches — which are *separate* crates and so can only see `pub`
/// items — must call, but which are not part of the curated public
/// surface. Under `test-helpers` the item is `pub` (reachable through
/// the then-`pub` internal modules); in a normal build it is
/// `pub(crate)`, so it stays internally callable but off the public
/// API. The `cargo-public-api` snapshot is generated without
/// `test-helpers`, so these never enter the public contract.
macro_rules! test_visible {
    ($(#[$m:meta])* fn $($rest:tt)*) => {
        #[cfg(feature = "test-helpers")]
        $(#[$m])*
        pub fn $($rest)*
        #[cfg(not(feature = "test-helpers"))]
        $(#[$m])*
        pub(crate) fn $($rest)*
    };
}

// Internal layers. `pub` in a `test-helpers` build so the layer-isolated
// integration tests and benches can reach format/storage internals;
// `pub(crate)` otherwise, so the curated public surface is exactly the
// crate-root re-exports below. The `cargo-public-api` snapshot is taken
// without `test-helpers`, keeping these subtrees off the public contract.
#[cfg(feature = "test-helpers")]
pub mod config;
#[cfg(not(feature = "test-helpers"))]
pub(crate) mod config;

#[cfg(feature = "test-helpers")]
pub mod storage;
#[cfg(not(feature = "test-helpers"))]
pub(crate) mod storage;

#[cfg(feature = "test-helpers")]
pub mod superfile;
#[cfg(not(feature = "test-helpers"))]
pub(crate) mod superfile;

#[cfg(feature = "test-helpers")]
pub mod supertable;
#[cfg(not(feature = "test-helpers"))]
pub(crate) mod supertable;

// `roaring` is already an internal dependency. Re-export it under
// `test-helpers` only, so a bench can build an allow-set for the filtered
// vector kernel without its own `roaring` dependency. Off the public
// contract (the `cargo-public-api` snapshot is taken without the feature).
#[cfg(feature = "test-helpers")]
pub use roaring;

// The catalog layer (`Connection` + `connect`). Internal module; its
// public items are re-exported at the crate root below.
mod catalog;
mod error;
mod memory;
mod runtime_bridge;

// ---- Curated public surface ----

/// Catalog entry points and handle: open a `Connection`, then create /
/// open / drop / list tables.
pub use catalog::{ColdFetchMode, ConnectOptions, Connection, IndexSpec, connect, connect_with};
pub use config::{CompactionSettings, OptimizeOptions};
/// The single public error type for the curated API.
pub use error::InfinoError;
/// Value types named by the public method signatures.
pub use superfile::VectorSearchOptions;
pub use superfile::{fts::reader::BoolMode, vector::distance::Metric};
/// Single-table handle: `append` / `update` / `delete` / `bm25_search`
/// / `vector_search` / `schema`.
pub use supertable::Supertable;
pub use supertable::{
    GcError, GcReport, MutationStats, OptimizeError, query::vector::VectorFilter,
};

/// Convenience builders for test fixtures. Visible to:
///   - Unit tests (via `cfg(test)` — always on for `cargo test`)
///   - Integration tests (via `cargo test --features test-helpers`,
///     wired into the `Makefile`)
///   - Benches (which pull `test-helpers` in transitively through
///     the `infino-bench-utils` dev-dependency)
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
