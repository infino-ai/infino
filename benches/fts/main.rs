// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! FTS superfile bench bundle (infino-only entry point). Supertable FTS
//! benches live in `benches/supertable/main.rs`, where they share one
//! combined 10M-row supertable with the vector supertable benches.
//!
//! The comparable build + search numbers are measured through the
//! engine-generic harness (`run_fts::<InfinoFtsEngine>`) — the same path
//! the cross-engine comparison uses. The
//! infino-only extras (correctness oracle, per-algorithm probe,
//! rayon-sharded build, cold S3 tier) are layered on top.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench superfile_fts                          # build + search
//! cargo bench --bench superfile_fts -- superfile_fts_build   # ingest only
//! cargo bench --bench superfile_fts -- superfile_fts_search  # search only
//! INFINO_BENCH_UPDATE_README=1 cargo bench --bench superfile_fts
//! ```

fn main() {
    infino_bench_utils::fts_superfile::run();
}
