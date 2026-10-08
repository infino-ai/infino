// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Supertable-layer integration tests.
//!
//! One test binary (`cargo test --test supertable`) covers:
//!
//! - **commit/**: the writer's append + commit pipeline,
//!   manifest-id increment, pointer-atomic publish, id
//!   uniqueness across threads, open-then-refresh, partition
//!   assignment, in-process concurrency, stats accessor.
//! - **query/**: hierarchical-manifest query path, skip-
//!   pruning end-to-end, brute-force BM25 oracle for
//!   multi-superfile search.
//! - **manifest/**: the eager-vs-lazy-open threshold path.
//! - **compact_gc / gc_stale_snapshot**: reclaiming orphaned
//!   objects, and the keep-set freshness that decides which
//!   objects those are.
//! - **disk_cache/**: the cold-fetch coordinator + hybrid /
//!   sweep policies + supertable-disk-cache integration.
//! - **storage/**: the supertable-driven S3 smoke run.
//!
//! - **reindex / reindex_invariance**: planning and repairing a
//!   stale table, and what a repair must carry across untouched,
//!   over the stale tables `reindex_fixture` writes.
//! - **reindex_crash**: killing a reindex mid-run and resuming
//!   it. Spawn-self, but bundled here rather than given its own
//!   top-level binary like `supertable_commit_crash_localfs.rs`:
//!   it shares `reindex_fixture` and the invariance helpers, and
//!   duplicating them to keep the file at the top level would cost
//!   more than it buys. `--exact` names the one test the child
//!   re-enters.
//! - **unreadable_index**: a table whose full-text index is too
//!   old to read refuses every public read and maintenance path
//!   as `Unsupported`, over the same fixture.
//!
//! Spawn-self tests
//! (`supertable_commit_crash_localfs.rs`,
//! `supertable_concurrent_processes.rs`) and the workspace
//! audit (`license_audit.rs`) stay at the top level of
//! `tests/` because they need their own binary.

#![deny(clippy::unwrap_used)]

mod bioasq_admit_diag;
mod commit;
mod compact_gc;
mod disk_cache;
mod drain_tombstones;
mod gc_stale_snapshot;
mod manifest;
mod query;
mod reindex;
mod reindex_crash;
mod reindex_fixture;
mod reindex_invariance;
mod schema;
mod storage;
mod unreadable_index;
mod update_crash_property;
mod vector_cosine_normalize;
mod vector_law_serving;
mod vector_phase_profile;
mod writer_mutations;
