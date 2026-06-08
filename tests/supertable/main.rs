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
//!   multi-segment search.
//! - **manifest/**: the eager-vs-lazy-open threshold path.
//! - **disk_cache/**: the cold-fetch coordinator + hybrid /
//!   sweep policies + supertable-disk-cache integration.
//! - **storage/**: the supertable-driven S3 smoke run.
//!
//! Spawn-self tests
//! (`supertable_commit_crash_localfs.rs`,
//! `supertable_concurrent_processes.rs`) and the workspace
//! audit (`license_audit.rs`) stay at the top level of
//! `tests/` because they need their own binary.

mod commit;
mod disk_cache;
mod manifest;
mod query;
mod storage;
mod update_crash_property;
mod writer_mutations;
