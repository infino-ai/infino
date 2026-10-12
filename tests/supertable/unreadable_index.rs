// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A table whose full-text index is older than this engine reads reaches a
//! caller as `Unsupported`, with the instruction that fixes it, on every
//! public path that opens its superfiles: the searches, SQL, and the
//! maintenance that rewrites them. Checked both reading straight from
//! storage and through a disk cache, since each wraps a failed open in its
//! own error.

use infino::{
    Bm25SearchOptions, ColdFetchMode, ConnectOptions, Connection, InfinoError, OptimizeError,
    OptimizeOptions, ReindexError, ReindexOptions, Supertable, connect_with,
    superfile::format::{
        footer::read_kv_metadata,
        fts::{U32_BYTES, VERSION_MIN, hdr::VERSION_OFF},
        kv,
    },
};
use tempfile::TempDir;

use crate::reindex_fixture::{N_DOCS, TABLE, edit_superfiles, probe_embedding, write_table};

/// Neighbours the vector search asks for.
const PROBE_NEIGHBOURS: usize = 16;

/// What every refusal tells the caller to do.
const FIX: &str = "reindex";

/// Where a reopened table reads its superfiles' bytes from.
#[derive(Debug, Clone, Copy)]
enum Tier {
    /// Straight from storage: no disk cache configured.
    Storage,
    /// Through a disk cache that fetches whole files. The default lazy mode
    /// opens from the header bytes the manifest carries, which the on-disk
    /// edit below does not reach.
    DiskCache,
}

/// A table with an index too old to read, reopened on a fresh connection.
struct TooOldTable {
    _root: TempDir,
    _cache: TempDir,
    db: Connection,
    table: Supertable,
}

/// `bytes` with its FTS blob stamped one version below the oldest this
/// engine reads.
fn stamp_too_old(bytes: &[u8]) -> Vec<u8> {
    let offset: usize = read_kv_metadata(bytes)
        .expect("read superfile key-value metadata")
        .get(kv::FTS_OFFSET)
        .expect("every fixture superfile carries an FTS blob")
        .parse()
        .expect("offset is a number");
    let mut out = bytes.to_vec();
    let at = offset + VERSION_OFF;
    out[at..at + U32_BYTES].copy_from_slice(&(VERSION_MIN - 1).to_le_bytes());
    out
}

/// Write the fixture, stamp every superfile too old, and open the table
/// on a fresh connection reading through `tier`.
fn open_too_old(tier: Tier) -> TooOldTable {
    let root = TempDir::new().expect("tempdir");
    let cache = TempDir::new().expect("cache tempdir");
    write_table(root.path());
    edit_superfiles(root.path(), stamp_too_old);
    let options = match tier {
        Tier::Storage => ConnectOptions::new(),
        Tier::DiskCache => ConnectOptions::new()
            .with_cache_dir(cache.path())
            .with_cold_fetch_mode(ColdFetchMode::HybridWithPrefetch),
    };
    let db = connect_with(root.path().to_str().expect("utf-8 path"), options).expect("connect");
    let table = db.open_table(TABLE).expect("open the stamped table");
    TooOldTable {
        _root: root,
        _cache: cache,
        db,
        table,
    }
}

fn assert_unsupported(tier: Tier, what: &str, err: &InfinoError) {
    assert!(
        matches!(err, InfinoError::Unsupported(_)),
        "{tier:?} {what}: expected Unsupported, got {err:?}"
    );
    assert!(err.to_string().contains(FIX), "{tier:?} {what}: {err}");
}

fn assert_every_path_unsupported(tier: Tier) {
    let too_old = open_too_old(tier);
    let (db, table) = (&too_old.db, &too_old.table);

    let bm25 = table
        .bm25_search(
            "body",
            "common",
            N_DOCS as usize,
            Bm25SearchOptions::new(),
            None,
        )
        .expect_err("bm25_search over an index too old to read");
    assert_unsupported(tier, "bm25_search", &bm25);
    let knn = table
        .vector_search(
            "emb",
            &probe_embedding(),
            PROBE_NEIGHBOURS,
            None,
            Some(&["_id", "body"]),
        )
        .expect_err("vector_search over an index too old to read");
    assert_unsupported(tier, "vector_search", &knn);
    let sql = db
        .query_sql(&format!("SELECT body FROM {TABLE}"))
        .expect_err("query_sql over an index too old to read");
    assert_unsupported(tier, "query_sql", &sql);

    let reindex = table
        .reindex(&ReindexOptions::rewriting())
        .expect_err("reindex of an index too old to read");
    assert!(
        matches!(reindex, ReindexError::Unsupported(_)) && reindex.to_string().contains(FIX),
        "{tier:?} reindex: {reindex:?}"
    );
    let optimize = table
        .optimize(&OptimizeOptions::default())
        .expect_err("optimize of an index too old to read");
    assert!(
        matches!(optimize, OptimizeError::Unsupported(_)) && optimize.to_string().contains(FIX),
        "{tier:?} optimize: {optimize:?}"
    );
}

#[test]
fn an_index_too_old_to_read_is_unsupported_reading_from_storage() {
    assert_every_path_unsupported(Tier::Storage);
}

#[test]
fn an_index_too_old_to_read_is_unsupported_through_a_disk_cache() {
    assert_every_path_unsupported(Tier::DiskCache);
}
