// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The checked-in corpus tables, each written by a real published engine
//! release, must keep the format shape they were generated for and must
//! still open and rank under the current engine.
//!
//! The tables themselves are generated, not committed — a few megabytes of
//! fixture bytes that `tests/corpus/generate.sh` rebuilds from the pinned
//! generators. A checkout without them skips these tests with a note rather
//! than failing; set `INFINO_CORPUS_REQUIRED=1` (as CI should) to turn a
//! missing corpus into a failure instead of a silent pass.
//!
//! Two jobs, and the second is the less obvious one. The shape assertions
//! stop a regenerated corpus from silently drifting into a weaker shape —
//! a too-sparse corpus makes an old builder stamp a lower version, and a
//! too-small one leaves a structure the version implies with nothing in
//! it. The recall assertions pin the tokenization defect *in the negative*:
//! these terms are unreachable in every shipped file, and a reindex that
//! re-analyzes is what makes them reachable. Without them a reanalysis
//! test would pass for the wrong reason.

use std::{fs, path::Path};

use infino::{Bm25SearchOptions, Supertable, connect};
use tempfile::TempDir;

/// Where the generated tables live, relative to the crate root.
const CORPUS_ROOT: &str = "tests/corpus/tables";
/// Set this to fail rather than skip when the corpus has not been
/// generated, so an environment that is supposed to have one cannot pass
/// these tests by having nothing to check.
const REQUIRED_ENV: &str = "INFINO_CORPUS_REQUIRED";
/// Table name every generator writes, so a test needs no per-shape name.
const TABLE: &str = "corpus";

/// Documents per generated table; mirrors the generators' shared corpus.
pub(crate) const N_DOCS: usize = 12_000;

/// A superfile holding at least this many documents gives a term present
/// in every document more than `BLOCK_LEN * COARSE_BLOCK_MAX_SPAN` (128 *
/// 32) postings, so its coarse block-max table holds more than one entry.
/// Below it, a file can carry the version that implies a coarse table
/// while that table summarises a single span.
const DOCS_FOR_MULTI_ENTRY_COARSE: u32 = 4096;

/// 8-byte magic at the start of an FTS blob; the version is the `u32`
/// immediately after it.
const FTS_MAGIC: &[u8; 8] = b"INFFTS01";

/// The directory holding one shape's table, or `None` when the corpus has
/// not been generated.
///
/// Returns `None` only when [`REQUIRED_ENV`] is unset; otherwise a missing
/// corpus is the failure it would be in CI.
pub(crate) fn corpus_dir(shape: &str) -> Option<std::path::PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(CORPUS_ROOT)
        .join(shape);
    if dir.is_dir() {
        return Some(dir);
    }
    assert!(
        std::env::var_os(REQUIRED_ENV).is_none(),
        "{REQUIRED_ENV} is set but the corpus table for {shape} is missing at {}",
        dir.display()
    );
    eprintln!(
        "skipping {shape}: no corpus table at {} — run tests/corpus/generate.sh",
        dir.display()
    );
    None
}

/// One superfile's FTS blob header, as far as a shape assertion cares,
/// plus the column metadata that sits beside it in the Parquet key-value
/// block.
struct BlobHeader {
    version: u32,
    n_docs: u32,
    /// The raw `inf.fts.columns` JSON, so a test can assert what the file
    /// records without going through this engine's deserializer and its
    /// defaults — the point being to see what the *file* says.
    columns_json: String,
}

/// Every superfile's FTS blob header under `root`, in path order.
///
/// Reads the raw bytes rather than going through the reader: the point is
/// to assert what the *file* says, independently of how this engine's
/// reader chooses to interpret it.
fn blob_headers(root: &Path) -> Vec<BlobHeader> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).expect("read corpus dir") {
            let path = entry.expect("dir entry").path();
            match path.is_dir() {
                true => stack.push(path),
                false if path.extension().is_some_and(|e| e == "parquet") => files.push(path),
                false => {}
            }
        }
    }
    files.sort();
    for path in files {
        let bytes = fs::read(&path).expect("read superfile");
        let at = bytes
            .windows(FTS_MAGIC.len())
            .position(|w| w == FTS_MAGIC)
            .unwrap_or_else(|| panic!("no FTS blob in {}", path.display()));
        let field = |off: usize| {
            let start = at + off;
            u32::from_le_bytes(bytes[start..start + 4].try_into().expect("u32 field"))
        };
        let columns_json = find_columns_json(&bytes)
            .unwrap_or_else(|| panic!("no inf.fts.columns in {}", path.display()));
        found.push(BlobHeader {
            version: field(8),
            n_docs: field(16),
            columns_json,
        });
    }
    found
}

/// The `inf.fts.columns` JSON array from a superfile's key-value
/// metadata, located by its leading `[{"name":` rather than by parsing
/// the Parquet footer — enough to assert what the file records.
fn find_columns_json(bytes: &[u8]) -> Option<String> {
    const OPEN: &[u8] = br#"[{"name":"#;
    let start = bytes.windows(OPEN.len()).position(|w| w == OPEN)?;
    let end = bytes[start..].windows(2).position(|w| w == b"}]")?;
    String::from_utf8(bytes[start..start + end + 2].to_vec()).ok()
}

/// Copy a corpus table into a temp dir and open it.
///
/// The checked-in bytes are a fixture: opening a table takes a lock and
/// can write manifest state, so a test that opened them in place would
/// mutate the thing it is asserting about.
pub(crate) fn open_corpus(shape: &str) -> Option<(TempDir, Supertable, std::path::PathBuf)> {
    let src = corpus_dir(shape)?;
    let tmp = TempDir::new().expect("tempdir");
    copy_tree(&src, tmp.path());
    let root = tmp.path().to_path_buf();

    let db = connect(root.to_str().expect("utf-8 path")).expect("connect to corpus");
    let table = db.open_table(TABLE).expect("open corpus table");
    Some((tmp, table, root))
}

fn copy_tree(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).expect("create dst");
    for entry in fs::read_dir(src).expect("read src") {
        let entry = entry.expect("dir entry");
        let to = dst.join(entry.file_name());
        match entry.file_type().expect("file type").is_dir() {
            true => copy_tree(&entry.path(), &to),
            false => {
                fs::copy(entry.path(), &to).expect("copy file");
            }
        }
    }
}

/// Rows a `bm25_search` returns for `query` on `column`.
pub(crate) fn hits(table: &Supertable, column: &str, query: &str) -> usize {
    table
        .bm25_search(column, query, N_DOCS, Bm25SearchOptions::new(), None)
        .expect("bm25 search")
        .iter()
        .map(|b| b.num_rows())
        .sum()
}

/// Asserts one shape: the version every superfile carries, and that the
/// table still opens and ranks.
fn assert_shape(shape: &str, expected_version: u32) {
    let Some(root) = corpus_dir(shape) else {
        return;
    };
    let headers = blob_headers(&root);
    assert!(!headers.is_empty(), "{shape}: no superfiles");
    let total: usize = headers.iter().map(|h| h.n_docs as usize).sum();
    assert_eq!(total, N_DOCS, "{shape}: document count drifted");

    for (i, h) in headers.iter().enumerate() {
        assert_eq!(
            h.version, expected_version,
            "{shape}: superfile {i} carries blob version {}, expected {expected_version}",
            h.version
        );
    }

    // Every shipped release predates the analysis revision, so no corpus
    // file records one. That is what makes the corpus a migration
    // fixture: a reader defaults the missing field to 0, the oldest
    // revision, so every one of these columns is stale by construction
    // and a reindex has something to do. A corpus that recorded a
    // revision would have been written by an engine that already had the
    // tokenization fix, and would prove nothing.
    for (i, h) in headers.iter().enumerate() {
        assert!(
            !h.columns_json.contains("analysis_rev"),
            "{shape}: superfile {i} records an analysis revision, so it was \
             not written by a pre-fix release: {}",
            h.columns_json
        );
    }

    // The version is self-proving for the structures it implies — the
    // builder stamps V4 only when a block chose the bitset encoding, V3
    // only when the positions region is non-empty, V5 only when a coarse
    // table was written. The one thing it cannot prove is that the coarse
    // table spans more than one entry, which needs the postings to exist.
    if expected_version >= 5 {
        for (i, h) in headers.iter().enumerate() {
            assert!(
                h.n_docs >= DOCS_FOR_MULTI_ENTRY_COARSE,
                "{shape}: superfile {i} holds {} documents, too few for a \
                 multi-entry coarse table — the corpus is too small to test \
                 the structure this version exists to add",
                h.n_docs
            );
        }
    }
}

/// The table opens under the current engine and ranks: the shape is not
/// just structurally intact on disk but readable.
fn assert_opens_and_ranks(shape: &str) {
    let Some((_tmp, table, _root)) = open_corpus(shape) else {
        return;
    };
    assert_eq!(
        hits(&table, "body", "common"),
        N_DOCS,
        "{shape}: the corpus-wide term did not match every document"
    );
}

/// The terms every shipped writer left unreachable. A reindex that
/// re-analyzes is what changes these; until then they must stay at zero,
/// or a later reanalysis test proves nothing.
fn assert_tokenization_defect(shape: &str) {
    let Some((_tmp, table, _root)) = open_corpus(shape) else {
        return;
    };

    // An unbroken run was indexed whole, so its leading capped piece —
    // what a query for it now tokenizes to — was never written.
    let capped_piece = "z".repeat(infino_max_token_chars());
    assert_eq!(
        hits(&table, "body", &capped_piece),
        0,
        "{shape}: the over-cap run is already reachable; the corpus was \
         not written by a pre-fix engine"
    );

    // Emoji fell out of the standard analyzer as though they were
    // punctuation, so they are absent from every shipped index.
    assert_eq!(
        hits(&table, "body", "🔥"),
        0,
        "{shape}: the emoji is already indexed"
    );
}

/// Mirrors `MAX_TOKEN_CHARS`, which is crate-internal. Kept as a literal
/// with this note rather than reaching for the internal constant: the
/// corpus is a fixture of what *older* engines wrote, so this number is
/// pinned to the cap in force when the fix landed and must not follow a
/// later change to it.
fn infino_max_token_chars() -> usize {
    255
}

macro_rules! shape_tests {
    ($($name:ident => ($shape:literal, $version:literal)),* $(,)?) => {
        $(
            mod $name {
                use super::*;

                #[test]
                fn carries_its_format_shape() {
                    assert_shape($shape, $version);
                }

                #[test]
                fn opens_and_ranks() {
                    assert_opens_and_ranks($shape);
                }

                #[test]
                fn leaves_the_tokenization_defect_in_place() {
                    assert_tokenization_defect($shape);
                }
            }
        )*
    };
}

shape_tests! {
    v2_positions_region => ("v2_positions_region", 2),
    v4_bitset_blocks => ("v4_bitset_blocks", 4),
    v5_positionless => ("v5_positionless", 5),
    v5_positional => ("v5_positional", 5),
}

/// The oldest shape is a special case, and the reason is not its blob.
///
/// A catalog record has named an analyzer per full-text column only since
/// v0.1.10; a table created before that names none, and `open_table`
/// refuses it rather than guess — guessing would tokenize queries one way
/// against an index built another and return wrong rows instead of an
/// error. So a v1-era *table* cannot be reached through the catalog by
/// this engine at all, independently of anything in its FTS blob.
///
/// What follows for the migration: the reader's v1 support is reachable
/// only by opening a superfile outside the catalog, so for catalog tables
/// it is already unreachable code. The bytes stay checked in as the v1
/// format fixture, and this pins the boundary so that a change making
/// these tables openable is a deliberate one.
mod v1_positionless {
    use super::*;

    #[test]
    fn carries_its_format_shape() {
        assert_shape("v1_positionless", 1);
    }

    #[test]
    fn predates_the_catalog_recording_an_analyzer() {
        let Some(src) = corpus_dir("v1_positionless") else {
            return;
        };
        let tmp = TempDir::new().expect("tempdir");
        copy_tree(&src, tmp.path());

        let db = connect(tmp.path().to_str().expect("utf-8 path")).expect("connect");
        let err = db
            .open_table(TABLE)
            .expect_err("a v1-era catalog record must not open");
        let msg = err.to_string();
        assert!(
            msg.contains("analyzer names recorded"),
            "expected the incomplete-record refusal, got: {msg}"
        );
    }
}

/// Every superfile's FTS blob version under `root`, in path order.
///
/// Read from the bytes rather than through the reader: a migration test
/// asserts what the files became, which is a different question from how
/// this engine chooses to interpret them.
pub(crate) fn blob_versions(root: &Path) -> Vec<u32> {
    blob_headers(root).into_iter().map(|h| h.version).collect()
}
