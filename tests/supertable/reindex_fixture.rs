// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Tables a reindex has work to do on, written by this engine.
//!
//! A table this engine writes is current, so each fixture is written
//! normally and then given exactly one kind of staleness on disk:
//!
//! - [`Staleness::Analysis`]: every FTS column records the analysis
//!   revision one below its chain's — the state every table is in the day
//!   the analyzer moves. Re-analysis is the repair.
//! - [`Staleness::DuplicatedFooter`]: every superfile's footer stores a
//!   stale copy of its vector region key ahead of the real one. A layout
//!   rewrite is the repair, and it is the only staleness the current
//!   writer's container can carry.
//!
//! Both edits leave every blob byte where the writer put it, so the
//! regions a reindex must carry across are this engine's own bytes.

use std::{
    collections::BTreeMap,
    fs,
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow_array::{
    ArrayRef, Decimal128Array, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use infino::{
    Bm25SearchOptions, Connection, FtsField, IndexSpec, Metric, ReindexOptions, Supertable,
    connect,
    superfile::format::{
        footer::{read_kv_metadata, with_forged_footer_kv},
        kv,
    },
    test_helpers::distinct_unit_vectors,
};
use parquet::file::metadata::ParquetMetaDataReader;
use serde_json::Value;
use tempfile::TempDir;

/// Name of the fixture table.
pub(crate) const TABLE: &str = "docs";
/// Documents in the fixture table, written by one append so every
/// superfile declares the same table-wide statistics.
pub(crate) const N_DOCS: u32 = 1_500;
/// Fewest superfiles the append may split into: enough that a run
/// interrupted after a couple of jobs still has jobs left to resume.
const MIN_SUPERFILES: usize = 3;

/// Dimension of the planted embedding: these tables make a vector index
/// present, not measure recall through one.
const EMBEDDING_DIM: usize = 16;
/// Seed for the planted embeddings.
const EMBEDDING_SEED: u64 = 7;
/// Neighbours a vector probe retrieves.
const PROBE_NEIGHBOURS: usize = 16;

/// Footer key fragment that precedes a column's recorded revision.
const REVISION_FIELD: &str = "\"analysis_revision\":";

/// How the fixture is behind what this engine writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Staleness {
    /// Terms recorded one analysis revision behind.
    Analysis,
    /// A stale vector region key stored ahead of the real one.
    DuplicatedFooter,
}

/// The cheapest options that repair `staleness`.
pub(crate) fn repairing(staleness: Staleness) -> ReindexOptions {
    match staleness {
        Staleness::Analysis => ReindexOptions::reanalyzing(),
        Staleness::DuplicatedFooter => ReindexOptions::rewriting(),
    }
}

/// The `body` column: `common` in every document, `shared` in three
/// length groups of four, so a two-term query ties across many documents.
fn body(i: u32) -> String {
    let group = match i % 4 {
        0 => "alpha shared pad",
        1 => "beta shared pad",
        2 => "gamma pad pad",
        _ => "delta shared pad",
    };
    format!("common {group} d{i}")
}

/// The `title` column, indexed with positions.
fn title(i: u32) -> String {
    match i % 3 {
        0 => format!("quick brown fox t{i}"),
        1 => format!("lazy dog jumps t{i}"),
        _ => format!("brown dog runs t{i}"),
    }
}

/// The `notes` column: null for three rows in four, so a column's
/// statistics count the documents carrying tokens rather than the rows.
fn notes(i: u32) -> Option<String> {
    i.is_multiple_of(4)
        .then(|| format!("note common sparse n{i}"))
}

/// Documents `docs`' embeddings, flattened: distinct unit vectors, so a
/// probe's neighbours are never decided by a tie.
fn embeddings(docs: Range<u32>) -> Vec<f32> {
    let all = distinct_unit_vectors(docs.end as usize, EMBEDDING_DIM, EMBEDDING_SEED);
    all[docs.start as usize * EMBEDDING_DIM..].to_vec()
}

/// The vector probe: document 0's own embedding.
pub(crate) fn probe_embedding() -> Vec<f32> {
    embeddings(0..1)
}

fn embedding_field() -> Field {
    Field::new(
        "emb",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            EMBEDDING_DIM as i32,
        ),
        false,
    )
}

/// The rows `docs` covers, as one batch matching `schema`.
fn batch(schema: &Arc<Schema>, docs: Range<u32>) -> RecordBatch {
    let emb = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        EMBEDDING_DIM as i32,
        Arc::new(Float32Array::from(embeddings(docs.clone()))),
        None,
    )
    .expect("embedding column");
    let columns: Vec<ArrayRef> = vec![
        Arc::new(LargeStringArray::from(
            docs.clone().map(body).collect::<Vec<_>>(),
        )),
        Arc::new(LargeStringArray::from(
            docs.clone().map(title).collect::<Vec<_>>(),
        )),
        Arc::new(LargeStringArray::from(docs.map(notes).collect::<Vec<_>>())),
        Arc::new(emb),
    ];
    RecordBatch::try_new(Arc::clone(schema), columns).expect("batch matches the schema")
}

/// Append the fixture's documents `docs` to `table`.
pub(crate) fn append_docs(table: &Supertable, docs: Range<u32>) {
    table
        .append(&batch(&table.schema(), docs))
        .expect("append one superfile");
}

/// Write the fixture table under `root`, make it stale as `staleness`
/// says, and return how many superfiles it holds.
pub(crate) fn write_stale_table(root: &Path, staleness: Staleness) -> usize {
    write_table(root);
    edit_superfiles(root, |bytes| match staleness {
        Staleness::Analysis => lower_recorded_revisions(bytes),
        Staleness::DuplicatedFooter => duplicate_vector_region_key(bytes),
    })
}

/// Write the fixture table, current, under `root`.
pub(crate) fn write_table(root: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("body", DataType::LargeUtf8, false),
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("notes", DataType::LargeUtf8, true),
        embedding_field(),
    ]));
    let spec = IndexSpec::new()
        .fts(FtsField::new("body"))
        .fts(FtsField::new("title").positions(true))
        .fts(FtsField::new("notes"))
        .vector("emb", EMBEDDING_DIM, Metric::Cosine);
    let db = connect(root.to_str().expect("utf-8 path")).expect("connect");
    let table = db
        .create_table(TABLE, Arc::clone(&schema), spec)
        .expect("create the fixture table");
    // One append per superfile, so the count never depends on how a
    // single append splits on this machine.
    let per_append = N_DOCS / MIN_SUPERFILES as u32;
    for i in 0..MIN_SUPERFILES as u32 {
        let end = if i + 1 == MIN_SUPERFILES as u32 {
            N_DOCS
        } else {
            (i + 1) * per_append
        };
        append_docs(&table, i * per_append..end);
    }
}

/// Rewrite every superfile of the fixture table under `root` through
/// `edit`, and return how many there are.
pub(crate) fn edit_superfiles(root: &Path, edit: impl Fn(&[u8]) -> Vec<u8>) -> usize {
    let paths = superfile_paths(root);
    assert!(
        paths.len() >= MIN_SUPERFILES,
        "the fixture wrote {} superfiles, too few to interrupt a run between jobs",
        paths.len()
    );
    for path in &paths {
        let bytes = fs::read(path).expect("read superfile");
        fs::write(path, edit(&bytes)).expect("write the edited superfile");
    }
    paths.len()
}

/// A fixture table in a temp dir, with a connection and its handle.
pub(crate) struct StaleTable {
    pub(crate) dir: TempDir,
    pub(crate) db: Connection,
    pub(crate) table: Supertable,
    /// Superfiles the table holds, every one of them stale.
    pub(crate) superfiles: usize,
}

impl StaleTable {
    pub(crate) fn root(&self) -> &Path {
        self.dir.path()
    }
}

/// Write a fixture table into a fresh temp dir and open it.
pub(crate) fn open_stale(staleness: Staleness) -> StaleTable {
    let dir = TempDir::new().expect("tempdir");
    let superfiles = write_stale_table(dir.path(), staleness);
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("connect");
    let table = db.open_table(TABLE).expect("open the fixture table");
    StaleTable {
        dir,
        db,
        table,
        superfiles,
    }
}

/// `bytes` with every FTS column's recorded analysis revision lowered by
/// one, edited in place in the footer.
///
/// Same-length by construction, so every blob offset the footer and the
/// manifest record stays true.
fn lower_recorded_revisions(bytes: &[u8]) -> Vec<u8> {
    let recorded = recorded_revisions(bytes);
    assert!(!recorded.is_empty(), "the superfile records no FTS columns");
    let mut out = bytes.to_vec();
    for revision in recorded {
        let current = format!("{REVISION_FIELD}{revision}");
        let lowered = format!(
            "{REVISION_FIELD}{}",
            revision
                .checked_sub(1)
                .expect("a recorded revision above the oldest")
        );
        assert_eq!(
            current.len(),
            lowered.len(),
            "lowering revision {revision} would move every offset after it"
        );
        let at = find(&out, current.as_bytes()).expect("the recorded revision is in the footer");
        out[at..at + lowered.len()].copy_from_slice(lowered.as_bytes());
    }
    out
}

/// `bytes` with a stale `inf.vec.offset` stored ahead of the real one —
/// the footer a reindex that carried its input's region keys wrote.
fn duplicate_vector_region_key(bytes: &[u8]) -> Vec<u8> {
    let real: u64 = read_kv_metadata(bytes)
        .expect("read superfile key-value metadata")
        .get(kv::VEC_OFFSET)
        .expect("every fixture superfile carries a vector region")
        .parse()
        .expect("offset is a number");
    let stale = (real + 1).to_string();
    with_forged_footer_kv(bytes, &[(kv::VEC_OFFSET, &stale)]).to_vec()
}

/// First position of `needle` in `haystack`.
fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// The analysis revision each FTS column in a superfile records, in
/// column order.
pub(crate) fn recorded_revisions(bytes: &[u8]) -> Vec<u32> {
    let columns = read_kv_metadata(bytes)
        .expect("read superfile key-value metadata")
        .get(kv::FTS_COLUMNS)
        .cloned()
        .expect("an FTS superfile records its columns");
    let columns: Vec<Value> = serde_json::from_str(&columns).expect("FTS columns are JSON");
    columns
        .iter()
        .map(|c| {
            c["analysis_revision"]
                .as_u64()
                .and_then(|r| u32::try_from(r).ok())
                .expect("every column records its analysis revision")
        })
        .collect()
}

/// Every file under `root` with `ext`, in path order.
pub(crate) fn files_with_extension(root: &Path, ext: &str) -> Vec<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).expect("read dir") {
            let path = entry.expect("dir entry").path();
            match path.is_dir() {
                true => stack.push(path),
                false if path.extension().is_some_and(|e| e == ext) => files.push(path),
                false => {}
            }
        }
    }
    files.sort();
    files
}

/// The fixture table's own directory under `root` (not the catalog's).
pub(crate) fn table_dir(root: &Path) -> PathBuf {
    fs::read_dir(root)
        .expect("read root")
        .map(|e| e.expect("dir entry").path())
        .find(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(&format!("{TABLE}-")))
        })
        .expect("fixture table directory")
}

/// Every superfile of the fixture table under `root`, in path order.
pub(crate) fn superfile_paths(root: &Path) -> Vec<PathBuf> {
    files_with_extension(&table_dir(root).join("data"), "parquet")
}

/// One footer key read from every superfile under `root`, in path order.
pub(crate) fn footer_values(root: &Path, key: &str) -> Vec<Option<String>> {
    superfile_paths(root)
        .iter()
        .map(|path| {
            let bytes = fs::read(path).expect("read superfile");
            read_kv_metadata(&bytes)
                .expect("read superfile key-value metadata")
                .get(key)
                .cloned()
        })
        .collect()
}

/// The lowest analysis revision each superfile under `root` records, in
/// path order.
pub(crate) fn file_revisions(root: &Path) -> Vec<u32> {
    superfile_paths(root)
        .iter()
        .map(|path| {
            let bytes = fs::read(path).expect("read superfile");
            recorded_revisions(&bytes)
                .into_iter()
                .min()
                .expect("an FTS superfile records its columns")
        })
        .collect()
}

/// Every key-value pair in a superfile's Parquet footer, in stored order
/// and duplicates kept.
///
/// Parsed with parquet-rs rather than this engine's reader: the engine
/// folds the list into a map, so it cannot see a key stored twice.
pub(crate) fn raw_footer_kvs(bytes: &Bytes) -> Vec<(String, String)> {
    let metadata = ParquetMetaDataReader::new()
        .parse_and_finish(bytes)
        .expect("parse superfile footer");
    metadata
        .file_metadata()
        .key_value_metadata()
        .map(|kvs| {
            kvs.iter()
                .filter_map(|e| Some((e.key.clone(), e.value.clone()?)))
                .collect()
        })
        .unwrap_or_default()
}

/// A blob's `(offset, length)` as a reader that takes the first stored
/// copy of each key resolves it.
pub(crate) fn first_region(
    kvs: &[(String, String)],
    offset: &str,
    length: &str,
) -> Option<(u64, u64)> {
    let first = |key: &str| {
        kvs.iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.parse::<u64>().expect("footer offset is a u64"))
    };
    Some((first(offset)?, first(length)?))
}

/// `(id, score)` pairs from a result batch set, in rank order.
fn ids_and_scores(batches: &[RecordBatch]) -> Vec<(i128, f32)> {
    let mut out = Vec::new();
    for batch in batches {
        let ids = batch
            .column_by_name("_id")
            .expect("_id column")
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("_id is Decimal128");
        let scores = batch
            .column_by_name("score")
            .expect("score column")
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("score is f32");
        for i in 0..batch.num_rows() {
            out.push((ids.value(i), scores.value(i)));
        }
    }
    out
}

/// Ids and distances a vector search returns, in rank order.
pub(crate) fn vector_hits(table: &Supertable, probe: &[f32]) -> Vec<(i128, f32)> {
    let batches = table
        .vector_search("emb", probe, PROBE_NEIGHBOURS, None, None)
        .expect("vector search");
    ids_and_scores(&batches)
}

/// Every matching document's score, keyed by id.
///
/// Compared by id rather than rank: `common shared` ties across most of
/// the table, and which tied document comes first falls out of superfile
/// layout — exactly what a reindex rewrites.
pub(crate) fn scores_by_id(
    table: &Supertable,
    column: &str,
    query: &str,
    k: usize,
) -> BTreeMap<i128, f32> {
    let batches = table
        .bm25_search(column, query, k, Bm25SearchOptions::new(), None)
        .expect("bm25 search");
    ids_and_scores(&batches).into_iter().collect()
}

/// Rows a `bm25_search` returns for `query` on `column`, taking `k`.
pub(crate) fn hits_k(table: &Supertable, column: &str, query: &str, k: usize) -> usize {
    table
        .bm25_search(column, query, k, Bm25SearchOptions::new(), None)
        .expect("bm25 search")
        .iter()
        .map(|b| b.num_rows())
        .sum()
}

/// Rows a `bm25_search` returns for `query` on `column`, over the whole
/// fixture.
pub(crate) fn hits(table: &Supertable, column: &str, query: &str) -> usize {
    hits_k(table, column, query, N_DOCS as usize)
}

/// Largest relative score change a reindex may produce.
///
/// Not slack — the bound on one intended effect. Each superfile scores at
/// the table's average document length as of its own commit, and a repair
/// bakes the average as of the repair. Every fixture document in a column
/// has the same length, so those averages agree and only float rounding
/// is left.
const MAX_SCORE_DRIFT: f32 = 1e-4;

/// The same documents match, with scores no further apart than
/// [`MAX_SCORE_DRIFT`]. The first assertion is the load-bearing one: a
/// repair that dropped or gained a document is broken however it scores.
pub(crate) fn assert_scores_equivalent(
    after: &BTreeMap<i128, f32>,
    before: &BTreeMap<i128, f32>,
    what: &str,
) {
    assert_eq!(
        after.keys().collect::<Vec<_>>(),
        before.keys().collect::<Vec<_>>(),
        "{what}: the set of matching documents changed"
    );
    for (id, &now) in after {
        let was = before[id];
        let drift = (now - was).abs() / was.abs().max(f32::MIN_POSITIVE);
        assert!(
            drift <= MAX_SCORE_DRIFT,
            "{what}: document {id} scored {was} and now scores {now} \
             (relative change {drift:.3e})"
        );
    }
}
