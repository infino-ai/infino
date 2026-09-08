// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! `Supertable::inspect` over durable storage: the manifest facts
//! come from the persisted list, every committed superfile is reported, and
//! the report costs only footer and header reads — no dictionary, posting
//! list, or row group is fetched.
//!
//! [`renders_a_mixed_format_table`] additionally pins what the report *says*,
//! as rendered text, on a table holding one current and one older superfile.
//! An assertion on the rendered form rather than on separate fields is
//! deliberate: what an operator reads is the whole picture, so the whole
//! picture is what a change has to update on purpose.

use std::sync::Arc;

use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::{
    Inspection, OptionsHashRule,
    superfile::{
        builder::FtsConfig,
        format::fts::{MAGIC as FTS_MAGIC, VERSION_V4, hdr},
        vector::rerank_codec::RerankCodec,
    },
    supertable::{
        Supertable, SupertableOptions,
        storage::{LocalFsStorageProvider, StorageProvider},
    },
    test_helpers::{build_title_batch, default_supertable_options, default_vector_config},
};
use tempfile::TempDir;

/// Commits, one superfile each.
const N_COMMITS: u64 = 4;

/// Object reads the report may issue per superfile: one footer read (the
/// size is known from the manifest, so the tail is a single ranged GET) and
/// one header probe for the full-text section.
const GETS_PER_SUPERFILE: u64 = 2;

/// Object reads the report may issue outside the superfiles: the manifest
/// pointer refresh and the list and part loads behind it.
const GETS_FOR_MANIFEST: u64 = 4;

/// Speculative footer tail the reader fetches per superfile.
const FOOTER_TAIL_BYTES: u64 = 64 * 1024;

/// Bytes a header probe reads: eight-byte magic plus the version word.
const HEADER_PROBE_BYTES: u64 = 12;

/// Slack for the manifest pointer, list, and parts, which are small.
const MANIFEST_BYTES_SLACK: u64 = 256 * 1024;

fn commit_titles(st: &Supertable, titles: &[&str]) {
    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(titles)).expect("append");
    w.commit().expect("commit");
}

#[test]
fn durable_table_reports_manifest_and_superfiles_within_a_header_read_budget() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");
    for i in 0..N_COMMITS {
        let title = format!("document number {i}");
        commit_titles(&st, &[title.as_str(), "filler alpha", "filler bravo"]);
    }

    let before = storage.usage_meter().snapshot();
    let report = st.inspect().expect("inspect");
    let after = storage.usage_meter().snapshot();

    let manifest = report
        .manifest
        .as_ref()
        .expect("durable table has a persisted list");
    assert_eq!(manifest.options_hash_rule, OptionsHashRule::Current);
    assert!(manifest.current, "{manifest:?}");

    assert_eq!(report.superfiles.len() as u64, N_COMMITS);
    assert!(report.is_current(), "{report:?}");
    assert!(report.superfiles.iter().all(|r| r.size_bytes > 0));
    assert!(report.superfiles.iter().all(|r| r.fts_version.is_some()));

    // The report never touches a dictionary, a posting list, or a row group:
    // its reads fit a footer-and-header budget per superfile.
    let gets = after.get_count - before.get_count;
    let bytes = after.get_bytes - before.get_bytes;
    assert!(
        gets <= N_COMMITS * GETS_PER_SUPERFILE + GETS_FOR_MANIFEST,
        "{gets} object reads for {N_COMMITS} superfiles"
    );
    assert!(
        bytes <= N_COMMITS * (FOOTER_TAIL_BYTES + HEADER_PROBE_BYTES) + MANIFEST_BYTES_SLACK,
        "{bytes} bytes read for {N_COMMITS} superfiles"
    );

    // A second call is the same picture.
    assert_eq!(st.inspect().expect("again"), report);
}

/// Embedding width for the mixed-format fixture; the engine's minimum.
const EMB_DIM: usize = 16;

/// Random-rotation seed for the fixture's vector index.
const ROT_SEED: u64 = 11;

/// Width of a section header's version word.
const VERSION_WORD_BYTES: usize = 4;

/// Rows per commit in the mixed-format fixture.
///
/// One, deliberately: a vector column partitions by cell, so a commit is
/// split across as many superfiles as its rows have distinct nearest
/// centroids. A single row can only land in one cell, which fixes the
/// superfile count at one per commit without the fixture depending on how
/// clustering happens to fall.
const ROWS_PER_COMMIT: usize = 1;

/// The report on a freshly built table: two commits, two superfiles, every
/// layer at what the builder writes today.
const ALL_CURRENT: &str = "\
table: current
manifest: format 1.0, options-hash rule current, current
superfile 1: container 1.1.0, fts V5, vector v2, id sidecar, current
superfile 2: container 1.1.0, fts V5, vector v2, id sidecar, current
containers: 1.1.0 x 2
fts sections: V5 x 2
vector sections: v2 x 2
stale superfiles: 0 of 2";

/// The same table after one superfile's full-text section is stamped back to
/// the previous header version: that file alone reads stale, and the roll-up
/// splits across two versions.
const ONE_STALE: &str = "\
table: stale
manifest: format 1.0, options-hash rule current, current
superfile 1: container 1.1.0, fts V5, vector v2, id sidecar, current
superfile 2: container 1.1.0, fts V4, vector v2, id sidecar, stale
containers: 1.1.0 x 2
fts sections: V4 x 1, V5 x 1
vector sections: v2 x 2
stale superfiles: 1 of 2";

/// A table with one full-text column and one vector column, on durable
/// storage, configured the way the engine configures one itself: a vector
/// column partitions by cell, and its superfiles carry the multi-cell vector
/// section header.
fn fts_and_vector_options() -> SupertableOptions {
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new(
            "emb",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                EMB_DIM as i32,
            ),
            false,
        ),
    ]));
    let mut vector_config = default_vector_config("emb", ROT_SEED);
    // The engine-default codec, i.e. what a catalog-created table gets.
    vector_config.rerank_codec = RerankCodec::default();
    SupertableOptions::new(schema, vec![FtsConfig::new("title")], vec![vector_config])
        .expect("valid options")
}

/// One commit's rows: distinct titles and one-hot embeddings.
fn titled_vector_batch(schema: Arc<Schema>, round: usize) -> RecordBatch {
    let titles: Vec<String> = (0..ROWS_PER_COMMIT)
        .map(|i| format!("document {round} {i}"))
        .collect();
    let mut flat = Vec::<f32>::with_capacity(ROWS_PER_COMMIT * EMB_DIM);
    for i in 0..ROWS_PER_COMMIT {
        // A distinct one-hot direction per row and per commit.
        let hot = (round + i) % EMB_DIM;
        for d in 0..EMB_DIM {
            flat.push(if d == hot { 1.0 } else { 0.0 });
        }
    }
    let embeddings = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        EMB_DIM as i32,
        Arc::new(Float32Array::from(flat)),
        None,
    )
    .expect("fixed-size list");
    let columns: Vec<ArrayRef> = vec![
        Arc::new(LargeStringArray::from(
            titles.iter().map(String::as_str).collect::<Vec<_>>(),
        )),
        Arc::new(embeddings),
    ];
    RecordBatch::try_new(schema, columns).expect("batch")
}

/// The inspection as an operator reads it: the table's verdict, the manifest,
/// one line per superfile, then the per-layer roll-up.
///
/// Superfiles are numbered rather than named by id, and byte sizes are left
/// out, so the rendering is stable across runs; both are asserted separately.
fn render(inspection: &Inspection) -> String {
    let mut lines = vec![format!(
        "table: {}",
        if inspection.is_current() {
            "current"
        } else {
            "stale"
        }
    )];
    match &inspection.manifest {
        Some(manifest) => lines.push(format!(
            "manifest: format {}, options-hash rule {}, {}",
            manifest.format_version,
            manifest.options_hash_rule.as_str(),
            if manifest.current { "current" } else { "stale" }
        )),
        None => lines.push("manifest: none (in-process table)".to_string()),
    }
    for (i, row) in inspection.superfiles.iter().enumerate() {
        let fts = row
            .fts_version
            .map_or_else(|| "none".to_string(), |v| format!("V{v}"));
        let vector = row
            .vector_version
            .map_or_else(|| "none".to_string(), |v| format!("v{v}"));
        lines.push(format!(
            "superfile {}: container {}, fts {fts}, vector {vector}, {}, {}{}",
            i + 1,
            row.container_version,
            if row.id_sidecar {
                "id sidecar"
            } else {
                "no id sidecar"
            },
            if row.current { "current" } else { "stale" },
            if row.vector_index {
                " (vector index)"
            } else {
                ""
            },
        ));
    }
    lines.push(format!(
        "containers: {}",
        histogram(inspection.container_versions().into_iter())
    ));
    lines.push(format!(
        "fts sections: {}",
        histogram(
            inspection
                .fts_versions()
                .into_iter()
                .map(|(v, n)| (format!("V{v}"), n))
        )
    ));
    lines.push(format!(
        "vector sections: {}",
        histogram(
            inspection
                .vector_versions()
                .into_iter()
                .map(|(v, n)| (format!("v{v}"), n))
        )
    ));
    lines.push(format!(
        "stale superfiles: {} of {}",
        inspection.stale_superfiles(),
        inspection.superfiles.len()
    ));
    lines.join("\n")
}

/// `label x count` pairs, or `-` when there is nothing to count.
fn histogram(counts: impl Iterator<Item = (String, usize)>) -> String {
    let rendered: Vec<String> = counts.map(|(k, n)| format!("{k} x {n}")).collect();
    if rendered.is_empty() {
        "-".to_string()
    } else {
        rendered.join(", ")
    }
}

/// Stamp one superfile's full-text section header back to the previous
/// version, in place, so the table holds a genuine mix.
///
/// Only the header's version word is rewritten: the report reads that word and
/// nothing else in the section, so the rest of the blob need not be a valid
/// older layout. The section's own magic locates the header, and the file is
/// found by the id the report gave.
fn stamp_fts_header_back(data_dir: &std::path::Path, superfile_id: &str) {
    let path = data_dir.join(format!("seg-{superfile_id}.sf.parquet"));
    let mut bytes = std::fs::read(&path).expect("read superfile");
    let magic_at = bytes
        .windows(FTS_MAGIC.len())
        .position(|window| window == FTS_MAGIC)
        .expect("the superfile carries a full-text section");
    let word = magic_at + hdr::VERSION_OFF;
    let end = word + VERSION_WORD_BYTES;
    assert!(
        end <= bytes.len(),
        "the version word sits inside the file: {end} > {}",
        bytes.len()
    );
    bytes[word..end].copy_from_slice(&VERSION_V4.to_le_bytes());
    std::fs::write(&path, &bytes).expect("write superfile");
}

#[test]
fn renders_a_mixed_format_table() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(fts_and_vector_options().with_storage(Arc::clone(&storage)))
        .expect("create");
    let schema = st.options().schema.clone();
    for round in 0..2 {
        let mut w = st.writer().expect("writer");
        w.append(&titled_vector_batch(Arc::clone(&schema), round))
            .expect("append");
        w.commit().expect("commit");
    }

    let inspection = st.inspect().expect("inspect");
    assert_eq!(render(&inspection), ALL_CURRENT);
    // Left out of the rendering because they are not stable run to run, so
    // asserted here: every row names a real file with real bytes behind it.
    assert!(
        inspection
            .superfiles
            .iter()
            .all(|row| row.size_bytes > 0 && !row.superfile_id.is_empty()),
        "{inspection:?}"
    );

    // Stamp the second superfile's full-text header back a version. The report
    // is not cached, so the next call reads the changed bytes.
    // Also the proof that the reported id names the object on disk: the patch
    // below locates the file by it, and fails loudly if it does not exist.
    let stale_id = inspection.superfiles[1].superfile_id.clone();
    stamp_fts_header_back(&dir.path().join("data"), &stale_id);

    let after = st.inspect().expect("inspect after the downgrade");
    assert_eq!(render(&after), ONE_STALE);
    assert_eq!(after.superfiles[1].superfile_id, stale_id);
}
