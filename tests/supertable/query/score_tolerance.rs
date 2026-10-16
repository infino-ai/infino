// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A caller who accepts a shortfall against the k-th score reads fewer
//! superfiles.
//!
//! The engine opens superfiles in descending score-ceiling order and stops
//! once the best unread ceiling cannot beat the running k-th score. A
//! tolerance raises the score a ceiling has to clear, so the fan-out stops
//! sooner. These tests pin both halves of the contract: that the default
//! still reads whatever exactness requires, and that a tolerance actually
//! reduces the reads rather than only being accepted.

use std::sync::Arc;

use arrow_array::{ArrayRef, LargeStringArray, RecordBatch};
use infino::{
    Bm25SearchOptions,
    runtime_metrics::op_stats::with_op_stats,
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::builder::FtsConfig,
    supertable::{Supertable, SupertableOptions},
};
use tempfile::TempDir;

/// Superfiles in the table. Enough that stopping early is visible as a
/// difference in how many were opened.
const N_SUPERFILES: usize = 8;
/// Documents per superfile.
const DOCS_PER_SUPERFILE: usize = 16;
/// Hits asked for. One, so a single superfile can satisfy the query and the
/// rest are decided purely by whether their ceiling can compete.
const TOP_K: usize = 1;
/// A tolerance no ceiling in this table can clear once a floor exists, which
/// is what makes the stop observable rather than marginal.
const WIDE_TOLERANCE: f32 = 10.0;

fn schema() -> Arc<arrow_schema::Schema> {
    Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "title",
        arrow_schema::DataType::LargeUtf8,
        false,
    )]))
}

/// One superfile's titles. `seg` scales how often the term repeats, so the
/// superfiles carry visibly different score ceilings and the descending open
/// order is not arbitrary.
fn titles_for(seg: usize) -> Vec<String> {
    (0..DOCS_PER_SUPERFILE)
        .map(|d| {
            let repeats = 1 + (seg * DOCS_PER_SUPERFILE + d) % 7;
            let mut words: Vec<&str> = vec!["alpha"; repeats];
            words.push("filler");
            words.join(" ")
        })
        .collect()
}

/// A storage-backed table of [`N_SUPERFILES`] superfiles. Storage is what
/// gives the table a term index, and the index is what supplies the ceilings
/// the skip decision compares; a storage-less table has none and fans out.
///
/// The open window is narrowed to one. At the shipped default of 64 every
/// superfile in a table this size opens in the first wave, before any k-th
/// score exists to compare a ceiling against, so the skip decision never runs
/// and a tolerance cannot show up. That is correct behaviour and not worth a
/// thousand-superfile fixture to escape, so the window is set to the value
/// that makes one open depend on the last.
fn table(dir: &TempDir) -> Supertable {
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let options = SupertableOptions::new(schema(), vec![FtsConfig::new("title")], vec![])
        .expect("options")
        .with_storage(storage)
        .with_bound_ordered_open_window(1);
    let st = Supertable::create(options).expect("create");
    let mut w = st.writer().expect("writer");
    for seg in 0..N_SUPERFILES {
        let owned = titles_for(seg);
        let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
        let col: ArrayRef = Arc::new(LargeStringArray::from(refs));
        let batch = RecordBatch::try_new(schema(), vec![col]).expect("batch");
        w.append(&batch).expect("append");
        w.commit().expect("commit");
    }
    drop(w);
    st
}

/// The hits a search returns and the superfiles it opened reaching them.
///
/// The reader is minted inside the metered scope: a reader carries the
/// collector it was created under, so one minted outside reports zero opens
/// for any query at all.
fn hits_and_opens(st: &Supertable, tolerance: f32) -> (usize, u64) {
    let (rows, stats) = with_op_stats(|| {
        st.bm25_search(
            "title",
            "alpha",
            TOP_K,
            Bm25SearchOptions::new().with_score_tolerance(tolerance),
            None,
        )
        .expect("search")
    });
    let hits = rows.iter().map(RecordBatch::num_rows).sum();
    (hits, stats.score_pruning_survived)
}

#[test]
fn a_tolerance_reads_fewer_superfiles_than_exact_scoring() {
    let dir = TempDir::new().expect("tempdir");
    let st = table(&dir);

    let (exact_hits, exact_opens) = hits_and_opens(&st, 0.0);
    let (loose_hits, loose_opens) = hits_and_opens(&st, WIDE_TOLERANCE);

    assert_eq!(exact_hits, TOP_K, "exact scoring answers the query");
    assert_eq!(
        loose_hits, TOP_K,
        "so does a tolerant one; the shortfall is in which document, not how many"
    );
    assert!(
        loose_opens < exact_opens,
        "a tolerance must actually stop the fan-out early: exact opened \
         {exact_opens} superfiles, tolerant opened {loose_opens}"
    );
}

/// The default has to be exactly what the engine did before a tolerance
/// existed. Asserted against a second exact run rather than a constant: the
/// number of superfiles exactness requires depends on the corpus, and pinning
/// a literal here would make this test about this fixture instead of about
/// the default.
#[test]
fn the_default_is_exact() {
    let dir = TempDir::new().expect("tempdir");
    let st = table(&dir);

    let (default_hits, default_opens) = {
        let (rows, stats) = with_op_stats(|| {
            st.bm25_search("title", "alpha", TOP_K, Bm25SearchOptions::new(), None)
                .expect("search")
        });
        (
            rows.iter().map(RecordBatch::num_rows).sum::<usize>(),
            stats.score_pruning_survived,
        )
    };
    let (zero_hits, zero_opens) = hits_and_opens(&st, 0.0);

    assert_eq!(default_hits, zero_hits);
    assert_eq!(
        default_opens, zero_opens,
        "the default option set must open what a zero tolerance opens"
    );
}
