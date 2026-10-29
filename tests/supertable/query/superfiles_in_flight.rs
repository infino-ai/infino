// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A SQL query works on at most its cap of files at once, and the cap changes
//! neither what a scan returns nor what it reads.

#![deny(clippy::unwrap_used)]

use std::{path::Path, sync::Arc, time::Duration};

use arrow::{
    array::{Int64Array, RecordBatch},
    compute::concat_batches,
};
use datafusion::prelude::{col, lit};
use infino::{
    supertable::{
        Supertable, SupertableReader,
        storage::{LocalFsStorageProvider, StorageProvider},
    },
    test_helpers::{
        build_title_batch, default_supertable_options, lazy_foreground_disk_cache,
        slow_storage::SlowStorage,
    },
};
use rayon::ThreadPoolBuilder;
use tempfile::TempDir;

/// Files in each table, one per commit; more than the smaller caps below.
const FILES: usize = 12;
/// Rows per file: enough that a cold scan reads each file's data, and its
/// dictionary, from storage rather than from the bytes its open already read.
const ROWS_PER_FILE: usize = 8_000;
/// Hex words in each title.
const WORDS_PER_TITLE: usize = 6;
/// Multiplier of the generator behind the titles (Knuth's MMIX constants).
const LCG_MULT: u64 = 6_364_136_223_846_793_005;
/// Increment of the generator behind the titles.
const LCG_INC: u64 = 1_442_695_040_888_963_407;
/// Caps on the files a scan works on at once, each compared against the
/// default: one at a time, and one that does not divide the file count.
const CAPS: [usize; 2] = [1, 7];
/// A cap under [`FILES`], for the tests that count files read at once.
const SMALL_CAP: usize = 3;
/// How long each read takes, so reads in flight overlap.
const READ_LATENCY: Duration = Duration::from_millis(20);
/// The key suffix of a superfile's object, to tell its reads from the
/// manifest's.
const SUPERFILE_SUFFIX: &str = ".sf.parquet";
/// The key suffix of a superfile's delete sidecar.
const SIDECAR_SUFFIX: &str = ".tombstones";
/// The word every even row holds. No hex word contains it or [`ODD_WORD`],
/// and no row holds both.
const EVEN_WORD: &str = "even";
/// The word every odd row holds.
const ODD_WORD: &str = "odd";

/// Titles that do not compress, so a file's bytes grow with its rows: each
/// is a few hex words drawn from a generator seeded by file and row, then
/// [`EVEN_WORD`] or [`ODD_WORD`] by the row's parity.
fn titles(file: usize, rows: usize) -> Vec<String> {
    (0..rows)
        .map(|row| {
            let mut state = (file as u64) << 32 | row as u64;
            let mut words: Vec<String> = (0..WORDS_PER_TITLE)
                .map(|_| {
                    state = state.wrapping_mul(LCG_MULT).wrapping_add(LCG_INC);
                    format!("{state:016x}")
                })
                .collect();
            let parity = if row % 2 == 0 { EVEN_WORD } else { ODD_WORD };
            words.push(parity.to_string());
            words.join(" ")
        })
        .collect()
}

/// A table of [`FILES`] files, one per commit of `titles_of(file)`, on
/// storage under `dir`.
fn table_of(
    dir: &Path,
    titles_of: impl Fn(usize) -> Vec<String>,
) -> (Supertable, Arc<dyn StorageProvider>) {
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir).expect("provider"));
    let table = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");
    let mut writer = table.writer().expect("writer");
    for file in 0..FILES {
        let titles = titles_of(file);
        let refs: Vec<&str> = titles.iter().map(String::as_str).collect();
        writer.append(&build_title_batch(&refs)).expect("append");
        writer.commit().expect("commit");
    }
    (table, storage)
}

/// [`table_of`] with [`ROWS_PER_FILE`] of [`titles`] in each file.
fn titles_table(dir: &Path) -> Arc<dyn StorageProvider> {
    table_of(dir, |file| titles(file, ROWS_PER_FILE)).1
}

/// One query's rows as a single batch, so results split into batches
/// differently still compare equal.
fn rows(batches: &[RecordBatch]) -> RecordBatch {
    let schema = batches.first().expect("a query returns a batch").schema();
    concat_batches(&schema, batches).expect("concat")
}

/// A reader opened cold, the slowed storage under it, and the directory of
/// its disk cache, which lives as long as this does.
struct ColdReader {
    storage: Arc<SlowStorage>,
    reader: SupertableReader,
    _cache_dir: TempDir,
}

/// A reader opened cold on `storage`, through slowed storage and a lazily
/// filled disk cache, as production reads. It works on at most `cap` files at
/// once, or the default when `None`.
///
/// Its reader pool has a thread per file. A scan checks exact predicates on
/// at most one file per pool thread, so a narrower pool could hold the scan
/// to fewer files than the cap and hide whether the cap is what bounds it.
fn cold_reader(storage: &Arc<dyn StorageProvider>, cap: Option<usize>) -> ColdReader {
    let slow = SlowStorage::wrap(Arc::clone(storage), READ_LATENCY);
    let slowed: Arc<dyn StorageProvider> = slow.clone();
    let cache_dir = TempDir::new().expect("cache tempdir");
    let cache = lazy_foreground_disk_cache(Arc::clone(&slowed), cache_dir.path());
    let pool = ThreadPoolBuilder::new()
        .num_threads(FILES)
        .build()
        .expect("reader pool");
    let mut options = default_supertable_options()
        .with_storage(slowed)
        .with_disk_cache(cache)
        .with_reader_pool(Arc::new(pool));
    if let Some(cap) = cap {
        options = options.with_superfiles_in_flight(cap);
    }
    let table = Supertable::open(options).expect("open");
    let reader = table.reader().expect("reader");
    assert_eq!(reader.n_superfiles(), FILES);
    ColdReader {
        storage: slow,
        reader,
        _cache_dir: cache_dir,
    }
}

/// Every query's rows and the reads each made, in order, from one cold
/// reader on `storage` that works on at most `cap` files at once.
fn cold_scan(
    storage: &Arc<dyn StorageProvider>,
    cap: Option<usize>,
    queries: &[&str],
) -> (Vec<RecordBatch>, Vec<usize>) {
    let ColdReader {
        storage, reader, ..
    } = cold_reader(storage, cap);
    let mut results = Vec::with_capacity(queries.len());
    let mut reads = Vec::with_capacity(queries.len());
    for sql in queries {
        let before = storage.reads();
        results.push(rows(&reader.query_sql(sql).expect("query")));
        reads.push(storage.reads() - before);
    }
    (results, reads)
}

#[test]
fn a_cold_scan_returns_and_reads_the_same_at_any_cap() {
    let storage_dir = TempDir::new().expect("storage tempdir");
    let storage = titles_table(storage_dir.path());

    // A word from one file only, so the dictionary decides which rows match
    // and the scan resolves candidate rows per file before reading any. It
    // runs first, on a cold reader, so its index reads are all counted.
    let word = titles(FILES / 2, ROWS_PER_FILE)[ROWS_PER_FILE / 2]
        .split(' ')
        .next()
        .expect("a word")
        .to_string();
    let word_query = format!("SELECT title FROM supertable WHERE title ILIKE '%{word}%'");
    let queries = [
        word_query.as_str(),
        "SELECT SUM(length(title)) AS n FROM supertable",
        "SELECT COUNT(*) AS n FROM supertable WHERE substr(title, 1, 1) = 'a'",
        "SELECT title FROM supertable WHERE substr(title, 2, 1) = 'b' ORDER BY title LIMIT 5",
    ];

    let (default_results, default_reads) = cold_scan(&storage, None, &queries);
    assert!(
        default_reads[0] > 0,
        "the first query reads from storage, so the read counts below compare something"
    );
    assert_eq!(
        default_results[0].num_rows(),
        1,
        "the word matches the one row it was taken from"
    );
    for cap in CAPS {
        let (results, reads) = cold_scan(&storage, Some(cap), &queries);
        assert_eq!(results, default_results, "results at cap {cap}");
        assert_eq!(
            reads, default_reads,
            "each query's reads at cap {cap}: the cap changes when files are read, \
             not what is read"
        );
    }
}

/// The single count `sql` returns.
fn count(reader: &SupertableReader, sql: &str) -> i64 {
    let out = reader.query_sql(sql).expect("query");
    out[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("count column")
        .value(0)
}

/// The most superfiles a cold count of rows holding both [`EVEN_WORD`] and
/// [`ODD_WORD`] read at once, from a reader that works on at most `cap` files
/// at once, and the count. Every file holds both words, so none is pruned and
/// each one's dictionary is read to find its candidate rows; no row holds
/// both, so there are none and no data is read after.
fn peak_files_read(storage: &Arc<dyn StorageProvider>, cap: Option<usize>) -> (usize, i64) {
    let cold = cold_reader(storage, cap);
    let count = count(
        &cold.reader,
        &format!(
            "SELECT COUNT(*) AS n FROM supertable \
             WHERE title ILIKE '%{EVEN_WORD}%' AND title ILIKE '%{ODD_WORD}%'"
        ),
    );
    (
        cold.storage.max_objects_in_flight_of(SUPERFILE_SUFFIX),
        count,
    )
}

/// Both of a scan's fan-outs before its data reads, the opens and the index
/// predicates' dictionary reads, keep at most the cap of files reading at
/// once.
#[test]
fn a_cold_scan_reads_at_most_the_cap_of_files_at_once() {
    let storage_dir = TempDir::new().expect("storage tempdir");
    let storage = titles_table(storage_dir.path());

    let (peak, count) = peak_files_read(&storage, None);
    assert_eq!(count, 0, "no row holds both words");
    assert!(
        peak > SMALL_CAP,
        "under the default cap more files than the small cap are read at once, so \
         the fan-out the cap limits is real (peak {peak})"
    );
    let (peak, count) = peak_files_read(&storage, Some(SMALL_CAP));
    assert_eq!(count, 0, "no row holds both words");
    assert!(
        (1..=SMALL_CAP).contains(&peak),
        "under a small cap no more files than it are read at once (peak {peak})"
    );
}

/// The most delete sidecars a cold count read at once, from a reader that
/// works on at most `cap` files at once, and the count.
fn peak_sidecars_read(storage: &Arc<dyn StorageProvider>, cap: Option<usize>) -> (usize, i64) {
    let cold = cold_reader(storage, cap);
    let count = count(&cold.reader, "SELECT COUNT(*) AS n FROM supertable");
    (cold.storage.max_objects_in_flight_of(SIDECAR_SUFFIX), count)
}

/// A query reads every file's delete sidecar before it plans, and keeps at
/// most the cap of them reading at once.
#[test]
fn a_cold_query_reads_at_most_the_cap_of_delete_sidecars_at_once() {
    let storage_dir = TempDir::new().expect("storage tempdir");
    let (table, storage) = table_of(storage_dir.path(), |file| {
        vec![format!("kept {file}"), format!("gone {file}")]
    });
    let deleted = table
        .delete(col("title").like(lit("gone%")))
        .expect("delete");
    assert_eq!(
        deleted.n_tombstoned(),
        FILES,
        "one row deleted in each file"
    );
    let live = FILES as i64;

    let (peak, count) = peak_sidecars_read(&storage, None);
    assert_eq!(count, live, "the deleted rows are not counted");
    assert!(
        peak > SMALL_CAP,
        "under the default cap more sidecars than the small cap are read at once, so \
         the fan-out the cap limits is real (peak {peak})"
    );
    let (peak, count) = peak_sidecars_read(&storage, Some(SMALL_CAP));
    assert_eq!(count, live, "the deleted rows are not counted");
    assert!(
        (1..=SMALL_CAP).contains(&peak),
        "under a small cap no more sidecars than it are read at once (peak {peak})"
    );
}
