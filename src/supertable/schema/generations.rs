// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A table whose files were written under ten shapes of its schema, and
//! the reads that must agree with the rows those files hold.
//!
//! Each generation builds a superfile the way a past schema would have —
//! a column missing, renamed, held in a narrower type, stored in another
//! order, or present though since dropped — stamps the table's field ids
//! on it, and commits it through a test-only hook. Every read path then
//! runs against the mixed table and is compared with the rows the files
//! logically hold, and compaction is checked to converge every file to the
//! table's current shape.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Decimal128Array, Float32Array, Float64Array, Int32Array, Int64Array,
    LargeStringArray, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use tempfile::TempDir;

use super::{FieldId, map::FileSchemaMap, with_field_id};
use crate::{
    config::CompactionSettings,
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::{
        VectorSearchOptions,
        builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
        fts::reader::{Bm25SearchOptions, BoolMode},
        vector::{builder::VectorConfig, distance::Metric},
    },
    supertable::{Supertable, SupertableOptions, writer::commit_built_superfile},
    test_helpers::{decimal128_id_field, decimal128_ids},
};

/// Rows per generation.
const ROWS: u64 = 3;
/// Field ids the table mints for its four columns, in declared order.
const TITLE: FieldId = FieldId(1);
const SCORE: FieldId = FieldId(2);
const RATING: FieldId = FieldId(3);
const TAG: FieldId = FieldId(4);
/// The id of a column the table never had: a file carrying it was written
/// under a schema that has since dropped the column.
const DROPPED: FieldId = FieldId(9);

/// One logical row as the table reads it.
#[derive(Debug, Clone, PartialEq)]
struct Row {
    id: i128,
    title: String,
    score: Option<i64>,
    rating: Option<f64>,
    tag: Option<String>,
}

fn table_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("score", DataType::Int64, true),
        Field::new("rating", DataType::Float64, true),
        Field::new("tag", DataType::LargeUtf8, true),
    ]))
}

fn table(dir: &TempDir) -> Supertable {
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let options = SupertableOptions::new(table_schema(), vec![FtsConfig::new("title")], vec![])
        .expect("options")
        .with_storage(storage);
    Supertable::create(options).expect("create")
}

/// A stored column of one generation: its name, id, type and values.
struct Col {
    name: &'static str,
    id: FieldId,
    array: ArrayRef,
}

fn i64s(values: &[i64]) -> ArrayRef {
    Arc::new(Int64Array::from(values.to_vec()))
}

fn i32s(values: &[i32]) -> ArrayRef {
    Arc::new(Int32Array::from(values.to_vec()))
}

fn f64s(values: &[f64]) -> ArrayRef {
    Arc::new(Float64Array::from(values.to_vec()))
}

fn f32s(values: &[f32]) -> ArrayRef {
    Arc::new(Float32Array::from(values.to_vec()))
}

fn large_strs(values: &[&str]) -> ArrayRef {
    Arc::new(LargeStringArray::from(values.to_vec()))
}

fn strs(values: &[&str]) -> ArrayRef {
    Arc::new(StringArray::from(values.to_vec()))
}

/// Build one generation's superfile: the id column, then `cols` in the
/// order given, every field stamped with its id, `title` indexed when
/// present.
fn build_generation(first_id: u64, cols: &[Col]) -> Bytes {
    let mut fields = vec![with_field_id(
        &decimal128_id_field("_id"),
        FieldId::ID_COLUMN,
    )];
    let mut arrays: Vec<ArrayRef> = vec![Arc::new(decimal128_ids(first_id..first_id + ROWS))];
    for col in cols {
        fields.push(with_field_id(
            &Field::new(col.name, col.array.data_type().clone(), true),
            col.id,
        ));
        arrays.push(Arc::clone(&col.array));
    }
    let schema = Arc::new(Schema::new(fields));
    let fts = cols
        .iter()
        .filter(|c| c.id == TITLE)
        .map(|c| FtsConfig::new(c.name))
        .collect();
    let opts = BuilderOptions::new(Arc::clone(&schema), "_id", fts, vec![]);
    let mut builder = SuperfileBuilder::new(opts).expect("builder");
    let batch = RecordBatch::try_new(schema, arrays).expect("batch");
    builder.add_batch(&batch, &[]).expect("add_batch");
    Bytes::from(builder.finish().expect("finish"))
}

/// The ten generations and the rows they logically hold.
fn generations() -> Vec<(Bytes, Vec<Row>)> {
    let titles =
        |g: u64| -> Vec<String> { (0..ROWS).map(|r| format!("shared gen{g} row{r}")).collect() };
    let title_col = |g: u64| -> Col {
        let t = titles(g);
        Col {
            name: "title",
            id: TITLE,
            array: large_strs(&t.iter().map(String::as_str).collect::<Vec<_>>()),
        }
    };
    let mut out = Vec::new();
    let mut generation = |g: u64, cols: Vec<Col>, rows: Vec<Row>| {
        out.push((build_generation(g * 100, &cols), rows));
    };
    let rows_of =
        |g: u64, score: [Option<i64>; 3], rating: [Option<f64>; 3], tag: [Option<&str>; 3]| {
            titles(g)
                .into_iter()
                .enumerate()
                .map(|(r, title)| Row {
                    id: (g * 100 + r as u64) as i128,
                    title,
                    score: score[r],
                    rating: rating[r],
                    tag: tag[r].map(str::to_owned),
                })
                .collect::<Vec<_>>()
        };

    // 0: the table's first shape — title and score only.
    generation(
        0,
        vec![
            title_col(0),
            Col {
                name: "score",
                id: SCORE,
                array: i64s(&[1, 2, 3]),
            },
        ],
        rows_of(0, [Some(1), Some(2), Some(3)], [None; 3], [None; 3]),
    );
    // 1: score held narrower than the table has it now.
    generation(
        1,
        vec![
            title_col(1),
            Col {
                name: "score",
                id: SCORE,
                array: i32s(&[11, 12, 13]),
            },
        ],
        rows_of(1, [Some(11), Some(12), Some(13)], [None; 3], [None; 3]),
    );
    // 2: score under its former name.
    generation(
        2,
        vec![
            title_col(2),
            Col {
                name: "points",
                id: SCORE,
                array: i64s(&[21, 22, 23]),
            },
        ],
        rows_of(2, [Some(21), Some(22), Some(23)], [None; 3], [None; 3]),
    );
    // 3: rating added, tag not yet.
    generation(
        3,
        vec![
            title_col(3),
            Col {
                name: "score",
                id: SCORE,
                array: i64s(&[31, 32, 33]),
            },
            Col {
                name: "rating",
                id: RATING,
                array: f64s(&[3.1, 3.2, 3.3]),
            },
        ],
        rows_of(
            3,
            [Some(31), Some(32), Some(33)],
            [Some(3.1), Some(3.2), Some(3.3)],
            [None; 3],
        ),
    );
    // 4: the current shape.
    generation(
        4,
        vec![
            title_col(4),
            Col {
                name: "score",
                id: SCORE,
                array: i64s(&[41, 42, 43]),
            },
            Col {
                name: "rating",
                id: RATING,
                array: f64s(&[4.1, 4.2, 4.3]),
            },
            Col {
                name: "tag",
                id: TAG,
                array: large_strs(&["a", "b", "c"]),
            },
        ],
        rows_of(
            4,
            [Some(41), Some(42), Some(43)],
            [Some(4.1), Some(4.2), Some(4.3)],
            [Some("a"), Some("b"), Some("c")],
        ),
    );
    // 5: the current shape plus a column dropped since.
    generation(
        5,
        vec![
            title_col(5),
            Col {
                name: "score",
                id: SCORE,
                array: i64s(&[51, 52, 53]),
            },
            Col {
                name: "rating",
                id: RATING,
                array: f64s(&[5.1, 5.2, 5.3]),
            },
            Col {
                name: "tag",
                id: TAG,
                array: large_strs(&["d", "e", "f"]),
            },
            Col {
                name: "extra",
                id: DROPPED,
                array: i64s(&[-1, -2, -3]),
            },
        ],
        rows_of(
            5,
            [Some(51), Some(52), Some(53)],
            [Some(5.1), Some(5.2), Some(5.3)],
            [Some("d"), Some("e"), Some("f")],
        ),
    );
    // 6: rating held as f32, score and tag absent.
    generation(
        6,
        vec![
            title_col(6),
            Col {
                name: "rating",
                id: RATING,
                array: f32s(&[6.5, 6.25, 6.0]),
            },
        ],
        rows_of(6, [None; 3], [Some(6.5), Some(6.25), Some(6.0)], [None; 3]),
    );
    // 7: tag held as a narrow string.
    generation(
        7,
        vec![
            title_col(7),
            Col {
                name: "tag",
                id: TAG,
                array: strs(&["g", "h", "i"]),
            },
        ],
        rows_of(7, [None; 3], [None; 3], [Some("g"), Some("h"), Some("i")]),
    );
    // 8: every column, in another order.
    generation(
        8,
        vec![
            Col {
                name: "tag",
                id: TAG,
                array: large_strs(&["j", "k", "l"]),
            },
            Col {
                name: "rating",
                id: RATING,
                array: f64s(&[8.1, 8.2, 8.3]),
            },
            Col {
                name: "score",
                id: SCORE,
                array: i64s(&[81, 82, 83]),
            },
            title_col(8),
        ],
        rows_of(
            8,
            [Some(81), Some(82), Some(83)],
            [Some(8.1), Some(8.2), Some(8.3)],
            [Some("j"), Some("k"), Some("l")],
        ),
    );
    // 9: the current shape once more.
    generation(
        9,
        vec![
            title_col(9),
            Col {
                name: "score",
                id: SCORE,
                array: i64s(&[91, 92, 93]),
            },
            Col {
                name: "rating",
                id: RATING,
                array: f64s(&[9.1, 9.2, 9.3]),
            },
            Col {
                name: "tag",
                id: TAG,
                array: large_strs(&["m", "n", "o"]),
            },
        ],
        rows_of(
            9,
            [Some(91), Some(92), Some(93)],
            [Some(9.1), Some(9.2), Some(9.3)],
            [Some("m"), Some("n"), Some("o")],
        ),
    );
    out
}

/// The table with every generation committed, and the rows it holds.
fn mixed_table(dir: &TempDir) -> (Supertable, Vec<Row>) {
    let st = table(dir);
    let mut rows = Vec::new();
    for (bytes, generation_rows) in generations() {
        commit_built_superfile(&st, bytes).expect("commit generation");
        rows.extend(generation_rows);
    }
    rows.sort_by_key(|r| r.id);
    (st, rows)
}

/// `SELECT _id, title, score, rating, tag … ORDER BY _id` as rows.
fn rows_of_batches(batches: &[RecordBatch]) -> Vec<Row> {
    let mut rows = Vec::new();
    for b in batches {
        let ids = b
            .column_by_name("_id")
            .and_then(|c| c.as_any().downcast_ref::<Decimal128Array>())
            .expect("_id");
        let titles = b
            .column_by_name("title")
            .and_then(|c| c.as_any().downcast_ref::<LargeStringArray>())
            .expect("title");
        let scores = b
            .column_by_name("score")
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
            .expect("score");
        let ratings = b
            .column_by_name("rating")
            .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
            .expect("rating");
        let tags = b
            .column_by_name("tag")
            .and_then(|c| c.as_any().downcast_ref::<LargeStringArray>())
            .expect("tag");
        for i in 0..b.num_rows() {
            rows.push(Row {
                id: ids.value(i),
                title: titles.value(i).to_owned(),
                score: scores.is_valid(i).then(|| scores.value(i)),
                rating: ratings.is_valid(i).then(|| ratings.value(i)),
                tag: tags.is_valid(i).then(|| tags.value(i).to_owned()),
            });
        }
    }
    rows.sort_by_key(|r| r.id);
    rows
}

#[test]
fn sql_reads_every_generation_as_the_table_sees_it() {
    let dir = TempDir::new().expect("tempdir");
    let (st, expected) = mixed_table(&dir);
    let reader = st.reader().expect("reader");
    let batches = reader
        .query_sql("SELECT _id, title, score, rating, tag FROM supertable ORDER BY _id")
        .expect("sql");
    assert_eq!(rows_of_batches(&batches), expected);

    // A predicate on a column that files hold in other types, under other
    // names, or not at all: the adapter casts, renames and null-fills
    // before the filter runs, and pruning never drops a file that holds a
    // match.
    let batches = reader
        .query_sql(
            "SELECT _id, title, score, rating, tag FROM supertable WHERE score > 20 ORDER BY _id",
        )
        .expect("sql");
    let want: Vec<Row> = expected
        .iter()
        .filter(|r| r.score.is_some_and(|s| s > 20))
        .cloned()
        .collect();
    assert_eq!(rows_of_batches(&batches), want);

    let batches = reader
        .query_sql("SELECT _id, title, score, rating, tag FROM supertable WHERE tag = 'h' OR rating < 3.5 ORDER BY _id")
        .expect("sql");
    let want: Vec<Row> = expected
        .iter()
        .filter(|r| r.tag.as_deref() == Some("h") || r.rating.is_some_and(|x| x < 3.5))
        .cloned()
        .collect();
    assert_eq!(rows_of_batches(&batches), want);
}

#[test]
fn full_text_hits_materialise_every_generation_through_its_map() {
    let dir = TempDir::new().expect("tempdir");
    let (st, expected) = mixed_table(&dir);
    let reader = st.reader().expect("reader");
    let batches = reader
        .bm25_search(
            "title",
            "shared",
            expected.len(),
            Bm25SearchOptions::new(),
            Some(&["_id", "title", "score", "rating", "tag"]),
        )
        .expect("search");
    assert_eq!(
        rows_of_batches(&batches),
        expected,
        "every row of every generation is a hit"
    );

    let batches = reader
        .bm25_search(
            "title",
            "gen2",
            10,
            Bm25SearchOptions::new(),
            Some(&["_id", "title", "score", "rating", "tag"]),
        )
        .expect("search");
    let want: Vec<Row> = expected
        .iter()
        .filter(|r| r.title.contains("gen2"))
        .cloned()
        .collect();
    assert_eq!(
        rows_of_batches(&batches),
        want,
        "a renamed column reads by id"
    );
}

#[test]
fn only_files_in_the_current_shape_take_the_identity_path() {
    let dir = TempDir::new().expect("tempdir");
    let (st, _) = mixed_table(&dir);
    let reader = st.reader().expect("reader");
    let manifest = reader.manifest();
    let table = manifest.table_schema();
    let mut entries = manifest.get_all_superfiles().to_vec();
    entries.sort_by_key(|e| e.id_min);
    let identity: Vec<bool> = entries
        .iter()
        .map(|e| {
            let physical = e.physical_schema.as_ref().expect("stamped");
            FileSchemaMap::new(&table, "_id", physical).is_identity()
        })
        .collect();
    // Generations 4, 5 (extra column), 8 (reordered) and 9 hold every table
    // column under its name and type; the rest are missing, renamed or
    // retyped somewhere.
    assert_eq!(
        identity,
        vec![
            false, false, false, false, true, true, false, false, true, true
        ]
    );
    let stale: Vec<bool> = entries
        .iter()
        .map(|e| {
            let physical = e.physical_schema.as_ref().expect("stamped");
            FileSchemaMap::new(&table, "_id", physical).has_stale_type()
        })
        .collect();
    assert_eq!(
        stale,
        vec![
            false, true, false, false, false, false, true, true, false, false
        ],
        "generations 1 (Int32), 6 (Float32) and 7 (Utf8) hold a column in another type"
    );
}

#[test]
fn compaction_converges_every_generation_to_the_current_shape() {
    let dir = TempDir::new().expect("tempdir");
    let (st, expected) = mixed_table(&dir);
    st.compact(&CompactionSettings {
        min_fill_percent: 0,
        min_superfiles_for_merge: 2,
        ..CompactionSettings::default()
    })
    .expect("compact");
    let reader = st.reader().expect("reader");
    let manifest = reader.manifest();
    let table = manifest.table_schema();
    let entries = manifest.get_all_superfiles();
    assert_eq!(entries.len(), 1, "ten small files merge into one");
    let physical = entries[0].physical_schema.as_ref().expect("stamped");
    assert!(FileSchemaMap::new(&table, "_id", physical).is_identity());
    assert!(
        physical.column_by_id(DROPPED).is_none(),
        "a column the table no longer has is not carried into the output"
    );
    let batches = reader
        .query_sql("SELECT _id, title, score, rating, tag FROM supertable ORDER BY _id")
        .expect("sql");
    assert_eq!(rows_of_batches(&batches), expected);
    let batches = reader
        .bm25_search(
            "title",
            "shared",
            expected.len(),
            Bm25SearchOptions::new(),
            Some(&["_id", "title", "score", "rating", "tag"]),
        )
        .expect("search");
    assert_eq!(rows_of_batches(&batches), expected);
}

#[test]
fn a_stale_typed_file_is_rewritten_alone_whatever_the_size_rules_say() {
    let dir = TempDir::new().expect("tempdir");
    let (st, expected) = mixed_table(&dir);
    // A zero target makes every file "above target" and a high floor
    // keeps ordinary files from merging: only the stale-typed ones move.
    st.compact(&CompactionSettings {
        target_superfile_size_mb: 0,
        min_fill_percent: 100,
        min_superfiles_for_merge: 1_000,
        ..CompactionSettings::default()
    })
    .expect("compact");
    let reader = st.reader().expect("reader");
    let manifest = reader.manifest();
    let table = manifest.table_schema();
    let entries = manifest.get_all_superfiles();
    assert_eq!(entries.len(), 10, "each stale file is rewritten by itself");
    assert!(
        entries.iter().all(|e| {
            let physical = e.physical_schema.as_ref().expect("stamped");
            !FileSchemaMap::new(&table, "_id", physical).has_stale_type()
        }),
        "no file holds a column in a type the table has moved on from"
    );
    let batches = reader
        .query_sql("SELECT _id, title, score, rating, tag FROM supertable ORDER BY _id")
        .expect("sql");
    assert_eq!(rows_of_batches(&batches), expected);
}

/// Dimension of the vector column in the vector fixture.
const DIM: usize = 16;
/// The vector column's id in the vector fixture (`title` is 1).
const EMB: FieldId = FieldId(2);

/// A table with a text and a vector column, holding one file written with
/// vectors and one written before the vector column existed.
fn vector_table(dir: &TempDir) -> Supertable {
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("emb", DataType::FixedSizeList(item, DIM as i32), true),
    ]));
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let options = SupertableOptions::new(
        schema,
        vec![FtsConfig::new("title")],
        vec![VectorConfig::new("emb".into(), DIM, 7, Metric::L2Sq)],
    )
    .expect("options")
    .with_storage(storage);
    let st = Supertable::create(options).expect("create");

    let scalar = Arc::new(Schema::new(vec![
        with_field_id(&decimal128_id_field("_id"), FieldId::ID_COLUMN),
        with_field_id(&Field::new("title", DataType::LargeUtf8, false), TITLE),
    ]));
    let titles = |g: u64| {
        large_strs(
            &(0..ROWS)
                .map(|r| format!("shared gen{g} row{r}"))
                .collect::<Vec<_>>()
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
        )
    };
    // Generation 0 carries vectors along the first axes; generation 1 was
    // written before the vector column existed.
    let batch0 = RecordBatch::try_new(
        Arc::clone(&scalar),
        vec![Arc::new(decimal128_ids(0..ROWS)), titles(0)],
    )
    .expect("batch");
    let mut flat = vec![0.0f32; ROWS as usize * DIM];
    for row in 0..ROWS as usize {
        flat[row * DIM + row] = 1.0;
    }
    let mut with_vectors = SuperfileBuilder::new(
        BuilderOptions::new(
            Arc::clone(&scalar),
            "_id",
            vec![FtsConfig::new("title")],
            vec![VectorConfig::new("emb".into(), DIM, 7, Metric::L2Sq)],
        )
        .with_vector_field_ids([("emb".to_string(), EMB)]),
    )
    .expect("builder");
    with_vectors
        .add_batch(&batch0, &[&flat])
        .expect("add_batch");
    commit_built_superfile(&st, Bytes::from(with_vectors.finish().expect("finish")))
        .expect("commit");

    let batch1 = RecordBatch::try_new(
        Arc::clone(&scalar),
        vec![Arc::new(decimal128_ids(100..100 + ROWS)), titles(1)],
    )
    .expect("batch");
    let mut without = SuperfileBuilder::new(BuilderOptions::new(
        scalar,
        "_id",
        vec![FtsConfig::new("title")],
        vec![],
    ))
    .expect("builder");
    without.add_batch(&batch1, &[]).expect("add_batch");
    commit_built_superfile(&st, Bytes::from(without.finish().expect("finish"))).expect("commit");
    st
}

fn ids_of(batches: &[RecordBatch]) -> Vec<i128> {
    let mut ids: Vec<i128> = batches
        .iter()
        .flat_map(|b| {
            let col = b
                .column_by_name("_id")
                .and_then(|c| c.as_any().downcast_ref::<Decimal128Array>())
                .expect("_id");
            col.values().to_vec()
        })
        .collect();
    ids.sort();
    ids
}

/// A file written before the vector column existed contributes nothing to
/// a vector or hybrid search and everything to a text search.
#[test]
fn a_file_without_the_vector_column_contributes_nothing_to_vector_search() {
    let dir = TempDir::new().expect("tempdir");
    let st = vector_table(&dir);
    let reader = st.reader().expect("reader");
    let mut q = vec![0.0f32; DIM];
    q[1] = 1.0;

    let batches = reader
        .vector_search(
            "emb",
            &q,
            10,
            VectorSearchOptions::new(),
            None,
            Some(&["_id", "title"]),
        )
        .expect("vector search");
    assert_eq!(
        ids_of(&batches),
        vec![0, 1, 2],
        "only the file with vectors answers"
    );

    let hits = reader
        .hybrid_search(
            "title",
            "shared",
            BoolMode::Or,
            "emb",
            &q,
            VectorSearchOptions::new(),
            10,
        )
        .expect("hybrid search");
    // Hybrid fuses the two legs as a union: every text hit ranks, and the
    // vector leg adds its signal only for the file that holds vectors.
    assert_eq!(hits.len(), 6, "a file without vectors still ranks on text");

    let batches = reader
        .bm25_search(
            "title",
            "shared",
            10,
            Bm25SearchOptions::new(),
            Some(&["_id"]),
        )
        .expect("bm25");
    assert_eq!(
        ids_of(&batches),
        vec![0, 1, 2, 100, 101, 102],
        "text search sees both files"
    );
}
