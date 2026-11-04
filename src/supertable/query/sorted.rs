// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`Supertable::sorted_search`]: the first `k` rows of the table by one
//! range-indexed column, ties broken by `_id` ascending, without SQL.
//!
//! A range index holds its column in order, and keeps the rows sharing a
//! value in `_id` order, so a superfile's first `k` entries in it are its
//! first `k` rows by the column and `_id`. The answer is the best `k` of
//! those across superfiles: each superfile reads a block or two of its
//! index, the candidates are ordered by key and `_id`, and only the `k`
//! kept are resolved to rows. No statement is planned, so the cost is the
//! index reads and the `k` rows, whatever the table's size.
//!
//! The index answers only when every superfile holds the column in a range
//! index with its ties in `_id` order, and when at least `k` rows have a
//! value (a missing value sorts last and is in no index). Otherwise the
//! same sort runs as SQL, which answers alike, only slower.

use std::{sync::Arc, time::Instant};

use arrow::record_batch::RecordBatch;
use roaring::RoaringBitmap;
use tracing::{Instrument, debug};

use crate::{
    InfinoError,
    superfile::{SuperfileReader, range::KeyKind},
    supertable::{
        error::QueryError,
        handle::{Supertable, SupertableReader},
        manifest::SuperfileEntry,
        query::{
            SuperfileHit,
            dispatch::{self, tag_hits, tombstone_deny_set},
            exec::common::resolve_hits_named,
        },
        reader_cache::disk::ForegroundQueryGuard,
        tombstones::SidecarCache,
    },
    utils::trace::detail_span,
};

/// The name the trailing synthesized column has in a search's output.
const SCORE_COLUMN: &str = "score";

/// The table a reader's SQL statements read.
const SQL_TABLE: &str = "supertable";

/// One superfile's candidates: `(key, hit)` for its first rows by the key.
type Candidates = Vec<(u64, SuperfileHit)>;

impl SupertableReader {
    /// The first `k` rows by `column` (descending with `descending`), ties
    /// by `_id` ascending, as hits in that order, read from the superfiles'
    /// range indexes. `None` when the indexes cannot answer: a superfile
    /// without the column's index, or without its ties in `_id` order, or
    /// fewer than `k` rows with a value. An error when `column` has no
    /// range index at all.
    pub(crate) async fn sorted_hits_async(
        &self,
        column: &str,
        descending: bool,
        k: usize,
    ) -> Result<Option<Vec<SuperfileHit>>, QueryError> {
        let manifest = self.manifest();
        let range_cols = manifest.table_schema().range_column_types();
        let Some((field, data_type)) = range_cols.get(column) else {
            return Err(QueryError::InvalidQuery(format!(
                "sorted_search: column '{column}' has no range index"
            )));
        };
        let Some(kind) = KeyKind::of(data_type) else {
            return Err(QueryError::InvalidQuery(format!(
                "sorted_search: column '{column}' is not of a range-indexed type"
            )));
        };
        if k == 0 {
            return Ok(Some(Vec::new()));
        }
        let field = *field;
        let units: Vec<_> = manifest
            .superfiles
            .iter()
            .cloned()
            .map(|e| (e, ()))
            .collect();
        let body = move |r: Arc<SuperfileReader>,
                         entry: Arc<SuperfileEntry>,
                         tombstone_cache: Option<Arc<SidecarCache>>,
                         now: Instant,
                         _: ()| async move {
            let deleted = match &tombstone_cache {
                Some(cache) => tombstone_deny_set(cache, entry.superfile_id, now)?,
                None => None,
            };
            let none = RoaringBitmap::new();
            let ends = r
                .range_ends(
                    field,
                    kind,
                    (0, u64::MAX),
                    k,
                    descending,
                    deleted.as_deref().unwrap_or(&none),
                )
                .await
                .map_err(QueryError::from)?;
            let Some(ends) = ends.filter(|ends| ends.ties_in_id_order) else {
                return Ok::<Option<Candidates>, QueryError>(None);
            };
            let keys: Vec<u64> = ends.pairs.iter().map(|(key, _)| *key).collect();
            let hits = tag_hits(
                &entry,
                ends.pairs.iter().map(|(_, row)| (*row, 0.0)).collect(),
            );
            Ok(Some(keys.into_iter().zip(hits).collect()))
        };
        let per_superfile = dispatch::fanout_with(self, units, true, self.open_intent(), body)
            .instrument(detail_span!("sorted.fanout"))
            .await?;
        let Some(candidates) = per_superfile
            .into_iter()
            .collect::<Option<Vec<Candidates>>>()
        else {
            return Ok(None);
        };
        let (keys, mut hits): (Vec<u64>, Vec<SuperfileHit>) =
            candidates.into_iter().flatten().unzip();
        if hits.len() < k {
            return Ok(None);
        }
        dispatch::attach_stable_ids_to_hits(self, &mut hits).await?;
        let mut order: Vec<usize> = (0..hits.len()).collect();
        order.sort_by(|&a, &b| {
            let by_key = match descending {
                false => keys[a].cmp(&keys[b]),
                true => keys[b].cmp(&keys[a]),
            };
            by_key.then_with(|| hits[a].stable_id.cmp(&hits[b].stable_id))
        });
        Ok(Some(order.into_iter().take(k).map(|i| hits[i]).collect()))
    }

    /// Sync [`Self::sorted_hits_async`] through the sync→async bridge.
    pub(crate) fn sorted_hits(
        &self,
        column: &str,
        descending: bool,
        k: usize,
    ) -> Result<Option<Vec<SuperfileHit>>, QueryError> {
        let _foreground = ForegroundQueryGuard::enter();
        self.block_on(self.sorted_hits_async(column, descending, k))
    }
}

impl Supertable {
    /// The first `k` rows by `column`, a range-indexed column, ascending
    /// or (with `descending`) descending, rows sharing a value ordered by
    /// `_id` ascending, and rows without a value last: what `SELECT … ORDER
    /// BY column, _id LIMIT k` answers, without planning a statement. Read
    /// from the column's range index, a block or two per superfile, however
    /// large the table. Returns Arrow rows in that order like
    /// [`Supertable::bm25_search`], with `score` fixed at `0.0`;
    /// `projection` follows the same rules.
    pub fn sorted_search(
        &self,
        column: &str,
        descending: bool,
        k: usize,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        debug!(column, descending, k, "sorted_search");
        let close_out = self.close_out();
        let reader = self.reader()?;
        let context = |e: QueryError| InfinoError::from(e).with_context("sorted_search", None);
        reader.check_projection(projection).map_err(context)?;
        let batches = match reader.sorted_hits(column, descending, k).map_err(context)? {
            Some(hits) => vec![
                self.block_on_query(
                    resolve_hits_named(&reader, &hits, projection)
                        .instrument(detail_span!("search.resolve", hits = hits.len())),
                )
                .map_err(context)?,
            ],
            None => reader
                .run_sql(&sorted_statement(
                    SQL_TABLE,
                    column,
                    descending,
                    k,
                    projection,
                    &reader.options().id_column,
                ))
                .map_err(context)?,
        };
        close_out.finish_batches(&batches);
        Ok(batches)
    }
}

/// The SQL over `table` that answers [`Supertable::sorted_search`] where the
/// range indexes cannot (and over the wire, where a hosted table has no
/// sorted call): the same order and limit, with the same output columns
/// (`_id` and `score` when `projection` names none, `score` as `0.0`).
pub(crate) fn sorted_statement(
    table: &str,
    column: &str,
    descending: bool,
    k: usize,
    projection: Option<&[&str]>,
    id_column: &str,
) -> String {
    let default = [id_column, SCORE_COLUMN];
    let select: Vec<String> = projection
        .unwrap_or(&default)
        .iter()
        .map(|name| match *name {
            SCORE_COLUMN => format!("CAST(0 AS FLOAT) AS {}", quote(SCORE_COLUMN)),
            name => quote(name),
        })
        .collect();
    let direction = if descending { "DESC" } else { "ASC" };
    format!(
        "SELECT {} FROM {} ORDER BY {} {direction} NULLS LAST, {} ASC LIMIT {k}",
        select.join(", "),
        quote(table),
        quote(column),
        quote(id_column),
    )
}

/// `name` as a quoted SQL identifier.
fn quote(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, Decimal128Array, Float32Array, Int64Array};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::{col, lit};
    use tempfile::TempDir;

    use super::*;
    use crate::{
        config::OptimizeOptions,
        storage::{LocalFsStorageProvider, StorageProvider},
        supertable::SupertableOptions,
    };

    /// Superfiles the table is written as, one commit each.
    const FILES: i64 = 3;
    /// Rows in each.
    const ROWS_PER_FILE: i64 = 3000;
    /// Distinct `key` values, so every cutoff ties on many rows.
    const KEYS: i64 = 50;
    /// Every this-many rows `key` is null.
    const NULL_EVERY: i64 = 17;
    /// Rows the first delete removes: those with `seq` below this.
    const DELETED_BELOW: i64 = 200;
    /// Rows the second delete removes: those with `seq` from this on, a
    /// larger share of the last superfile than the first takes of the first,
    /// so an optimize merges the last superfile first.
    const LATE_DELETED_FROM: i64 = 8_000;
    /// The `(descending, k)` each pass asks for; the last is past the rows
    /// with a value, so the nulls come in and the index cannot answer.
    const ASKS: &[(bool, usize)] = &[
        (false, 1),
        (false, 10),
        (true, 10),
        (true, 125),
        (false, 600),
        (true, 7_700),
    ];

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("key", DataType::Int64, true),
        ]))
    }

    /// `key` of row `seq`: not in the order the rows arrive, null now and
    /// then.
    fn key(seq: i64) -> Option<i64> {
        (seq % NULL_EVERY != 0).then_some((seq * 31) % KEYS)
    }

    /// The table: `key` range indexed, written in [`FILES`] commits, its
    /// rows below [`DELETED_BELOW`] and from [`LATE_DELETED_FROM`] deleted.
    fn table(dir: &TempDir) -> Supertable {
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let options = SupertableOptions::new(schema(), vec![], vec![])
            .expect("options")
            .with_storage(storage)
            .with_range_indexes(&["key".to_string()]);
        let st = Supertable::create(options).expect("create");
        for file in 0..FILES {
            let seqs: Vec<i64> = (file * ROWS_PER_FILE..(file + 1) * ROWS_PER_FILE).collect();
            let keys: Vec<Option<i64>> = seqs.iter().map(|s| key(*s)).collect();
            let batch = RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(Int64Array::from(seqs)),
                    Arc::new(Int64Array::from(keys)),
                ],
            )
            .expect("batch");
            let mut writer = st.writer().expect("writer");
            writer.append(&batch).expect("append");
            writer.commit().expect("commit");
        }
        st.delete(col("seq").lt(lit(DELETED_BELOW)))
            .expect("delete");
        st.delete(col("seq").gt_eq(lit(LATE_DELETED_FROM)))
            .expect("delete");
        st
    }

    /// One `Int64` column of `batches`, by name, across them in order.
    fn ints(batches: &[RecordBatch], name: &str) -> Vec<Option<i64>> {
        batches
            .iter()
            .flat_map(|b| {
                let column = b
                    .column_by_name(name)
                    .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
                    .expect(name)
                    .clone();
                (0..column.len()).map(move |i| column.is_valid(i).then(|| column.value(i)))
            })
            .collect()
    }

    /// The `_id` column of `batches`, in order.
    fn ids(batches: &[RecordBatch]) -> Vec<i128> {
        batches
            .iter()
            .flat_map(|b| {
                let ids = b
                    .column_by_name("_id")
                    .and_then(|c| c.as_any().downcast_ref::<Decimal128Array>())
                    .expect("_id")
                    .clone();
                (0..ids.len()).map(move |i| ids.value(i))
            })
            .collect()
    }

    /// Every ask answers as the same sort in SQL does, row for row, before
    /// and after an optimize merges the superfiles out of `_id` order; the
    /// asks within the rows with a value come from the range indexes.
    #[test]
    fn sorted_search_answers_as_the_sql_sort() {
        let dir = TempDir::new().expect("tempdir");
        let st = table(&dir);
        for pass in ["appended", "optimized"] {
            if pass == "optimized" {
                st.optimize(&OptimizeOptions::default()).expect("optimize");
            }
            let reader = st.reader().expect("reader");
            for &(descending, k) in ASKS {
                let got = st
                    .sorted_search("key", descending, k, Some(&["_id", "seq", "key"]))
                    .expect("sorted_search");
                let direction = if descending { "DESC" } else { "ASC" };
                let want = reader
                    .query_sql(&format!(
                        "SELECT _id, seq, key FROM supertable ORDER BY key {direction} NULLS LAST, _id ASC LIMIT {k}"
                    ))
                    .expect("sql");
                assert_eq!(
                    ids(&got),
                    ids(&want),
                    "{pass}: descending {descending}, k {k}"
                );
                assert_eq!(
                    ints(&got, "seq"),
                    ints(&want, "seq"),
                    "{pass}: {descending} {k}"
                );
                assert_eq!(
                    ints(&got, "key"),
                    ints(&want, "key"),
                    "{pass}: {descending} {k}"
                );
                let from_index = reader
                    .sorted_hits("key", descending, k)
                    .expect("hits")
                    .is_some();
                let with_values = ints(&want, "key").iter().all(Option::is_some);
                assert_eq!(from_index, with_values, "{pass}: {descending} {k}");
            }
        }
    }

    /// With no projection the rows are `_id` and a `score` of `0.0`, on the
    /// index path and on the SQL one alike.
    #[test]
    fn a_bare_sorted_search_returns_ids_and_a_zero_score() {
        let dir = TempDir::new().expect("tempdir");
        let st = table(&dir);
        for k in [10, 7_700] {
            let batches = st
                .sorted_search("key", false, k, None)
                .expect("sorted_search");
            let names: Vec<String> = batches[0]
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect();
            assert_eq!(
                names,
                vec!["_id".to_string(), SCORE_COLUMN.to_string()],
                "k {k}"
            );
            let scores = batches[0]
                .column(1)
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("score");
            assert!(scores.values().iter().all(|s| *s == 0.0), "k {k}");
        }
    }

    /// A column without a range index is refused by name, and a `k` of
    /// zero answers no rows.
    #[test]
    fn sorted_search_needs_a_range_index_and_takes_zero() {
        let dir = TempDir::new().expect("tempdir");
        let st = table(&dir);
        let error = st
            .sorted_search("seq", false, 10, None)
            .expect_err("seq has no range index");
        assert!(error.to_string().contains("seq"), "{error}");
        let none = st.sorted_search("key", true, 0, None).expect("k 0");
        assert_eq!(none.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
    }
}
