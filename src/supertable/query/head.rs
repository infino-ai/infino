// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`Supertable::head`]: the first `k` live rows of the table in storage
//! order, without SQL.
//!
//! A page of rows in no particular order, `SELECT … LIMIT k` with no
//! `WHERE` and no `ORDER BY`, plans a statement whose planning costs more
//! than reading the rows. The rows are the superfiles' own, in the order
//! the manifest lists the superfiles and each superfile holds its rows,
//! less those deleted: only the superfiles that can hold the page are
//! read, and only the `k` rows kept are resolved.

use std::{sync::Arc, time::Instant};

use arrow::record_batch::RecordBatch;
use tracing::{Instrument, debug};

use crate::{
    InfinoError,
    superfile::SuperfileReader,
    supertable::{
        error::QueryError,
        handle::{Supertable, SupertableReader},
        manifest::SuperfileEntry,
        query::{
            SuperfileHit,
            dispatch::{self, tag_hits, tombstone_deny_set},
            hits_page::Page,
        },
        tombstones::SidecarCache,
    },
    utils::trace::detail_span,
};

impl SupertableReader {
    /// The first `k` live rows in storage order, as hits: the superfiles in
    /// the manifest's order, each one's rows in its own order, deleted rows
    /// skipped. Superfiles are read a group at a time, each group holding
    /// enough rows for the rest of the page were none of them deleted, so a
    /// page of a large table reads the first superfile or two. Room for the
    /// hits is reserved only as rows are found, so a `k` past the rows the
    /// table holds reserves nothing for them.
    pub(crate) async fn head_hits_async(&self, k: usize) -> Result<Vec<SuperfileHit>, QueryError> {
        let mut hits: Vec<SuperfileHit> = Vec::new();
        let manifest = self.manifest();
        let mut entries = manifest.superfiles.iter().cloned();
        while hits.len() < k {
            let want = k - hits.len();
            let mut units: Vec<(Arc<SuperfileEntry>, usize)> = Vec::new();
            let mut held = 0u64;
            for entry in entries.by_ref() {
                held = held.saturating_add(entry.n_docs);
                units.push((entry, want));
                if held >= want as u64 {
                    break;
                }
            }
            if units.is_empty() {
                break;
            }
            let body = |_: Arc<SuperfileReader>,
                        entry: Arc<SuperfileEntry>,
                        tombstone_cache: Option<Arc<SidecarCache>>,
                        now: Instant,
                        want: usize| async move {
                let deleted = match &tombstone_cache {
                    Some(cache) => tombstone_deny_set(cache, entry.superfile_id, now)?,
                    None => None,
                };
                let rows = u32::try_from(entry.n_docs).unwrap_or(u32::MAX);
                let live: Vec<(u32, f32)> = (0..rows)
                    .filter(|row| deleted.as_ref().is_none_or(|d| !d.contains(*row)))
                    .take(want)
                    .map(|row| (row, 0.0))
                    .collect();
                Ok::<Vec<SuperfileHit>, QueryError>(tag_hits(&entry, live))
            };
            let per_superfile = dispatch::fanout_with(self, units, true, self.open_intent(), body)
                .instrument(detail_span!("head.fanout"))
                .await?;
            hits.extend(per_superfile.into_iter().flatten().take(want));
        }
        dispatch::attach_stable_ids_to_hits(self, &mut hits).await?;
        Ok(hits)
    }
}

impl Supertable {
    /// The first `k` live rows of the table in storage order, without
    /// planning a statement: rows `SELECT … LIMIT k` with no `WHERE` and no
    /// `ORDER BY` could return, in the order the superfiles hold them, with
    /// deleted rows left out. Fewer come back when the table holds fewer.
    /// Returns Arrow rows like [`Supertable::bm25_search`], with `score`
    /// fixed at `0.0`; `projection` follows the same rules.
    pub fn head(
        &self,
        k: usize,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        debug!(k, "head");
        self.hits_page("head", k, projection, |reader, k| {
            reader.block_on(reader.head_hits_async(k)).map(Page::Hits)
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, sync::Arc};

    use arrow_array::{Array, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::{col, lit};
    use tempfile::TempDir;

    use crate::{
        config::OptimizeOptions,
        storage::{LocalFsStorageProvider, StorageProvider},
        supertable::{Supertable, SupertableOptions},
    };

    /// Superfiles the table is written as, one commit each.
    const FILES: i64 = 3;
    /// Rows in each.
    const ROWS_PER_FILE: i64 = 1000;
    /// The rows the first delete removes: every third one.
    const DELETED_EVERY: i64 = 3;
    /// Rows the second delete removes: the whole first superfile from this
    /// `seq` on, so the first superfile alone cannot fill a larger page.
    const FIRST_FILE_DELETED_FROM: i64 = 100;

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new("seq", DataType::Int64, false)]))
    }

    fn empty_table(dir: &TempDir) -> Supertable {
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let options = SupertableOptions::new(schema(), vec![], vec![])
            .expect("options")
            .with_storage(storage);
        Supertable::create(options).expect("create")
    }

    fn table(dir: &TempDir) -> Supertable {
        let st = empty_table(dir);
        for file in 0..FILES {
            let seqs: Vec<i64> = (file * ROWS_PER_FILE..(file + 1) * ROWS_PER_FILE).collect();
            let batch = RecordBatch::try_new(schema(), vec![Arc::new(Int64Array::from(seqs))])
                .expect("batch");
            let mut writer = st.writer().expect("writer");
            writer.append(&batch).expect("append");
            writer.commit().expect("commit");
        }
        st.delete((col("seq") % lit(DELETED_EVERY)).eq(lit(0_i64)))
            .expect("delete");
        st.delete(
            col("seq")
                .gt_eq(lit(FIRST_FILE_DELETED_FROM))
                .and(col("seq").lt(lit(ROWS_PER_FILE))),
        )
        .expect("delete");
        st
    }

    fn seqs(batches: &[RecordBatch]) -> Vec<i64> {
        batches
            .iter()
            .flat_map(|b| {
                let c = b
                    .column_by_name("seq")
                    .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
                    .expect("seq")
                    .clone();
                (0..c.len()).map(move |i| c.value(i))
            })
            .collect()
    }

    /// The live rows, as the table's own statement reads them.
    fn live(st: &Supertable) -> BTreeSet<i64> {
        let reader = st.reader().expect("reader");
        seqs(&reader.query_sql("SELECT seq FROM supertable").expect("sql"))
            .into_iter()
            .collect()
    }

    /// Page sizes asked for: none, a few, the first superfile's live rows
    /// and one more, every live row and past it, and sizes no table here
    /// holds, which must not reserve room for that many rows.
    fn page_sizes(live: usize) -> [usize; 11] {
        [
            0,
            1,
            10,
            66,
            67,
            500,
            live,
            live + 10,
            100_000_000,
            usize::MAX / 2,
            usize::MAX,
        ]
    }

    /// Every page is the first `k` live rows in storage order, or every live
    /// row when the table holds fewer. Written in `seq` order, one superfile
    /// per commit, the table's storage order is `seq` order, so each page is
    /// the `k` lowest live `seq` values in ascending order: deleted rows
    /// never come back, and a page past the first superfile's live rows
    /// reads on into the next. An optimize rewrites superfiles in an order
    /// of its own, so there each page is the first `k` rows of the page of
    /// every live row, which holds each live row once.
    #[test]
    fn head_returns_the_first_live_rows_in_storage_order() {
        let dir = TempDir::new().expect("tempdir");
        let st = table(&dir);
        for pass in ["appended", "optimized"] {
            if pass == "optimized" {
                st.optimize(&OptimizeOptions::default()).expect("optimize");
            }
            let live = live(&st);
            let all = seqs(&st.head(usize::MAX, Some(&["seq"])).expect("head"));
            let distinct: BTreeSet<i64> = all.iter().copied().collect();
            assert_eq!(distinct, live, "{pass}: every live row");
            assert_eq!(all.len(), live.len(), "{pass}: each live row once");
            if pass == "appended" {
                let ascending: Vec<i64> = live.iter().copied().collect();
                assert_eq!(all, ascending, "{pass}: storage order is seq order");
            }
            for k in page_sizes(live.len()) {
                let got = seqs(&st.head(k, Some(&["seq"])).expect("head"));
                assert_eq!(
                    got,
                    all[..k.min(all.len())],
                    "{pass}, k {k}: the first k live rows in storage order"
                );
            }
        }
    }

    /// A table with no rows answers every page size with no rows.
    #[test]
    fn an_empty_table_heads_to_no_rows() {
        let dir = TempDir::new().expect("tempdir");
        let st = empty_table(&dir);
        for k in page_sizes(0) {
            let got = seqs(&st.head(k, Some(&["seq"])).expect("head"));
            assert!(got.is_empty(), "k {k}: no rows");
        }
    }

    /// With no projection the rows carry `_id` and a zero `score`, as the
    /// other search calls return them; an unknown column is refused.
    #[test]
    fn a_bare_head_returns_ids_and_a_zero_score() {
        let dir = TempDir::new().expect("tempdir");
        let st = table(&dir);
        let batches = st.head(5, None).expect("head");
        let names: Vec<String> = batches[0]
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(names, ["_id", "score"]);
        let score = batches[0]
            .column_by_name("score")
            .expect("score")
            .as_any()
            .downcast_ref::<arrow_array::Float32Array>()
            .expect("f32");
        assert!(score.values().iter().all(|s| *s == 0.0));
        assert!(st.head(5, Some(&["nope"])).is_err());
    }
}
