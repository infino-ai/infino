//! BM25 fan-out on [`Supertable`](super::super::Supertable).
//!
//! ## Public API
//!
//! ```ignore
//! let hits: Vec<SuperfileHit> =
//!     supertable.bm25_search("title", "rust async", 10, BoolMode::Or)?;
//!
//! let prefix_hits: Vec<SuperfileHit> =
//!     supertable.bm25_search_prefix("title", "rus", 10)?;
//! ```
//!
//! Both methods return [`SuperfileHit`]s sorted by score *descending*
//! — higher BM25 score is more relevant. `local_doc_id` is the row
//! offset within `segment`; doc-id space is local to a segment in
//! v1.
//!
//! ## Strategy
//!
//! Internally pins a snapshot reader and drives the async
//! kernel to completion via the sync→async bridge. The reader
//! holds a pinned `Arc<Manifest>`; for each visible segment we:
//!
//!   1. Fetch the segment's `SuperfileReader` from the store.
//!   2. Delegate to `SuperfileReader::bm25_search` /
//!      `bm25_search_prefix` (already implemented at the superfile
//!      layer; per-segment top-k with BlockMaxWAND skip).
//!   3. Tag each `(local_doc_id, score)` with the segment URI.
//!   4. Concatenate across superfiles and global-top-k by score.
//!
//! Rayon fan-out runs on `options.reader_pool`. For an N-segment
//! supertable we issue N parallel per-segment searches; the pool
//! caps concurrency at the configured reader thread count.
//!
//! ## Score comparability across superfiles
//!
//! BM25's IDF is computed from per-segment `n_docs` and `df`,
//! so a rare term in a small segment can score higher than the
//! same term in a larger segment. This is the classical sharded-
//! BM25 problem:
//! treating per-segment scores as comparable is a documented
//! approximation, accepted in v1 because (a) global IDF would
//! require either a manifest-wide df table or a two-pass query
//! (df gather + score), both with non-trivial memory/latency
//! cost; (b) for k ≥ 10 and reasonably balanced superfiles the top-k
//! *set* converges to the global answer even if score *order*
//! within the set wiggles. Oracle tests assert set membership at
//! `k = 10` against a single-segment ground truth.
//!
//! Manifest-level skip pruning is wired in: each call computes a
//! per-segment keep/prune mask from the FTS bloom (exact-term
//! mode) or the lex term range (prefix mode) before issuing
//! per-segment work, so pruned superfiles never trigger a
//! `SuperfileReaderCache::reader` call. Vector + SQL skip remain
//! deferred (see those modules' headers).

use std::sync::Arc;

use crate::superfile::SuperfileReader;
pub use crate::superfile::fts::reader::BoolMode;
use crate::superfile::fts::tokenize::{AsciiLowerTokenizer, Tokenizer};
use crate::supertable::error::QueryError;
use crate::supertable::handle::{Supertable, SupertableReader};
use crate::supertable::manifest::{Manifest, SuperfileEntry};

use super::SuperfileHit;

impl SupertableReader {
    /// Single-column BM25 search across the pinned manifest's
    /// superfiles. Returns up to `k` highest-scoring hits, sorted
    /// descending by score.
    ///
    /// `query` is tokenized by the v1 [`AsciiLowerTokenizer`] —
    /// the same tokenizer used at index time. Returns
    /// [`QueryError::Store`] if any segment is unreachable, or
    /// [`QueryError::Parquet`] if a segment's bytes can't be
    /// queried (column missing from the segment's FTS index, etc.).
    ///
    /// Empty supertable (no superfiles) returns an empty `Vec`
    /// without consulting the store.
    ///
    /// `pub(crate)` async kernel — the public surface is the sync
    /// [`Supertable::bm25_search`], which drives this via the
    /// sync→async bridge after applying the read-consistency policy.
    ///
    /// [`AsciiLowerTokenizer`]: crate::superfile::fts::tokenize::AsciiLowerTokenizer
    pub(crate) async fn bm25_search(
        &self,
        column: &str,
        query: &str,
        k: usize,
        mode: BoolMode,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let manifest = self.manifest();
        let pool_threads = manifest.options.reader_pool.current_num_threads();
        let column_owned = column.to_owned();

        // Tokenize ONCE at the orchestrator. The pre-tokenized
        // term slice is reused both for the list-level + per-
        // segment bloom-skip masks AND for each per-segment
        // search via SuperfileReader::bm25_search_pretokenized —
        // eliminating the (N+1)·T redundant tokenizations a
        // per-segment bm25_search would incur for N superfiles +
        // a T-token query.
        let term_strings: Vec<String> = AsciiLowerTokenizer.tokenize(query).collect();
        let term_refs: Vec<&str> = term_strings.iter().map(|s| s.as_str()).collect();

        // Segment selection via the shared two-tier prune
        // (`query::prune::select_segments`): part-level bloom-union
        // skip → lazy-load surviving parts → per-segment bloom skip.
        // FTS exact search is the single-`TermPresence`-leaf case of
        // the same path SQL scalar filtering uses.
        let kept = crate::supertable::query::prune::select_segments(
            manifest.as_ref(),
            &[crate::supertable::query::prune::PruneLeaf::TermPresence {
                column: column_owned.clone(),
                terms: term_strings.clone(),
                mode,
            }],
        )
        .await?;
        if kept.is_empty() {
            return Ok(Vec::new());
        }

        // Build the work-unit list. When the reader pool has more
        // threads than there are kept superfiles AND we're on the
        // multi-term OR hot path, slice each segment into doc_id
        // sub-ranges so the fan-out can saturate every pool thread.
        // Single-term OR and AND stay on the un-ranged call.
        let kept_refs: Vec<&Arc<SuperfileEntry>> = kept.iter().collect();
        let work_units = build_or_work_units(&kept_refs, mode, term_refs.len(), pool_threads);
        let units: Vec<(Arc<SuperfileEntry>, Option<(u32, u32)>)> =
            work_units.into_iter().map(|u| (u.entry, u.range)).collect();

        let term_arc: Arc<Vec<String>> = Arc::new(term_strings);
        let column_arc = Arc::new(column_owned);

        // One shared fan-out (`query::dispatch::fanout`) — the same
        // orchestrator the vector path uses. It warms the tombstone
        // sidecars in one batch, opens each segment reader and runs the
        // kernel under `tokio::spawn` so cold GETs overlap, then tags +
        // tombstone-filters each unit's hits. The per-unit `params` is
        // the optional doc-id sub-range; `None` searches the whole
        // segment.
        let kernel = move |r: Arc<SuperfileReader>, range: Option<(u32, u32)>| {
            let column_arc = Arc::clone(&column_arc);
            let term_arc = Arc::clone(&term_arc);
            async move {
                let term_refs: Vec<&str> = term_arc.iter().map(|s| s.as_str()).collect();
                match range {
                    Some((start, end)) => r
                        .bm25_search_or_range_pretokenized(&column_arc, &term_refs, k, start, end)
                        .await
                        .map_err(|e| QueryError::Parquet(e.to_string())),
                    None => r
                        .bm25_search_pretokenized(&column_arc, &term_refs, k, mode)
                        .await
                        .map_err(|e| QueryError::Parquet(e.to_string())),
                }
            }
        };
        let per_unit = crate::supertable::query::dispatch::fanout(self, units, kernel).await?;

        Ok(top_k_descending(per_unit, k))
    }

    /// Prefix-expanded BM25 search across the pinned manifest's
    /// superfiles. The prefix is ASCII-lowercased before expansion
    /// (matching the v1 tokenizer) and expanded per-segment to the
    /// concrete term list before `BoolMode::Or` BM25 scoring.
    ///
    /// Returns up to `k` highest-scoring hits, sorted descending
    /// by score.
    ///
    /// Empty supertable (no superfiles) and `k == 0` short-circuit
    /// to an empty `Vec`.
    ///
    /// `pub(crate)` async kernel — the public surface is the sync
    /// [`Supertable::bm25_search_prefix`].
    pub(crate) async fn bm25_search_prefix(
        &self,
        column: &str,
        prefix: &str,
        k: usize,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let manifest = self.manifest();
        let pool_threads = manifest.options.reader_pool.current_num_threads();
        let column_owned = column.to_owned();
        let prefix_owned = prefix.to_owned();

        // Manifest-level term-range skip uses the same
        // lowercased prefix bytes the v1 tokenizer +
        // FST-expansion path use, so the skip's
        // lex-range overlap test exactly matches the
        // tokenizer's interpretation of the prefix.
        let prefix_lower = prefix_owned.to_ascii_lowercase();

        // Segment selection via the shared two-tier prune — the
        // single-`Prefix`-leaf case (part-level term-range skip →
        // lazy-load surviving parts → per-segment term-range skip).
        let kept = crate::supertable::query::prune::select_segments(
            manifest.as_ref(),
            &[crate::supertable::query::prune::PruneLeaf::Prefix {
                column: column_owned.clone(),
                prefix: prefix_lower.as_bytes().to_vec(),
            }],
        )
        .await?;
        if kept.is_empty() {
            return Ok(Vec::new());
        }

        // Same sub-range fan-out logic as `bm25_search`. `n_terms=2`
        // stands in for "multi-term OR enabled" — the prefix path
        // always runs BoolMode::Or and expansion typically yields ≥2
        // terms at the scales we care about.
        let kept_refs: Vec<&Arc<SuperfileEntry>> = kept.iter().collect();
        let work_units = build_or_work_units(&kept_refs, BoolMode::Or, 2, pool_threads);
        let units: Vec<(Arc<SuperfileEntry>, Option<(u32, u32)>)> =
            work_units.into_iter().map(|u| (u.entry, u.range)).collect();

        let column_arc = Arc::new(column_owned);
        let prefix_arc = Arc::new(prefix_owned);

        // Shared fan-out — see `bm25_search` for the rationale; the
        // kernel differs only in calling the prefix search variants.
        let kernel = move |r: Arc<SuperfileReader>, range: Option<(u32, u32)>| {
            let column_arc = Arc::clone(&column_arc);
            let prefix_arc = Arc::clone(&prefix_arc);
            async move {
                match range {
                    Some((start, end)) => r
                        .bm25_search_prefix_range(&column_arc, &prefix_arc, k, start, end)
                        .await
                        .map_err(|e| QueryError::Parquet(e.to_string())),
                    None => r
                        .bm25_search_prefix(&column_arc, &prefix_arc, k)
                        .await
                        .map_err(|e| QueryError::Parquet(e.to_string())),
                }
            }
        };
        let per_unit = crate::supertable::query::dispatch::fanout(self, units, kernel).await?;

        Ok(top_k_descending(per_unit, k))
    }
}

impl Supertable {
    /// Single-column BM25 search over the current snapshot.
    ///
    /// Pins a reader at call entry, applies the read-consistency
    /// policy, and drives the internal async kernel to completion
    /// via the sync→async bridge ([`Supertable::block_on_query`]).
    /// Returns up to `k` hits sorted by BM25 score *descending*.
    pub fn bm25_search(
        &self,
        column: &str,
        query: &str,
        k: usize,
        mode: BoolMode,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        self.ensure_fresh();
        let reader = self.reader();
        self.block_on_query(reader.bm25_search(column, query, k, mode))
    }

    /// Prefix-expanded BM25 search — see [`Supertable::bm25_search`]
    /// for the bridge semantics.
    pub fn bm25_search_prefix(
        &self,
        column: &str,
        prefix: &str,
        k: usize,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        self.ensure_fresh();
        let reader = self.reader();
        self.block_on_query(reader.bm25_search_prefix(column, prefix, k))
    }
}

/// One unit of per-segment search work scheduled into the reader
/// pool's `par_iter`. `range == None` means "the whole segment" and
/// dispatches to the un-ranged BM25 API; `range == Some((start,
/// end))` means "only doc_ids in [start, end)" and dispatches to
/// the range-aware OR path.
struct WorkUnit {
    entry: Arc<SuperfileEntry>,
    range: Option<(u32, u32)>,
}

/// Minimum docs per sub-range. Below this width, splitting adds
/// more pool-scheduling + per-shard top-K-merge overhead than it
/// saves in scoring work. Tuned to be coarse — the heuristic only
/// needs to avoid splitting toy superfiles; production superfiles at
/// the scales we benchmark (1.25M docs/segment after 10M × cpus/2
/// row-shard) are well above this floor.
const SUBRANGE_MIN_DOCS: u32 = 50_000;

/// Decide how to slice the kept superfiles into parallel work units.
/// Returns one [`WorkUnit`] per (segment, doc_id sub-range) tuple.
///
/// Fan-out happens only when:
///   1. The query is `BoolMode::Or` with two or more terms — the
///      only shape the range-aware BMM is wired up for.
///   2. The reader pool has more threads than kept superfiles —
///      otherwise every thread is already saturated by one segment
///      and splitting just adds overhead.
///   3. The candidate sub-range width is at least
///      `SUBRANGE_MIN_DOCS` — below that, BMM bookkeeping +
///      cross-sub-range top-K merge dominate the parallel win.
///
/// Otherwise each kept segment becomes a single un-ranged work unit
/// — identical to the original `par_iter` over superfiles shape.
fn build_or_work_units(
    kept: &[&Arc<SuperfileEntry>],
    mode: BoolMode,
    n_terms: usize,
    pool_threads: usize,
) -> Vec<WorkUnit> {
    let fanout_eligible = mode == BoolMode::Or && n_terms >= 2;
    let want_subranges = pool_threads.div_ceil(kept.len().max(1)).max(1);
    if !fanout_eligible || want_subranges <= 1 {
        return kept
            .iter()
            .map(|e| WorkUnit {
                entry: Arc::clone(e),
                range: None,
            })
            .collect();
    }

    let mut units: Vec<WorkUnit> = Vec::with_capacity(kept.len() * want_subranges);
    for entry in kept {
        let n_docs = entry.n_docs as u32;
        if n_docs == 0 {
            continue;
        }
        // Round the sub-range count down to avoid producing
        // narrower-than-floor slices. With `want_subranges = 2` on
        // a 1.25M-doc segment, stride = 625K (well above floor) so
        // both sub-ranges fire. With a tiny segment (e.g., 10K
        // docs, well below `SUBRANGE_MIN_DOCS`), the division
        // collapses to 1 sub-range = full segment.
        let cap_by_floor = (n_docs / SUBRANGE_MIN_DOCS).max(1) as usize;
        let n_sub = want_subranges.min(cap_by_floor);
        if n_sub <= 1 {
            units.push(WorkUnit {
                entry: Arc::clone(entry),
                range: None,
            });
            continue;
        }
        let stride = n_docs.div_ceil(n_sub as u32);
        let mut start: u32 = 0;
        while start < n_docs {
            let end = start.saturating_add(stride).min(n_docs);
            units.push(WorkUnit {
                entry: Arc::clone(entry),
                range: Some((start, end)),
            });
            start = end;
        }
    }
    units
}

/// Merge per-segment hits and return the top-k by *descending*
/// score (highest BM25 = most relevant). Uses a min-heap of size k
/// so we never sort more than k elements.
fn top_k_descending(per_segment: Vec<Vec<SuperfileHit>>, k: usize) -> Vec<SuperfileHit> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    #[derive(PartialEq)]
    struct MinByScore(SuperfileHit);
    impl Eq for MinByScore {}
    impl PartialOrd for MinByScore {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for MinByScore {
        fn cmp(&self, other: &Self) -> Ordering {
            other
                .0
                .score
                .partial_cmp(&self.0.score)
                .unwrap_or(Ordering::Equal)
        }
    }

    let mut heap = BinaryHeap::with_capacity(k + 1);
    for hit in per_segment.into_iter().flatten() {
        if heap.len() < k {
            heap.push(MinByScore(hit));
        } else if let Some(worst) = heap.peek()
            && hit.score > worst.0.score
        {
            heap.pop();
            heap.push(MinByScore(hit));
        }
    }
    let mut result: Vec<SuperfileHit> = heap.into_iter().map(|m| m.0).collect();
    result.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    result
}

/// Helper used by [`Manifest`] consumers in tests and downstream
/// skip-layer code. Exposed at module scope (not `pub`) for the
/// same reason the writer's helpers stay private: this is internal
/// plumbing, not API surface.
#[allow(dead_code)]
fn _manifest_doc_total(manifest: &Manifest) -> u64 {
    manifest.n_docs_total()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};

    use crate::superfile::builder::{FtsConfig, SuperfileBuilder};

    use crate::supertable::error::QueryError;
    use crate::supertable::{Supertable, SupertableOptions};

    use super::BoolMode;

    use crate::test_helpers::default_tokenizer as tok;

    fn schema_id_title() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new(
            "title",
            DataType::LargeUtf8,
            false,
        )]))
    }

    fn options_one_segment_per_commit() -> SupertableOptions {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        SupertableOptions::new(
            schema_id_title(),
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            Some(tok()),
        )
        .expect("valid options")
        .with_writer_pool(pool)
    }

    fn build_batch(_start: u64, titles: &[&str]) -> RecordBatch {
        let titles_arr = LargeStringArray::from(titles.to_vec());
        RecordBatch::try_new(schema_id_title(), vec![Arc::new(titles_arr)]).expect("batch")
    }

    /// Build a single SuperfileBuilder containing the same docs as
    /// the supertable across all superfiles. Used as the oracle for
    /// per-segment-vs-global BM25 set-membership tests.
    fn build_oracle_superfile(titles: &[&str]) -> Arc<crate::superfile::SuperfileReader> {
        // The oracle path goes directly through SuperfileBuilder
        // (not through Supertable::append's auto-injection), so
        // we build the effective schema by hand: `_id` is
        // `Decimal128(38, 0)`, ids are 0..n.
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            Field::new(
                "_id",
                DataType::Decimal128(
                    crate::supertable::options::DECIMAL128_PRECISION,
                    crate::supertable::options::DECIMAL128_SCALE,
                ),
                false,
            ),
            Field::new("title", DataType::LargeUtf8, false),
        ]));
        let opts = crate::superfile::builder::BuilderOptions::new(
            schema.clone(),
            "_id",
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            Some(tok()),
        );
        let mut b = SuperfileBuilder::new(opts).expect("builder");
        let n = titles.len();
        let ids = arrow_array::Decimal128Array::from((0..n as i128).collect::<Vec<_>>())
            .with_precision_and_scale(
                crate::supertable::options::DECIMAL128_PRECISION,
                crate::supertable::options::DECIMAL128_SCALE,
            )
            .expect("decimal128");
        let titles_arr = LargeStringArray::from(titles.to_vec());
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(titles_arr)]).expect("batch");
        b.add_batch(&batch, &[]).expect("add_batch");
        let bytes = bytes::Bytes::from(b.finish().expect("finish"));
        Arc::new(crate::superfile::SuperfileReader::open(bytes).expect("open"))
    }

    #[tokio::test]
    async fn bm25_search_empty_supertable_returns_empty_without_store_calls() {
        let st = Supertable::create(options_one_segment_per_commit()).expect("create");
        let r = st.reader();
        let hits = r
            .bm25_search("title", "rust", 5, BoolMode::Or)
            .await
            .expect("query");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn bm25_search_k_zero_short_circuits() {
        let st = Supertable::create(options_one_segment_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["rust async"])).expect("append");
        w.commit().expect("commit");
        let r = st.reader();
        let hits = r
            .bm25_search("title", "rust", 0, BoolMode::Or)
            .await
            .expect("query");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn bm25_search_returns_descending_score_order() {
        let st = Supertable::create(options_one_segment_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(
            0,
            &[
                "rust rust rust async",
                "rust async runtime",
                "rust embedded",
                "python data",
            ],
        ))
        .expect("append");
        w.commit().expect("commit");
        let r = st.reader();
        let hits = r
            .bm25_search("title", "rust", 4, BoolMode::Or)
            .await
            .expect("query");
        // Should return 3 hits (the python doc has no `rust`).
        assert_eq!(hits.len(), 3);
        // Strictly descending.
        for w in hits.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[tokio::test]
    async fn bm25_search_carries_segment_uri_for_each_hit() {
        let st = Supertable::create(options_one_segment_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["rust rust async"])).expect("a1");
        w.commit().expect("c1");
        w.append(&build_batch(10, &["rust runtime"])).expect("a2");
        w.commit().expect("c2");

        let r = st.reader();
        assert_eq!(r.n_superfiles(), 2);
        let hits = r
            .bm25_search("title", "rust", 5, BoolMode::Or)
            .await
            .expect("query");
        assert_eq!(hits.len(), 2);
        // Both segment URIs should appear.
        let mut uris: Vec<_> = hits.iter().map(|h| h.segment).collect();
        uris.sort();
        let expected: Vec<_> = {
            let mut v: Vec<_> = r.manifest().superfiles.iter().map(|e| e.uri).collect();
            v.sort();
            v
        };
        assert_eq!(uris, expected);
    }

    #[tokio::test]
    async fn bm25_search_oracle_top_k_set_matches_single_superfile() {
        // Plant a corpus where the top-k under BM25 is unambiguous
        // regardless of per-segment-vs-global IDF variation: 3 docs
        // contain the rare term `nimblefox`, distributed across 3
        // superfiles; the other 9 docs share only generic terms with
        // each other and with the query, so they score zero against
        // `nimblefox`. The set membership check survives even
        // though per-segment IDF for `nimblefox` differs from
        // global IDF (it's `df=1` in each segment vs `df=3` global).
        let titles = vec![
            "lookup nimblefox special token",   // 0  — match
            "ordinary common everyday text",    // 1
            "more usual filler corpus copy",    // 2
            "something boring without it",      // 3
            "mid corpus another nimblefox row", // 4  — match
            "generic page that adds nothing",   // 5
            "another stuffer no rare terms",    // 6
            "more padding here for filler",     // 7
            "tail nimblefox final segment",     // 8  — match
            "another tail row",                 // 9
            "yet another normal title",         // 10
            "wrapping up the corpus today",     // 11
        ];

        let st = Supertable::create(options_one_segment_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        for chunk_start in (0..titles.len()).step_by(4) {
            let end = (chunk_start + 4).min(titles.len());
            let chunk = &titles[chunk_start..end];
            w.append(&build_batch(chunk_start as u64, chunk))
                .expect("append");
            w.commit().expect("commit");
        }
        assert_eq!(st.reader().n_superfiles(), 3);

        let oracle = build_oracle_superfile(&titles);
        let oracle_hits = oracle
            .bm25_search("title", "nimblefox", 5, BoolMode::Or)
            .await
            .expect("oracle");
        // Oracle should find exactly 3 docs containing `nimblefox`.
        assert_eq!(oracle_hits.len(), 3);
        let oracle_set: std::collections::HashSet<u32> =
            oracle_hits.iter().map(|(d, _)| *d).collect();
        assert_eq!(oracle_set, [0u32, 4, 8].iter().copied().collect());

        let st_reader = st.reader();
        let st_hits = st_reader
            .bm25_search("title", "nimblefox", 5, BoolMode::Or)
            .await
            .expect("supertable query");
        assert_eq!(st_hits.len(), 3);
        // Resolve supertable hits to global doc-ids via segment
        // ordering (superfiles appear in append order; chunk size = 4).
        let manifest = st_reader.manifest();
        let st_globals: std::collections::HashSet<u32> = st_hits
            .iter()
            .map(|h| {
                let seg_idx = manifest
                    .superfiles
                    .iter()
                    .position(|e| e.uri == h.segment)
                    .expect("segment in manifest");
                (seg_idx as u32) * 4 + h.local_doc_id
            })
            .collect();
        assert_eq!(st_globals, oracle_set);
    }

    #[tokio::test]
    async fn bm25_search_prefix_oracle_top_k_set_matches_single_superfile() {
        let titles = vec![
            "rust async runtime",
            "rust embedded systems",
            "ruby gemfile config",
            "rustacean conference",
            "python machine learning",
            "python web framework",
            "rusty pipe rebuild",
            "go concurrency model",
        ];
        let st = Supertable::create(options_one_segment_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        for chunk_start in (0..titles.len()).step_by(2) {
            let end = (chunk_start + 2).min(titles.len());
            let chunk = &titles[chunk_start..end];
            w.append(&build_batch(chunk_start as u64, chunk))
                .expect("append");
            w.commit().expect("commit");
        }

        let oracle = build_oracle_superfile(&titles);
        let oracle_hits = oracle
            .bm25_search_prefix("title", "rust", 5)
            .await
            .expect("oracle");
        let oracle_globals: std::collections::HashSet<u32> =
            oracle_hits.iter().map(|(d, _)| *d).collect();

        let st_reader = st.reader();
        let st_hits = st_reader
            .bm25_search_prefix("title", "rust", 5)
            .await
            .expect("supertable query");
        let manifest = st_reader.manifest();
        let st_globals: std::collections::HashSet<u32> = st_hits
            .iter()
            .map(|h| {
                let seg_idx = manifest
                    .superfiles
                    .iter()
                    .position(|e| e.uri == h.segment)
                    .expect("segment in manifest");
                (seg_idx as u32) * 2 + h.local_doc_id
            })
            .collect();
        assert_eq!(st_hits.len(), oracle_hits.len());
        assert_eq!(st_globals, oracle_globals);
        // Prefix-expansion sanity: we should hit "rust*" and
        // "rusty*" / "rustacean*" but not "ruby*".
        assert!(st_hits.len() >= 4);
    }

    #[tokio::test]
    async fn bm25_search_prefix_unmatched_prefix_returns_empty() {
        let st = Supertable::create(options_one_segment_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["rust async"])).expect("append");
        w.commit().expect("commit");

        let r = st.reader();
        let hits = r
            .bm25_search_prefix("title", "zzzz", 10)
            .await
            .expect("query");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn bm25_search_prefix_lowercases_input() {
        // Index stores tokenized terms (lowercased); user provides
        // mixed-case prefix; we lowercase before expansion so the
        // FST walk finds the matching subtree.
        let st = Supertable::create(options_one_segment_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["Rust async runtime"]))
            .expect("append");
        w.commit().expect("commit");

        let r = st.reader();
        let hits = r
            .bm25_search_prefix("title", "RUST", 5)
            .await
            .expect("query");
        assert_eq!(hits.len(), 1);
    }

    #[tokio::test]
    async fn bm25_search_unknown_column_errors() {
        let st = Supertable::create(options_one_segment_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["rust"])).expect("append");
        w.commit().expect("commit");

        let r = st.reader();
        let err = r
            .bm25_search("missing_column", "rust", 5, BoolMode::Or)
            .await
            .expect_err("expected error");
        assert!(matches!(err, QueryError::Parquet(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn bm25_search_results_global_top_k_caps_at_k() {
        // 4 superfiles × 1 doc each = 4 hits; ask for k=2; expect 2.
        let st = Supertable::create(options_one_segment_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        for i in 0..4 {
            w.append(&build_batch(i * 10, &["rust async runtime"]))
                .expect("a");
            w.commit().expect("c");
        }
        let r = st.reader();
        let hits = r
            .bm25_search("title", "rust", 2, BoolMode::Or)
            .await
            .expect("query");
        assert_eq!(hits.len(), 2);
    }
}
