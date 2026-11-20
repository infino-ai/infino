// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! BM25 over several full-text columns at once, scored *best fields*.
//!
//! A query is a list of [`FieldClause`]s, each a query text and the columns
//! it searches. A row's score for one clause is the best of its columns:
//! the largest `weight × BM25(column, text)` over the clause's columns that
//! match. Its score for the query is the sum of its clause scores, over the
//! clauses it matches. Under [`BoolMode::Or`] a row that matches any clause
//! is a hit; under [`BoolMode::And`] it must match every clause. The mode
//! also applies inside each clause, to its text's bare terms, exactly as it
//! does for [`Supertable::bm25_search`](crate::Supertable::bm25_search).
//!
//! That is the score a `dis_max` per clause with a tie breaker of 0, summed
//! across clauses, gives: what a multi-field `query_string` or
//! `multi_match` of type `best_fields` ranks by. Summing every matching
//! column instead (*most fields*) is a different ranking, and not this one.
//!
//! ## How the top k is found
//!
//! Each (clause, column) pair is one ordinary single-column search, so each
//! column keeps its own analyzer, the table-wide statistics every BM25
//! search scores with, and the scope a `WHERE` lowers to. The pairs are
//! combined by a threshold algorithm:
//!
//! 1. Each pair returns its `w` best rows (`w` starts at `k`).
//! 2. Every row any pair returned is a candidate; a candidate one pair did
//!    not return is scored on that pair by a search confined to the
//!    candidates, so every candidate's combined score is exact.
//! 3. A row no pair returned scores at most the pair's last score on each
//!    pair (0 on a pair that returned every row it matches), so its
//!    combined score is at most the sum, over clauses, of the clause's
//!    largest such bound. Once the k-th best candidate scores strictly
//!    above that bound, no unseen row can enter the top k or tie into it.
//! 4. Otherwise `w` grows and the rounds repeat. They also stop once the
//!    pairs have returned every row they match (every pair under `Or`, every
//!    pair of some clause under `And`), or `w` reaches the table's size.
//!
//! The answer is the exact top k, in score order with ties broken by
//! ascending `_id`, as `bm25_search` breaks them.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::Arc,
};

use arrow::record_batch::RecordBatch;
use futures::future::try_join_all;
use roaring::RoaringBitmap;
use serde::Deserialize;
use tracing::{Instrument, debug};

#[cfg(feature = "remote")]
use crate::supertable::query::{exec::common::SCORE_COLUMN, sorted::quote};
use crate::{
    InfinoError,
    superfile::fts::reader::{Bm25SearchOptions, BoolMode},
    supertable::{
        error::QueryError,
        handle::{Supertable, SupertableReader},
        manifest::{SuperfileEntry, SuperfileUri},
        query::{
            SuperfileHit,
            candidate::{CandidatePlan, CandidateScope},
            exec::common::{OVER_FETCH_GROWTH, resolve_hits_named},
            fts::select_top_k_stable,
        },
        reader_cache::disk::ForegroundQueryGuard,
    },
    utils::trace::detail_span,
};

/// One clause of a multi-field BM25 query: a query text and the full-text
/// columns it searches. The text is read as
/// [`Supertable::bm25_search`](crate::Supertable::bm25_search) reads it
/// (bare terms under the query's mode, `+must`, `-must_not`, quoted
/// phrases), and tokenized by each column's own analyzer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldClause {
    query: String,
    fields: Vec<String>,
}

impl FieldClause {
    /// A clause searching `query` over `fields`.
    pub fn new<S: Into<String>>(
        query: impl Into<String>,
        fields: impl IntoIterator<Item = S>,
    ) -> Self {
        Self {
            query: query.into(),
            fields: fields.into_iter().map(Into::into).collect(),
        }
    }

    /// The clause's query text.
    pub fn query(&self) -> &str {
        &self.query
    }

    /// The columns the clause searches.
    pub fn fields(&self) -> &[String] {
        &self.fields
    }
}

/// Options for a multi-field BM25 search: the boolean `mode` and per-column
/// weights. [`Default`] is [`BoolMode::Or`] with every column weighted 1.
///
/// `#[non_exhaustive]`: construct with [`Bm25FieldsOptions::new`] and the
/// `with_*` setters.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct Bm25FieldsOptions {
    /// Under `Or` a row matching any clause is a hit; under `And` it must
    /// match every clause. Also the mode of each clause's bare terms.
    pub mode: BoolMode,
    /// What each named column's BM25 score is multiplied by before the
    /// clause takes its best column. A column not named weighs 1, and a
    /// weight on a column no clause searches has no effect, so one map can
    /// serve many queries. Every weight must be finite and greater than 0.
    pub weights: BTreeMap<String, f32>,
}

impl Bm25FieldsOptions {
    /// Default options: `Or` mode, every column weighted 1.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the boolean mode.
    pub fn with_mode(mut self, mode: BoolMode) -> Self {
        self.mode = mode;
        self
    }

    /// Weight `column`'s scores by `weight`.
    pub fn with_weight(mut self, column: impl Into<String>, weight: f32) -> Self {
        self.weights.insert(column.into(), weight);
        self
    }
}

/// One clause as JSON: `{"query": "...", "fields": ["a", "b"]}`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ClauseJson {
    query: String,
    fields: Vec<String>,
}

/// Parse the clauses of the SQL form: a JSON array of
/// `{"query": "...", "fields": ["column", ...]}` objects.
pub(crate) fn clauses_from_json(json: &str) -> Result<Vec<FieldClause>, QueryError> {
    let parsed: Vec<ClauseJson> = serde_json::from_str(json).map_err(|e| {
        QueryError::InvalidQuery(format!(
            "clauses must be a JSON array of {{\"query\": string, \"fields\": [string]}} \
             objects: {e}"
        ))
    })?;
    Ok(parsed
        .into_iter()
        .map(|c| FieldClause::new(c.query, c.fields))
        .collect())
}

/// Parse the weights of the SQL form: a JSON object of column to number.
pub(crate) fn weights_from_json(json: &str) -> Result<BTreeMap<String, f32>, QueryError> {
    serde_json::from_str(json).map_err(|e| {
        QueryError::InvalidQuery(format!(
            "weights must be a JSON object of column name to number: {e}"
        ))
    })
}

/// The catalog SQL statement that runs a multi-field search on `table`:
/// how a hosted table carries the call, as the `bm25_search_fields` TVF.
/// `projection` defaults to `_id` (`id_column`) + `score`; rows come back
/// best first, ties by ascending `_id`.
#[cfg(feature = "remote")]
pub(crate) fn fields_statement(
    table: &str,
    clauses: &[FieldClause],
    k: usize,
    opts: &Bm25FieldsOptions,
    projection: Option<&[&str]>,
    id_column: &str,
) -> String {
    let literal = |text: &str| format!("'{}'", text.replace('\'', "''"));
    let clauses_json = serde_json::Value::Array(
        clauses
            .iter()
            .map(|c| serde_json::json!({ "query": c.query, "fields": c.fields }))
            .collect(),
    )
    .to_string();
    let mode = match opts.mode {
        BoolMode::And => "and",
        BoolMode::Or => "or",
    };
    let mut args = vec![
        literal(table),
        literal(&clauses_json),
        k.to_string(),
        literal(mode),
    ];
    if !opts.weights.is_empty() {
        let weights = serde_json::to_string(&opts.weights).expect("a map of finite numbers");
        args.push(literal(&weights));
    }
    let default = [id_column, SCORE_COLUMN];
    let select: Vec<String> = projection
        .unwrap_or(&default)
        .iter()
        .map(|name| quote(name))
        .collect();
    format!(
        "SELECT {} FROM bm25_search_fields({}) ORDER BY {} DESC, {} ASC",
        select.join(", "),
        args.join(", "),
        quote(SCORE_COLUMN),
        quote(id_column),
    )
}

/// One (clause, column) search: the unit the threshold algorithm reads.
#[derive(Debug, Clone)]
struct Pair {
    clause: usize,
    column: String,
    weight: f32,
}

/// A validated multi-field query: its clauses' texts and the (clause,
/// column) pairs they search, each column at most once per clause.
#[derive(Debug, Clone)]
pub(crate) struct FieldsQuery {
    texts: Vec<String>,
    pairs: Vec<Pair>,
    mode: BoolMode,
}

impl FieldsQuery {
    /// Validate `clauses` under `opts`: at least one clause, every clause
    /// with at least one column, every weight finite and above 0. A column a
    /// clause names twice is searched once. Whether each column has a
    /// full-text index is checked by the searches themselves.
    pub(crate) fn new(
        clauses: &[FieldClause],
        opts: &Bm25FieldsOptions,
    ) -> Result<Self, QueryError> {
        if clauses.is_empty() {
            return Err(QueryError::InvalidQuery(
                "bm25_search_fields needs at least one clause".to_string(),
            ));
        }
        for (column, weight) in &opts.weights {
            if !(weight.is_finite() && *weight > 0.0) {
                return Err(QueryError::InvalidQuery(format!(
                    "bm25_search_fields weight for `{column}` must be finite and greater \
                     than 0, got {weight}"
                )));
            }
        }
        let mut texts = Vec::with_capacity(clauses.len());
        let mut pairs = Vec::new();
        for (index, clause) in clauses.iter().enumerate() {
            if clause.fields.is_empty() {
                return Err(QueryError::InvalidQuery(format!(
                    "bm25_search_fields clause {index} names no column"
                )));
            }
            let mut seen = HashSet::new();
            for column in &clause.fields {
                if !seen.insert(column.as_str()) {
                    continue;
                }
                pairs.push(Pair {
                    clause: index,
                    column: column.clone(),
                    weight: opts.weights.get(column).copied().unwrap_or(1.0),
                });
            }
            texts.push(clause.query.clone());
        }
        Ok(Self {
            texts,
            pairs,
            mode: opts.mode,
        })
    }

    /// The query's mode.
    pub(crate) fn mode(&self) -> BoolMode {
        self.mode
    }

    /// Each distinct (column, text) the query searches, for the callers that
    /// widen a `WHERE` to the rows the query matches.
    pub(crate) fn searches(&self) -> impl Iterator<Item = (&str, &str)> {
        self.pairs
            .iter()
            .map(|p| (p.column.as_str(), self.texts[p.clause].as_str()))
    }

    fn opts(&self) -> Bm25SearchOptions {
        Bm25SearchOptions::new().with_mode(self.mode)
    }
}

impl SupertableReader {
    /// Multi-field BM25 search over this reader's pinned snapshot, scored
    /// best fields (see the [module docs](self)), materialized as Arrow
    /// rows.
    ///
    /// Returns up to `k` rows, best first, ties by ascending `_id`.
    /// `projection` names the columns to return, as for
    /// [`SupertableReader::bm25_search`]; `None` returns `_id` + `score`.
    pub fn bm25_search_fields(
        &self,
        clauses: &[FieldClause],
        k: usize,
        opts: &Bm25FieldsOptions,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, QueryError> {
        self.check_projection(projection)?;
        let query = FieldsQuery::new(clauses, opts)?;
        let _foreground = ForegroundQueryGuard::enter();
        self.block_on(async {
            let hits = self
                .bm25_search_fields_scoped_async(&query, k, None)
                .await?;
            let batch = resolve_hits_named(self, &hits, projection)
                .instrument(detail_span!("search.resolve", hits = hits.len()))
                .await?;
            Ok(vec![batch])
        })
    }
}

impl Supertable {
    /// BM25 over several full-text columns at once, scored *best fields*:
    /// each [`FieldClause`] scores a row by its best column (the largest
    /// `weight × BM25(column, clause text)` among the clause's columns), and
    /// the row's score is the sum over the clauses it matches. Under
    /// [`BoolMode::Or`] a row matching any clause is a hit; under
    /// [`BoolMode::And`] it must match every clause. The mode is also the
    /// mode of each clause's bare terms, as in [`Supertable::bm25_search`].
    ///
    /// This is the ranking of a `dis_max` per clause with a tie breaker of
    /// 0, summed across clauses: a multi-field `query_string` or
    /// `multi_match` of type `best_fields`. Each column is tokenized by its
    /// own analyzer and scored with the table-wide statistics, so the
    /// result is the exact top `k` of that score, ties by ascending `_id`.
    ///
    /// Returns rows like [`Supertable::bm25_search`]: `projection` names
    /// the columns to return (`_id`, scalar columns, `score`); `None`
    /// returns `_id` + `score`.
    ///
    /// ```no_run
    /// # use infino::{Bm25FieldsOptions, BoolMode, FieldClause, Supertable};
    /// # fn demo(posts: &Supertable) -> Result<(), Box<dyn std::error::Error>> {
    /// let clauses = [
    ///     FieldClause::new("climate policy", ["title", "body"]),
    ///     FieldClause::new("europe", ["tags"]),
    /// ];
    /// let opts = Bm25FieldsOptions::new().with_weight("title", 2.0);
    /// let hits = posts.bm25_search_fields(&clauses, 10, &opts, None)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn bm25_search_fields(
        &self,
        clauses: &[FieldClause],
        k: usize,
        opts: &Bm25FieldsOptions,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        debug!(clauses = clauses.len(), k, mode = ?opts.mode, "bm25_search_fields");
        let close_out = self.close_out();
        let batches = self
            .reader()?
            .bm25_search_fields(clauses, k, opts, projection)
            .map_err(InfinoError::from)
            .map_err(|e| e.with_context("bm25_search_fields", None))?;
        close_out.finish_batches(&batches);
        Ok(batches)
    }
}

/// A row some pair returned, with its weighted score on each pair (`None`
/// where the pair does not match it).
struct Candidate {
    hit: SuperfileHit,
    scores: Vec<Option<f32>>,
}

/// A candidate's key: the superfile it lives in and its row there.
type RowKey = (SuperfileUri, u32);

impl SupertableReader {
    /// The `k` best rows for `query` within `scope` (`None` searches the
    /// whole table), scored best fields: see the [module docs](self).
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(clauses = query.texts.len(), pairs = query.pairs.len(), k = k, mode = ?query.mode, role = self.role().as_str(), origin = crate::utils::trace::OpOrigin::Query.as_str()))
    )]
    pub(crate) async fn bm25_search_fields_scoped_async(
        &self,
        query: &FieldsQuery,
        k: usize,
        scope: Option<&CandidateScope>,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        if k == 0 || query.pairs.is_empty() {
            return Ok(Vec::new());
        }
        let total = usize::try_from(self.manifest().n_docs_total()).unwrap_or(usize::MAX);
        let opts = query.opts();
        // The superfiles a candidate can live in, by URI: what a search
        // confined to the candidates is scoped to, and each superfile's
        // smallest `_id` for the final tie order. Read once, on first need.
        let mut entries: Option<HashMap<SuperfileUri, Arc<SuperfileEntry>>> = None;
        // Each pair's every match, once a round has returned them: later
        // rounds reuse them rather than search the pair again.
        let mut complete_lists: Vec<Option<Vec<SuperfileHit>>> = vec![None; query.pairs.len()];
        let mut want = k;
        loop {
            let fetched = try_join_all(
                query
                    .pairs
                    .iter()
                    .enumerate()
                    .filter(|(p, _)| complete_lists[*p].is_none())
                    .map(|(p, pair)| async move {
                        let hits = self
                            .bm25_search_scoped_async(
                                &pair.column,
                                &query.texts[pair.clause],
                                want,
                                opts,
                                scope,
                            )
                            .await?;
                        Ok::<_, QueryError>((p, hits))
                    }),
            )
            .await?;
            // A pair that returned fewer rows than asked returned every row
            // it matches: the kernels admit only live rows within the scope
            // into their heaps. One asked for the whole table has too.
            let mut exhausted: Vec<bool> = complete_lists.iter().map(Option::is_some).collect();
            let mut lists: Vec<Vec<SuperfileHit>> = complete_lists
                .iter_mut()
                .map(|list| list.take().unwrap_or_default())
                .collect();
            for (p, hits) in fetched {
                exhausted[p] = hits.len() < want || want >= total;
                lists[p] = hits;
            }
            let mut candidates: HashMap<RowKey, Candidate> = HashMap::new();
            for (p, hits) in lists.iter().enumerate() {
                let weight = query.pairs[p].weight;
                for hit in hits {
                    let candidate = candidates
                        .entry((hit.superfile, hit.local_doc_id))
                        .or_insert_with(|| Candidate {
                            hit: *hit,
                            scores: vec![None; query.pairs.len()],
                        });
                    candidate.scores[p] = Some(weight * hit.score);
                    if candidate.hit.stable_id.is_none() {
                        candidate.hit.stable_id = hit.stable_id;
                    }
                }
            }
            if entries.is_none() {
                entries = Some(self.scope_entries(scope).await?);
            }
            let entries = entries.as_ref().expect("entries read above");
            self.score_missing(query, &lists, &exhausted, &mut candidates, entries)
                .await?;

            let clauses = query.texts.len();
            let mut scored: Vec<SuperfileHit> = candidates
                .into_values()
                .filter_map(|candidate| combined(query, clauses, candidate))
                .collect();

            // The most a row no pair returned can score.
            let mut bound_per_clause = vec![0.0f32; clauses];
            for (p, hits) in lists.iter().enumerate() {
                let pair = &query.pairs[p];
                let theta = match (exhausted[p], hits.last()) {
                    (false, Some(last)) => pair.weight * last.score,
                    _ => 0.0,
                };
                let bound = &mut bound_per_clause[pair.clause];
                *bound = bound.max(theta);
            }
            let bound: f32 = bound_per_clause.iter().sum();

            let complete = match query.mode {
                BoolMode::Or => exhausted.iter().all(|&e| e),
                BoolMode::And => (0..clauses).any(|c| {
                    query
                        .pairs
                        .iter()
                        .zip(&exhausted)
                        .filter(|(pair, _)| pair.clause == c)
                        .all(|(_, &e)| e)
                }),
            };
            let settled = scored.len() >= k && {
                scored.select_nth_unstable_by(k - 1, |a, b| b.score.total_cmp(&a.score));
                scored[k - 1].score > bound
            };
            if complete || settled || want >= total {
                let id_mins: HashMap<SuperfileUri, i128> = entries
                    .iter()
                    .map(|(uri, entry)| (*uri, entry.id_min))
                    .collect();
                return select_top_k_stable(self, vec![scored], k, &id_mins).await;
            }
            for (p, list) in lists.into_iter().enumerate() {
                if exhausted[p] {
                    complete_lists[p] = Some(list);
                }
            }
            want = want.saturating_mul(OVER_FETCH_GROWTH).min(total.max(k));
        }
    }

    /// Every superfile `scope` admits (the whole table's under `None`), by
    /// URI.
    async fn scope_entries(
        &self,
        scope: Option<&CandidateScope>,
    ) -> Result<HashMap<SuperfileUri, Arc<SuperfileEntry>>, QueryError> {
        let whole;
        let scope = match scope {
            Some(scope) => scope,
            None => {
                whole = self.candidate_scope(&[], &CandidatePlan::Unbounded).await?;
                &whole
            }
        };
        Ok(scope
            .admitted_superfiles()
            .map(|entry| (entry.uri, Arc::clone(entry)))
            .collect())
    }

    /// Score every candidate on each pair that did not return it, by a
    /// search of the pair confined to those candidates. A pair that returned
    /// every row it matches needs none: a candidate it left out does not
    /// match it.
    async fn score_missing(
        &self,
        query: &FieldsQuery,
        lists: &[Vec<SuperfileHit>],
        exhausted: &[bool],
        candidates: &mut HashMap<RowKey, Candidate>,
        entries: &HashMap<SuperfileUri, Arc<SuperfileEntry>>,
    ) -> Result<(), QueryError> {
        let opts = query.opts();
        let mut searches = Vec::new();
        for (p, pair) in query.pairs.iter().enumerate() {
            if exhausted[p] {
                continue;
            }
            let returned: HashSet<RowKey> = lists[p]
                .iter()
                .map(|hit| (hit.superfile, hit.local_doc_id))
                .collect();
            let mut allow: HashMap<SuperfileUri, RoaringBitmap> = HashMap::new();
            let mut missing = 0usize;
            for key in candidates.keys() {
                if !returned.contains(key) {
                    allow.entry(key.0).or_default().insert(key.1);
                    missing += 1;
                }
            }
            if missing == 0 {
                continue;
            }
            let superfiles: Vec<Arc<SuperfileEntry>> = allow
                .keys()
                .map(|uri| {
                    entries.get(uri).cloned().ok_or_else(|| {
                        QueryError::Internal(format!(
                            "bm25_search_fields: a hit names superfile {uri:?}, which the \
                             snapshot does not hold"
                        ))
                    })
                })
                .collect::<Result<_, _>>()?;
            let scope = CandidateScope {
                superfiles,
                allow: Some(
                    allow
                        .into_iter()
                        .map(|(uri, rows)| (uri, Arc::new(rows)))
                        .collect(),
                ),
            };
            searches.push(async move {
                let hits = self
                    .bm25_search_scoped_async(
                        &pair.column,
                        &query.texts[pair.clause],
                        missing,
                        opts,
                        Some(&scope),
                    )
                    .await?;
                Ok::<_, QueryError>((p, hits))
            });
        }
        for (p, hits) in try_join_all(searches).await? {
            let weight = query.pairs[p].weight;
            for hit in hits {
                if let Some(candidate) = candidates.get_mut(&(hit.superfile, hit.local_doc_id)) {
                    candidate.scores[p] = Some(weight * hit.score);
                }
            }
        }
        Ok(())
    }
}

/// `candidate` with its combined score: the sum, in clause order, of each
/// matched clause's best weighted column score. `None` when it is no hit:
/// under `And`, a clause it does not match.
fn combined(query: &FieldsQuery, clauses: usize, candidate: Candidate) -> Option<SuperfileHit> {
    let mut best: Vec<Option<f32>> = vec![None; clauses];
    for (pair, score) in query.pairs.iter().zip(&candidate.scores) {
        if let Some(score) = score {
            let slot = &mut best[pair.clause];
            *slot = Some(slot.map_or(*score, |b| b.max(*score)));
        }
    }
    if query.mode == BoolMode::And && best.iter().any(Option::is_none) {
        return None;
    }
    let mut score = 0.0f32;
    let mut matched = false;
    for clause_score in best.into_iter().flatten() {
        score += clause_score;
        matched = true;
    }
    matched.then_some(SuperfileHit {
        score,
        ..candidate.hit
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{
        collections::{BTreeSet, HashMap},
        ops::Deref,
    };

    use arrow_array::{Array, Decimal128Array, Float32Array, Int64Array, LargeStringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::{col, lit};
    use tempfile::TempDir;

    use super::*;
    use crate::{
        Stemmer,
        storage::{LocalFsStorageProvider, StorageProvider},
        superfile::builder::FtsConfig,
        supertable::{Supertable, SupertableOptions, query::exec::common::SCORE_COLUMN},
    };

    /// `title` (FTS, English stemmer), `body` and `tags` (FTS, no stemmer),
    /// `bucket` (Int64, no index).
    pub(crate) fn corpus_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("body", DataType::LargeUtf8, false),
            Field::new("tags", DataType::LargeUtf8, false),
            Field::new("bucket", DataType::Int64, false),
        ]))
    }

    /// A corpus table and the directory its storage lives in, which lasts
    /// as long as the table: deletes need durable storage.
    pub(crate) struct Corpus {
        table: Supertable,
        _dir: TempDir,
    }

    impl Deref for Corpus {
        type Target = Supertable;

        fn deref(&self) -> &Supertable {
            &self.table
        }
    }

    /// The corpus schema's indexes, every text column with positions for
    /// phrases, stored under `dir`.
    fn corpus_options(dir: &TempDir) -> SupertableOptions {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        SupertableOptions::new(
            corpus_schema(),
            vec![
                FtsConfig::new("title")
                    .stemmer(Stemmer::English)
                    .positions(true),
                FtsConfig::new("body").positions(true),
                FtsConfig::new("tags").positions(true),
            ],
            vec![],
        )
        .expect("valid options")
        .with_writer_pool(pool)
        .with_storage(
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"))
                as Arc<dyn StorageProvider>,
        )
    }

    /// One corpus row: (title, body, tags, bucket).
    pub(crate) type Row = (String, String, String, i64);

    /// `n` deterministic rows over small vocabularies, so every term
    /// matches many rows on some columns and none on others, plus runs of
    /// identical rows (exact score ties across superfiles).
    pub(crate) fn corpus_rows(n: usize) -> Vec<Row> {
        const TITLE: &[&str] = &[
            "rust", "async", "run", "runs", "running", "cloud", "java", "python", "data", "systems",
        ];
        const BODY: &[&str] = &[
            "rust", "cloud", "storage", "async", "java", "query", "engine", "fast", "slow", "run",
            "the", "a",
        ];
        const TAGS: &[&str] = &["cloud", "rust", "db", "ml", "web"];
        let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = move |bound: usize| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            usize::try_from(state >> 33).expect("31 bits") % bound
        };
        let words = |vocab: &[&str], max_len: usize, next: &mut dyn FnMut(usize) -> usize| {
            let len = 1 + next(max_len);
            (0..len)
                .map(|_| vocab[next(vocab.len())])
                .collect::<Vec<_>>()
                .join(" ")
        };
        (0..n)
            .map(|i| {
                if i % 7 == 3 {
                    // A run of identical rows: ties on every query.
                    return (
                        "rust cloud".to_string(),
                        "rust storage engine".to_string(),
                        "db".to_string(),
                        i64::try_from(i % 4).expect("small"),
                    );
                }
                let title = words(TITLE, 4, &mut next);
                let body = words(BODY, 12, &mut next);
                let tags = words(TAGS, 2, &mut next);
                let bucket = i64::try_from(next(4)).expect("small");
                (title, body, tags, bucket)
            })
            .collect()
    }

    fn batch_of(rows: &[Row]) -> RecordBatch {
        let column = |f: fn(&Row) -> &str| {
            Arc::new(LargeStringArray::from(
                rows.iter().map(f).collect::<Vec<_>>(),
            )) as Arc<dyn Array>
        };
        RecordBatch::try_new(
            corpus_schema(),
            vec![
                column(|r| r.0.as_str()),
                column(|r| r.1.as_str()),
                column(|r| r.2.as_str()),
                Arc::new(Int64Array::from(
                    rows.iter().map(|r| r.3).collect::<Vec<_>>(),
                )),
            ],
        )
        .expect("batch")
    }

    /// A table of `rows`, committed `chunk` rows at a time: one superfile
    /// per commit.
    pub(crate) fn corpus_table(rows: &[Row], chunk: usize) -> Corpus {
        let dir = TempDir::new().expect("tempdir");
        let st = Supertable::create(corpus_options(&dir)).expect("create");
        let mut w = st.writer().expect("writer");
        for part in rows.chunks(chunk) {
            w.append(&batch_of(part)).expect("append");
            w.commit().expect("commit");
        }
        drop(w);
        Corpus {
            table: st,
            _dir: dir,
        }
    }

    /// The default multi-file corpus: 240 rows over six superfiles, the
    /// `tags = 'ml'` rows deleted.
    pub(crate) fn multi_file_corpus() -> Corpus {
        let st = corpus_table(&corpus_rows(240), 40);
        st.delete(col("tags").eq(lit("ml"))).expect("delete");
        assert!(
            st.reader().expect("reader").n_superfiles() > 1,
            "the corpus spans several superfiles"
        );
        st
    }

    /// A row's `_id`: a `Decimal128(38, 0)`.
    pub(crate) type Id = i128;

    /// The values of an `_id` column.
    pub(crate) fn row_ids(column: &dyn Array) -> Vec<Id> {
        let ids = column
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("_id is Decimal128");
        (0..ids.len()).map(|i| ids.value(i)).collect()
    }

    /// `(_id, score)` per row of `batches`, in order.
    pub(crate) fn ids_scores(batches: &[RecordBatch]) -> Vec<(Id, f32)> {
        let mut out = Vec::new();
        for b in batches {
            let ids = row_ids(b.column(b.schema().index_of("_id").expect("_id")));
            let scores = b
                .column(b.schema().index_of(SCORE_COLUMN).expect("score"))
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("f32")
                .clone();
            for (i, id) in ids.iter().enumerate() {
                out.push((*id, scores.value(i)));
            }
        }
        out
    }

    /// The brute-force answer: every row each (clause, column) search
    /// matches, scored on every column it matches, combined as the module
    /// docs define, ranked by score then ascending `_id`.
    pub(crate) fn oracle(
        st: &Supertable,
        clauses: &[FieldClause],
        k: usize,
        opts: &Bm25FieldsOptions,
        live: Option<&dyn Fn(Id) -> bool>,
    ) -> Vec<(Id, f32)> {
        let all = 1_000;
        let search = Bm25SearchOptions::new().with_mode(opts.mode);
        let per_clause: Vec<HashMap<Id, f32>> = clauses
            .iter()
            .map(|clause| {
                let mut best: HashMap<Id, f32> = HashMap::new();
                let columns: BTreeSet<&str> = clause.fields().iter().map(String::as_str).collect();
                for column in columns {
                    let weight = opts.weights.get(column).copied().unwrap_or(1.0);
                    let rows = st
                        .bm25_search(column, clause.query(), all, search, Some(&["_id", "score"]))
                        .expect("single-column search");
                    let rows = ids_scores(&rows);
                    assert!(rows.len() < all, "the oracle sees every match");
                    for (id, score) in rows {
                        let weighted = weight * score;
                        best.entry(id)
                            .and_modify(|b| *b = b.max(weighted))
                            .or_insert(weighted);
                    }
                }
                best
            })
            .collect();
        let ids: BTreeSet<Id> = per_clause.iter().flat_map(|m| m.keys().copied()).collect();
        let mut out: Vec<(Id, f32)> = ids
            .into_iter()
            .filter(|id| live.is_none_or(|keep| keep(*id)))
            .filter(|id| opts.mode == BoolMode::Or || per_clause.iter().all(|m| m.contains_key(id)))
            .map(|id| {
                let mut score = 0.0f32;
                for m in &per_clause {
                    if let Some(s) = m.get(&id) {
                        score += s;
                    }
                }
                (id, score)
            })
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out.truncate(k);
        out
    }

    fn fields(
        st: &Supertable,
        clauses: &[FieldClause],
        k: usize,
        opts: &Bm25FieldsOptions,
    ) -> Vec<(Id, f32)> {
        ids_scores(
            &st.bm25_search_fields(clauses, k, opts, None)
                .expect("bm25_search_fields"),
        )
    }

    /// The queries the equivalence tests run: one clause over two columns,
    /// overlapping columns across clauses, a clause no row matches, a
    /// column named twice, operators, and a stemmed term.
    pub(crate) fn query_cases() -> Vec<Vec<FieldClause>> {
        vec![
            vec![FieldClause::new("rust async", ["title", "body"])],
            vec![
                FieldClause::new("rust", ["title", "body"]),
                FieldClause::new("cloud", ["body", "tags"]),
            ],
            vec![
                FieldClause::new("java", ["title", "body", "tags"]),
                FieldClause::new("db web", ["tags"]),
                FieldClause::new("zebra", ["title", "body"]),
            ],
            vec![
                FieldClause::new("query engine", ["body", "body", "title"]),
                FieldClause::new("python", ["title"]),
            ],
            vec![
                FieldClause::new("+rust -java", ["title", "body"]),
                FieldClause::new("\"storage engine\"", ["body"]),
            ],
            vec![FieldClause::new("running", ["title", "body"])],
            vec![
                FieldClause::new("the a", ["body"]),
                FieldClause::new("rust", ["tags", "title"]),
            ],
        ]
    }

    fn assert_matches_oracle(
        st: &Supertable,
        clauses: &[FieldClause],
        k: usize,
        opts: &Bm25FieldsOptions,
    ) {
        let got = fields(st, clauses, k, opts);
        let want = oracle(st, clauses, k, opts, None);
        assert_eq!(
            got, want,
            "bm25_search_fields vs brute force: clauses {clauses:?}, k {k}, opts {opts:?}"
        );
    }

    /// Every case, under both modes, with and without weights, at every
    /// `k` from 1 past the match count, equals the brute-force ranking:
    /// same rows, same order (ties by `_id`), same scores.
    #[test]
    fn matches_brute_force_on_a_multi_file_table() {
        let st = multi_file_corpus();
        let weighted = Bm25FieldsOptions::new()
            .with_weight("title", 2.5)
            .with_weight("tags", 0.5);
        for clauses in query_cases() {
            for mode in [BoolMode::Or, BoolMode::And] {
                for base in [Bm25FieldsOptions::new(), weighted.clone()] {
                    let opts = base.with_mode(mode);
                    for k in [1, 3, 10, 25, 500] {
                        assert_matches_oracle(&st, &clauses, k, &opts);
                    }
                }
            }
        }
    }

    /// Ties: identical rows across superfiles score equally and come back
    /// by ascending `_id`, and a `k` that cuts through the tied run keeps
    /// the smallest ids.
    #[test]
    fn ties_break_by_ascending_id_across_superfiles() {
        let st = multi_file_corpus();
        let clauses = [
            FieldClause::new("rust cloud", ["title"]),
            FieldClause::new("storage engine", ["body"]),
            FieldClause::new("db", ["tags"]),
        ];
        let opts = Bm25FieldsOptions::new().with_mode(BoolMode::And);
        let all = fields(&st, &clauses, 500, &opts);
        let top = all[0].1;
        let tied: Vec<Id> = all
            .iter()
            .take_while(|(_, s)| *s == top)
            .map(|(id, _)| *id)
            .collect();
        assert!(tied.len() > 3, "the tied run spans several rows: {all:?}");
        assert!(
            tied.windows(2).all(|w| w[0] < w[1]),
            "ascending ids: {tied:?}"
        );
        let cut = fields(&st, &clauses, 3, &opts);
        assert_eq!(
            cut.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            tied[..3].to_vec()
        );
        assert_matches_oracle(&st, &clauses, 3, &opts);
    }

    /// `k` past the match count returns every match, no more.
    #[test]
    fn k_past_the_match_count_returns_every_match() {
        let st = multi_file_corpus();
        let clauses = [FieldClause::new("python", ["title"])];
        let got = fields(&st, &clauses, 500, &Bm25FieldsOptions::new());
        let want = oracle(&st, &clauses, usize::MAX, &Bm25FieldsOptions::new(), None);
        assert!(!want.is_empty() && want.len() < 500);
        assert_eq!(got, want);
        assert!(
            fields(
                &st,
                &[FieldClause::new("zebra", ["title", "body"])],
                10,
                &Bm25FieldsOptions::new()
            )
            .is_empty()
        );
        assert!(fields(&st, &clauses, 0, &Bm25FieldsOptions::new()).is_empty());
    }

    /// Each column keeps its own analyzer: `running` matches `run` and
    /// `runs` on the stemmed `title`, but only the literal token on the
    /// unstemmed `body`.
    #[test]
    fn each_column_uses_its_own_analyzer() {
        let st = multi_file_corpus();
        let opts = Bm25FieldsOptions::new();
        let title_only = fields(&st, &[FieldClause::new("running", ["title"])], 500, &opts);
        let title_run = fields(&st, &[FieldClause::new("run", ["title"])], 500, &opts);
        assert_eq!(
            title_only.iter().map(|r| r.0).collect::<BTreeSet<_>>(),
            title_run.iter().map(|r| r.0).collect::<BTreeSet<_>>(),
            "the stemmer folds running/runs/run on title"
        );
        let body_only = fields(&st, &[FieldClause::new("running", ["body"])], 500, &opts);
        assert!(body_only.is_empty(), "body holds no literal `running`");
        let both = fields(
            &st,
            &[FieldClause::new("running", ["title", "body"])],
            500,
            &opts,
        );
        assert_eq!(both, title_only);
    }

    /// The table-wide statistics: the same rows split over several
    /// superfiles score as they do in one.
    #[test]
    fn multi_file_scores_equal_single_file_scores() {
        let rows = corpus_rows(240);
        let split = corpus_table(&rows, 40);
        let single = corpus_table(&rows, rows.len());
        assert!(split.reader().expect("reader").n_superfiles() > 1);
        assert_eq!(single.reader().expect("reader").n_superfiles(), 1);
        for clauses in query_cases() {
            for mode in [BoolMode::Or, BoolMode::And] {
                let opts = Bm25FieldsOptions::new().with_mode(mode);
                let a = fields(&split, &clauses, 25, &opts);
                let b = fields(&single, &clauses, 25, &opts);
                assert_eq!(a.len(), b.len(), "{clauses:?}");
                // `_id` names a row's superfile, so the two layouts give the
                // same rows different ids: the ranked scores are compared.
                for ((_, sa), (_, sb)) in a.iter().zip(&b) {
                    assert!(
                        (sa - sb).abs() <= 1e-5 * sa.abs().max(1.0),
                        "{clauses:?}: {sa} vs {sb}"
                    );
                }
            }
        }
    }

    /// A projection returns the named columns, as `bm25_search` does.
    #[test]
    fn projection_returns_named_columns() {
        let st = multi_file_corpus();
        let clauses = [FieldClause::new("rust", ["title", "body"])];
        let rows = st
            .bm25_search_fields(
                &clauses,
                5,
                &Bm25FieldsOptions::new(),
                Some(&["_id", "title", "bucket", "score"]),
            )
            .expect("projected");
        let schema = rows[0].schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, ["_id", "title", "bucket", "score"]);
        assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 5);
    }

    #[test]
    fn invalid_queries_are_rejected() {
        let st = multi_file_corpus();
        let opts = Bm25FieldsOptions::new();
        let err = |clauses: &[FieldClause], opts: &Bm25FieldsOptions| {
            st.bm25_search_fields(clauses, 5, opts, None)
                .expect_err("rejected")
                .to_string()
        };
        assert!(err(&[], &opts).contains("at least one clause"));
        assert!(
            err(&[FieldClause::new("rust", Vec::<String>::new())], &opts)
                .contains("names no column")
        );
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let weighted = Bm25FieldsOptions::new().with_weight("title", bad);
            assert!(
                err(&[FieldClause::new("rust", ["title"])], &weighted)
                    .contains("finite and greater than 0"),
                "weight {bad}"
            );
        }
        // A column without a full-text index, and one the table lacks.
        st.bm25_search_fields(&[FieldClause::new("1", ["bucket"])], 5, &opts, None)
            .expect_err("no FTS index on bucket");
        st.bm25_search_fields(&[FieldClause::new("rust", ["nope"])], 5, &opts, None)
            .expect_err("no such column");
    }

    #[test]
    fn clauses_and_weights_parse_from_json() {
        let clauses = clauses_from_json(
            r#"[{"query": "a b", "fields": ["x", "y"]}, {"query": "c", "fields": ["z"]}]"#,
        )
        .expect("clauses");
        assert_eq!(
            clauses,
            [
                FieldClause::new("a b", ["x", "y"]),
                FieldClause::new("c", ["z"])
            ]
        );
        for bad in [
            "{}",
            "[{\"query\": \"a\"}]",
            "[{\"query\": \"a\", \"fields\": [\"x\"], \"boost\": 2}]",
            "[{\"query\": 1, \"fields\": [\"x\"]}]",
            "not json",
        ] {
            assert!(clauses_from_json(bad).is_err(), "{bad}");
        }
        let weights = weights_from_json(r#"{"x": 2, "y": 0.5}"#).expect("weights");
        assert_eq!(weights.get("x"), Some(&2.0));
        assert_eq!(weights.get("y"), Some(&0.5));
        assert!(weights_from_json("[1]").is_err());
        assert!(weights_from_json(r#"{"x": "2"}"#).is_err());
    }

    /// The hosted statement: quotes doubled, weights only when set, the
    /// default projection `_id` + `score`, best first then ascending `_id`.
    #[cfg(feature = "remote")]
    #[test]
    fn fields_statement_carries_the_call() {
        let clauses = [FieldClause::new("o'brien", ["title", "body"])];
        let plain = fields_statement(
            "my'table",
            &clauses,
            7,
            &Bm25FieldsOptions::new(),
            None,
            "_id",
        );
        assert_eq!(
            plain,
            r#"SELECT "_id", "score" FROM bm25_search_fields('my''table', '[{"fields":["title","body"],"query":"o''brien"}]', 7, 'or') ORDER BY "score" DESC, "_id" ASC"#
        );
        let weighted = fields_statement(
            "t",
            &clauses,
            3,
            &Bm25FieldsOptions::new()
                .with_mode(BoolMode::And)
                .with_weight("title", 2.0),
            Some(&["_id", "title", "score"]),
            "_id",
        );
        assert!(
            weighted.starts_with(r#"SELECT "_id", "title", "score" FROM"#),
            "{weighted}"
        );
        assert!(
            weighted.contains(r#", 3, 'and', '{"title":2.0}')"#),
            "{weighted}"
        );
        // The statement parses back to the same call.
        let json = plain
            .split("', '")
            .nth(1)
            .and_then(|rest| rest.split("', 7").next())
            .expect("clauses literal")
            .replace("''", "'");
        assert_eq!(clauses_from_json(&json).expect("round trip"), clauses);
    }
}
