// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Per-query OPANN recall breakdown (`INFINO_DIAG_OPANN_RECALL=1`).

use std::collections::{HashMap, HashSet};

use arrow_array::{Array, Int64Array, RecordBatch};

use super::{
    opann_diag_hidden_hits, opann_diag_read_i64_column_all_rows, remap_hidden_hits_to_user_hits,
    VectorSearchOptions,
};
use crate::supertable::query::{exec::common::resolve_hits_named, vector_probe::select_opann_probe_leaves, SuperfileHit};
use crate::{
    supertable::{
        error::QueryError,
        handle::{Supertable, SupertableReader},
        manifest::SuperfileEntry,
    },
    InfinoError,
};

/// How many correctness queries to print in detail.
const DEFAULT_DIAG_QUERY_COUNT: usize = 3;
/// Expanded hidden fetch for rank-below-top-k checks.
const EXPANDED_FETCH_K: usize = 100;

/// One query's recall breakdown.
#[derive(Debug, Clone)]
pub struct OpannRecallQueryBreakdown {
    pub query_index: usize,
    pub recall_at_k: f32,
    pub probed_cells: usize,
    pub total_cells: usize,
    pub returned_doc_keys: Vec<u32>,
    pub truth_top_k: Vec<u32>,
    pub oracle_top_k_in_probed_cells: Vec<u32>,
    pub misses: Vec<u32>,
    pub misses_in_unprobed_cell: usize,
    pub misses_in_probed_cell: usize,
    pub oracle_recall_in_probed_cells: f32,
    pub misses_found_in_expanded_fetch: usize,
    pub hits_match_search: bool,
}

fn diag_query_count() -> usize {
    std::env::var("INFINO_DIAG_OPANN_RECALL_QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_DIAG_QUERY_COUNT)
        .max(1)
}

test_visible! {
/// Print a per-query OPANN recall breakdown for the first N queries.
fn run_opann_recall_breakdown(
    table: &Supertable,
    column: &str,
    doc_key_column: &str,
    vectors: &[f32],
    dim: usize,
    queries: &[Vec<f32>],
    truths: &[Vec<u32>],
    k: usize,
) -> Result<Vec<OpannRecallQueryBreakdown>, QueryError> {
    let n = diag_query_count().min(queries.len());
    table.reader().block_on(async {
        let reader = table.reader();
        let doc_key_to_cell =
            build_doc_key_to_hidden_cell_map(&reader, doc_key_column).await?;
        let total_cells = reader
            .vector_index_table()
            .map(|vit| vit.reader().manifest().superfiles.len())
            .unwrap_or(0);
        let opts = VectorSearchOptions::default();
        let mut out = Vec::with_capacity(n);
        for qi in 0..n {
            out.push(
                diagnose_one_query(
                    table,
                    &reader,
                    &doc_key_to_cell,
                    total_cells,
                    column,
                    doc_key_column,
                    vectors,
                    dim,
                    &queries[qi],
                    &truths[qi],
                    k,
                    &opts,
                    qi,
                )
                .await?,
            );
        }
        Ok(out)
    })
}
}

fn recall_at_k(returned: &[u32], truth: &[u32]) -> f32 {
    if truth.is_empty() {
        return 1.0;
    }
    let got: HashSet<u32> = returned.iter().copied().collect();
    let hits = truth.iter().filter(|id| got.contains(id)).count();
    hits as f32 / truth.len() as f32
}

fn brute_force_topk_among(
    vectors: &[f32],
    dim: usize,
    query: &[f32],
    allow: &HashSet<u32>,
    k: usize,
) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = allow
        .iter()
        .map(|&id| {
            let off = id as usize * dim;
            let mut dot = 0f32;
            for d in 0..dim {
                dot += vectors[off + d] * query[d];
            }
            (-dot, id)
        })
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    scored.into_iter().take(k).map(|(_, id)| id).collect()
}

fn doc_keys_from_search_batches(
    table: &Supertable,
    column: &str,
    doc_key_column: &str,
    query: &[f32],
    k: usize,
    opts: &VectorSearchOptions,
) -> Result<Vec<u32>, QueryError> {
    let batches = table
        .vector_search(
            column,
            query,
            k,
            opts.clone(),
            None,
            Some(&[doc_key_column]),
        )
        .map_err(|e: InfinoError| QueryError::Execute(e.to_string()))?;
    let mut keys = Vec::new();
    for batch in &batches {
        keys.extend(i64_doc_keys_from_batch(batch, doc_key_column)?);
    }
    Ok(keys)
}

fn doc_keys_from_hits(
    reader: &SupertableReader,
    hits: &[SuperfileHit],
    doc_key_column: &str,
) -> Result<Vec<u32>, QueryError> {
    reader.block_on(async {
        let batch = resolve_hits_named(
            reader,
            hits,
            Some(&[doc_key_column]),
            "opann_recall_diag",
        )
        .await
        .map_err(|e| QueryError::Execute(e.to_string()))?;
        i64_doc_keys_from_batch(&batch, doc_key_column)
    })
}

fn i64_doc_keys_from_batch(batch: &RecordBatch, col: &str) -> Result<Vec<u32>, QueryError> {
    let arr = batch
        .column_by_name(col)
        .ok_or_else(|| QueryError::Execute(format!("column {col:?} missing from batch")))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| QueryError::Execute(format!("column {col:?} is not Int64")))?;
    Ok((0..arr.len()).map(|i| arr.value(i) as u32).collect())
}

async fn build_doc_key_to_hidden_cell_map(
    user_reader: &SupertableReader,
    doc_key_column: &str,
) -> Result<HashMap<u32, u128>, QueryError> {
    let user_manifest = user_reader.manifest();
    let mut doc_key_to_user_cell = HashMap::new();
    for entry in &user_manifest.superfiles {
        let keys = read_doc_keys_for_superfile(user_manifest, entry, doc_key_column).await?;
        let cell = entry.superfile_id.as_u128();
        for key in keys {
            doc_key_to_user_cell.insert(key, cell);
        }
    }

    let hidden = user_reader
        .vector_index_table()
        .ok_or_else(|| QueryError::Execute("hidden vector index missing".into()))?;
    let hidden_reader = hidden.reader();
    let hidden_manifest = hidden_reader.manifest();

    // Hidden OPANN leaves carry hidden `superfile_id`s; user shards carry
    // different ids for the same logical cell. Pair by the aligned id span.
    let mut user_to_hidden: HashMap<u128, u128> = HashMap::new();
    for hidden_entry in &hidden_manifest.superfiles {
        let Some(user_entry) = user_manifest.superfiles.iter().find(|u| {
            u.id_min == hidden_entry.id_min
                && u.id_max == hidden_entry.id_max
                && u.n_docs == hidden_entry.n_docs
        }) else {
            continue;
        };
        user_to_hidden.insert(
            user_entry.superfile_id.as_u128(),
            hidden_entry.superfile_id.as_u128(),
        );
    }

    let mut out = HashMap::with_capacity(doc_key_to_user_cell.len());
    for (doc_key, user_cell) in doc_key_to_user_cell {
        if let Some(&hidden_cell) = user_to_hidden.get(&user_cell) {
            out.insert(doc_key, hidden_cell);
        }
    }
    Ok(out)
}

async fn read_doc_keys_for_superfile(
    manifest: &crate::supertable::manifest::Manifest,
    entry: &SuperfileEntry,
    doc_key_column: &str,
) -> Result<Vec<u32>, QueryError> {
    let raw =
        opann_diag_read_i64_column_all_rows(manifest, entry, doc_key_column).await?;
    Ok(raw.into_iter().map(|v| v as u32).collect())
}

async fn diagnose_one_query(
    table: &Supertable,
    user_reader: &SupertableReader,
    doc_key_to_cell: &HashMap<u32, u128>,
    total_cells: usize,
    column: &str,
    doc_key_column: &str,
    vectors: &[f32],
    dim: usize,
    query: &[f32],
    truth: &[u32],
    k: usize,
    opts: &VectorSearchOptions,
    query_index: usize,
) -> Result<OpannRecallQueryBreakdown, QueryError> {
    let returned =
        doc_keys_from_search_batches(table, column, doc_key_column, query, k, opts)?;

    let hidden = user_reader
        .vector_index_table()
        .ok_or_else(|| QueryError::Execute("hidden vector index missing".into()))?;
    let hidden_reader = hidden.reader();

    let probed_cells: HashSet<u128> = match select_opann_probe_leaves(
        &hidden_reader,
        hidden_reader.manifest(),
        column,
        query,
        opts,
        |_| true,
    )
    .await?
    {
        Some(leaves) => leaves
            .iter()
            .map(|(leaf, _, _)| leaf.superfile_id)
            .collect(),
        None => HashSet::new(),
    };

    let hits = user_reader.vector_hits(column, query, k, opts.clone(), None)?;
    let hits_doc_keys = doc_keys_from_hits(user_reader, &hits, doc_key_column)?;
    let hits_match_search = {
        let a: HashSet<u32> = returned.iter().copied().collect();
        let b: HashSet<u32> = hits_doc_keys.iter().copied().collect();
        a == b
    };

    let expanded_hits =
        opann_diag_hidden_hits(user_reader, column, query, EXPANDED_FETCH_K, opts.clone())
            .await?;
    let expanded_remapped =
        remap_hidden_hits_to_user_hits(user_reader, &expanded_hits).await?;
    let expanded_keys =
        doc_keys_from_hits(user_reader, &expanded_remapped, doc_key_column)?;

    let returned_set: HashSet<u32> = returned.iter().copied().collect();
    let misses: Vec<u32> = truth
        .iter()
        .copied()
        .filter(|id| !returned_set.contains(id))
        .collect();

    let mut misses_in_unprobed_cell = 0usize;
    let mut misses_in_probed_cell = 0usize;
    for miss in &misses {
        match doc_key_to_cell.get(miss) {
            Some(c) if probed_cells.contains(c) => misses_in_probed_cell += 1,
            Some(_) => misses_in_unprobed_cell += 1,
            None => misses_in_unprobed_cell += 1,
        }
    }

    let expanded_set: HashSet<u32> = expanded_keys.iter().copied().collect();
    let misses_found_in_expanded_fetch = misses
        .iter()
        .filter(|id| expanded_set.contains(id))
        .count();

    let probed_doc_keys: HashSet<u32> = doc_key_to_cell
        .iter()
        .filter(|(_, cell)| probed_cells.contains(cell))
        .map(|(&dk, _)| dk)
        .collect();
    let oracle_top_k_in_probed_cells =
        brute_force_topk_among(vectors, dim, query, &probed_doc_keys, k);
    let oracle_recall_in_probed_cells =
        recall_at_k(&returned, &oracle_top_k_in_probed_cells);

    let breakdown = OpannRecallQueryBreakdown {
        query_index,
        recall_at_k: recall_at_k(&returned, truth),
        probed_cells: probed_cells.len(),
        total_cells,
        returned_doc_keys: returned,
        truth_top_k: truth.to_vec(),
        oracle_top_k_in_probed_cells,
        misses,
        misses_in_unprobed_cell,
        misses_in_probed_cell,
        oracle_recall_in_probed_cells,
        misses_found_in_expanded_fetch,
        hits_match_search,
    };
    print_breakdown(&breakdown, k);
    Ok(breakdown)
}

fn print_breakdown(b: &OpannRecallQueryBreakdown, k: usize) {
    eprintln!(
        "[opann-recall-diag] query {}: recall@{k}={:.3} probed_cells={}/{} \
         hits_match_search={}",
        b.query_index,
        b.recall_at_k,
        b.probed_cells,
        b.total_cells,
        b.hits_match_search,
    );
    eprintln!("[opann-recall-diag]   truth={:?}", b.truth_top_k);
    eprintln!(
        "[opann-recall-diag]   returned doc_key={:?}",
        b.returned_doc_keys
    );
    if b.misses.is_empty() {
        eprintln!("[opann-recall-diag]   no misses");
        return;
    }
    eprintln!(
        "[opann-recall-diag]   misses={:?} ({} in unprobed cell, {} in probed cell)",
        b.misses, b.misses_in_unprobed_cell, b.misses_in_probed_cell,
    );
    eprintln!(
        "[opann-recall-diag]   oracle top-{k} within probed cells only={:?}",
        b.oracle_top_k_in_probed_cells,
    );
    eprintln!(
        "[opann-recall-diag]   recall@{k} vs returned={:.3} (if << {:.3} → within-cell/remap; \
         if oracle ≈ {:.3} but returned low → remap/scoring)",
        recall_at_k(&b.returned_doc_keys, &b.oracle_top_k_in_probed_cells),
        b.recall_at_k,
        b.recall_at_k,
    );
    eprintln!(
        "[opann-recall-diag]   {}/{} misses appear in expanded top-{EXPANDED_FETCH_K} fetch",
        b.misses_found_in_expanded_fetch,
        b.misses.len(),
    );
}
