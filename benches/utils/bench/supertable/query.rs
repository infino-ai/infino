// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Supertable search helpers shared by vector + FTS benches.

use infino::supertable::Supertable;
use infino::supertable::query::vector::VectorSearchOptions;

use crate::corpus;
use crate::ingest::supertable::VEC_COLUMN;

pub fn vector_topk_global(
    st: &Supertable,
    query: &[f32],
    k: usize,
    options: VectorSearchOptions,
) -> Vec<i128> {
    let batches = st
        .reader()
        .vector_search(VEC_COLUMN, query, k, options, None, None)
        .expect("vector_search");
    corpus::id_scores_from_vector_search(&batches)
        .into_iter()
        .map(|(id, _)| id)
        .collect()
}
