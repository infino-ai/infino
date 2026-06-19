// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Supertable search helpers shared by vector + FTS benches.

use infino::supertable::Supertable;
use infino::supertable::query::SuperfileHit;
use infino::supertable::query::vector::VectorSearchOptions;

use crate::ingest::supertable::VEC_COLUMN;

pub fn vector_topk_global(
    st: &Supertable,
    query: &[f32],
    k: usize,
    options: VectorSearchOptions,
) -> Vec<u32> {
    let hits: Vec<SuperfileHit> = st
        .reader()
        .vector_hits(VEC_COLUMN, query, k, options, None)
        .expect("vector_search");
    let r = st.reader();
    let manifest = r.manifest();
    let mut offsets: Vec<u32> = Vec::with_capacity(manifest.superfiles.len());
    let mut acc: u32 = 0;
    for entry in manifest.superfiles.iter() {
        offsets.push(acc);
        acc = acc.saturating_add(entry.n_docs as u32);
    }
    hits.into_iter()
        .map(|h| {
            let seg_idx = manifest
                .superfiles
                .iter()
                .position(|e| e.uri == h.superfile)
                .expect("superfile in manifest");
            offsets[seg_idx] + h.local_doc_id
        })
        .collect()
}
