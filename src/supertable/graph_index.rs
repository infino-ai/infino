// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The resident adjacency index over an edge table — the knowledge graph's
//! counterpart of the `hnsw` build, in the same lifecycle: `optimize()`
//! scans the edge rows once into per-node adjacency lists, lays them out as
//! a single-level [`Hnsw`](crate::superfile::vector::hnsw::Hnsw) graph with
//! the node ids and keys (`superfile::vector::adjacency`), publishes one
//! content-addressed blob through the writer's resident-index publish, and
//! stamps its reference on the manifest under the commit protocol every
//! maintenance stamp shares. Hydration, memory-mapping, the resident slot
//! and GC are the vector index's, unchanged; the walks are
//! `query::graph`. Behind the `graph-index` feature.

use std::{sync::Arc, time::Instant};

use arrow_array::{Array, Int64Array, LargeStringArray, StringArray, StringViewArray};
use tracing::debug;

use super::{
    error::BuildError,
    handle::{Supertable, SupertableReader},
    writer::{publish_resident_index, resident_index_population_key, stamp_with_retries},
};
use crate::{
    config::AdjacencySpec,
    superfile::vector::{
        adjacency::AdjacencyBuild,
        hnsw::{GRAPH_BUNDLE_HEADER_BYTES, PayloadKind, resident_envelope_header},
    },
};

/// The strings of a key column, whichever string layout the scan produced.
fn key_strings(column: &dyn Array) -> Option<Vec<Option<&str>>> {
    let any = column.as_any();
    if let Some(a) = any.downcast_ref::<LargeStringArray>() {
        Some(a.iter().collect())
    } else if let Some(a) = any.downcast_ref::<StringArray>() {
        Some(a.iter().collect())
    } else {
        any.downcast_ref::<StringViewArray>()
            .map(|a| a.iter().collect())
    }
}

/// Every edge of the table `reader` pins, as `(source, destination, source
/// key)` in row order, gathered into adjacency lists: one SQL scan of the
/// three columns `spec` names.
async fn scan_edges(
    reader: &SupertableReader,
    spec: &AdjacencySpec,
) -> Result<AdjacencyBuild, BuildError> {
    let quote = |column: &str| format!("\"{}\"", column.replace('"', "\"\""));
    let sql = format!(
        "SELECT {}, {}, {} FROM supertable",
        quote(&spec.src),
        quote(&spec.dst),
        quote(&spec.key)
    );
    let ctx = reader
        .sql_session_context()
        .map_err(|e| BuildError::Store(e.to_string()))?;
    let batches = ctx
        .sql(&sql)
        .await
        .map_err(|e| BuildError::Store(format!("adjacency scan: {e}")))?
        .collect()
        .await
        .map_err(|e| BuildError::Store(format!("adjacency scan: {e}")))?;
    let mut edges: Vec<(i64, i64, Option<&str>)> = Vec::new();
    for batch in &batches {
        let src = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| BuildError::Store(format!("adjacency: {} is not Int64", spec.src)))?;
        let dst = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| BuildError::Store(format!("adjacency: {} is not Int64", spec.dst)))?;
        let keys = key_strings(batch.column(2).as_ref()).ok_or_else(|| {
            BuildError::Store(format!("adjacency: {} is not a string column", spec.key))
        })?;
        for (row, key) in keys.into_iter().enumerate() {
            if src.is_null(row) || dst.is_null(row) {
                continue;
            }
            edges.push((src.value(row), dst.value(row), key));
        }
    }
    Ok(AdjacencyBuild::from_edges(edges.into_iter()))
}

/// Build and publish the adjacency index over an edge table's CURRENT rows
/// and stamp its reference on a successor manifest — the knowledge graph's
/// counterpart of the `hnsw` build, in the same lifecycle: one
/// content-addressed blob through [`publish_resident_index`], the manifest's
/// resident-index ref, hydration through the resident slot, GC while
/// referenced. Maintenance-only: `optimize` calls it after compaction, so
/// the index describes the post-merge rows. A no-op when the stamped blob's
/// header already names this row population. A lost commit race retries
/// with a fresh scan under [`stamp_with_retries`]; the walks keep serving
/// the prior generation until a pass republishes.
pub(in crate::supertable) async fn stamp_adjacency(
    table: &Supertable,
    spec: &AdjacencySpec,
) -> Result<(), BuildError> {
    let inner = table.inner();
    let Some(storage) = inner.options.storage.clone() else {
        return Ok(());
    };
    stamp_with_retries(inner, &storage, "adjacency", |old| {
        let storage = Arc::clone(&storage);
        async move {
            let entries = old.get_all_superfiles();
            if entries.is_empty() {
                return Ok(None);
            }
            let population_key = resident_index_population_key(&old);
            if let Some(current) = old.resident_vector_index_blob() {
                let header = storage
                    .get_range(&current.uri, 0..GRAPH_BUNDLE_HEADER_BYTES as u64)
                    .await
                    .ok();
                if header
                    .as_deref()
                    .and_then(resident_envelope_header)
                    .is_some_and(|(key, _)| key == population_key)
                {
                    return Ok(None);
                }
            }
            let high_water = entries.iter().map(|e| e.id_max).max().unwrap_or(0);
            // The scan pins the handle's current manifest; if a commit lands
            // between that and `old`, the CAS below fails on `old`'s etag
            // and the retry rescans.
            let reader = table.pinned_reader();
            let build = scan_edges(&reader, spec).await?;
            if build.is_empty() {
                return Ok(None);
            }
            let t0 = Instant::now();
            let (nodes, edge_count) = (build.node_count(), build.edge_count());
            let payload = build.encode();
            let Some(reference) = publish_resident_index(
                storage.as_ref(),
                population_key,
                high_water,
                PayloadKind::Adjacency,
                &payload,
            )
            .await
            else {
                return Err(BuildError::Store("adjacency publish failed".into()));
            };
            debug!(
                nodes,
                edges = edge_count,
                payload_mib = payload.len() / (1024 * 1024),
                wall_s = t0.elapsed().as_secs_f64(),
                "adjacency: built and published"
            );
            Ok(Some(old.with_adjacency_ref(reference)))
        }
    })
    .await
}
