# Hidden Vector Index: Superfile SPFresh Plan

Status: **design/spec**. This document is the implementation plan for the
hidden vector index after auditing the current codebase.

The corrected design is simple:

- Infino uses **supertables** as the table/manifest/commit abstraction.
- Infino uses **superfiles** as the durable file format for both user data and
  hidden vector-index data.
- User and hidden superfiles share the same vector builder/reader/query code.
  They differ in lifecycle and size, not in fundamental format.
- The current IVF vector subsection is not the target. The superfile envelope
  stays, but the embedded vector blob becomes a specialized SPFresh-style
  vector layout.
- The existing global `VectorCell` routing remains. The new work is inside
  those global cells: one maintained routing tree per global cell, plus
  optimized cluster/run bytes inside superfiles.

This is essentially turbopuffer/SPFresh using Infino supertables and
superfiles as the durable substrate.

## Non-Negotiables

1. **Keep `VectorCell` outer routing.** The global cell grid is still the
   first routing layer. Do not delete it, bypass it, or replace it with a flat
   global fragment table.
2. **Same superfile format for user and hidden data.** User and hidden
   superfiles use the same vector layout and the same reader/builder code.
3. **The superfile envelope stays.** Superfiles remain valid superfiles:
   Parquet scalar body, metadata, vector subsection offsets, manifest entries,
   cache hints, and MVCC commits all stay in the existing machinery.
4. **The vector blob changes.** We do not need the current IVF internals
   (`cluster_idx`, per-superfile IVF centroids as the query routing layer,
   codec metadata shaped around IVF). Replace that with a compact vector blob
   optimized for cell-local tree/runs.
5. **No OPANN radius field.** Radius/radii/radius-aware probing are not part
   of this plan. Do not add them to the new manifest model.
6. **No separate hidden object format.** Hidden vector-index files are still
   superfiles, not headerless raw object blobs.
7. **Drain and compaction are maintenance, not new storage systems.** Reuse the
   existing supertable drain, commit, compaction, cache, and GC paths.

## Mental Model

There are two supertables:

- The **user supertable** is the source of truth. It is append-oriented and
  has smaller superfiles.
- The **hidden vector-index supertable** is a derived maintained index. It has
  larger superfiles and is continuously rewritten through drain and compaction.

Both tables use superfiles. Both superfile classes carry a vector blob in the
same format. The difference is operational:

- User superfiles are fresh segments. They are generally smaller and always
  built against the global cell grid.
- Hidden superfiles are maintained SPFresh/LSM segments. Drain folds user
  superfiles into hidden superfiles, and compaction merges long-lived hidden
  superfiles.

The routing hierarchy is:

```text
query
  -> global VectorCell selection
  -> one tree for each selected global cell
  -> cluster/run refs inside hidden superfiles
  -> fetch vector bytes
  -> rerank
```

There should be `GLOBAL_VECTOR_CELL_COUNT` cell-local trees. In today's code
that count is `GLOBAL_VECTOR_CELL_COUNT = 64`; the name may become configurable
later, but the concept remains a coarse global cell count, not a corpus-scaled
OPANN K.

## Comparison To Turbopuffer

Same core idea:

- Object storage is durable.
- Hot routing metadata lives in memory.
- Query first routes in memory, then fetches a small number of contiguous
  cluster/list ranges, then reranks.
- Maintenance is SPFresh-like: drain, local merge, split/reassign, compaction.
- Immutable blobs are updated by writing new blobs and swapping metadata, not
  by in-place mutation.

Infino differs in the substrate:

- Turbopuffer is a vector service. Infino is an embedded multimodal table
  engine.
- Turbopuffer owns a vector-native storage format. Infino uses superfiles for
  both user data and hidden index data.
- Turbopuffer maintains service-side cache tiers. Infino uses the existing
  local reader cache / disk cache and supertable manifest.
- Turbopuffer's vector index is the primary artifact. Infino's hidden vector
  index is derived from a user supertable and can be rebuilt.

Implementation goal: copy the SPFresh/turbopuffer shape, but express it with
Infino's supertable manifest, superfile vector blob, drain, compaction, and GC.

## What Exists Today

Validated against the current codebase:

- `src/supertable/handle.rs`
  - `GLOBAL_VECTOR_CELL_COUNT` is the current coarse global cell count.
  - `build_vector_index_options()` creates the hidden vector-index supertable.
  - Hidden vector table options currently use `VectorLayout::Ivf`.
  - Hidden vector table partitioning is `PartitionStrategy::VectorCell`.
  - `train_global_centroids()` bootstraps/open-time trains the global cell grid.
- `src/supertable/writer.rs`
  - `commit_appends_internal()` bootstraps `global_vector_index` on the user
    table's first vector commit.
  - `bootstrap_centroids_from_batch()` trains the first global cell grid.
  - `drain_user_superfiles_to_hidden_cells()` is the existing drain harness.
  - `materialized_ivf_rows_in_doc_order()` materializes current IVF rows for
    maintenance.
  - `build_one_shard_from_materialized()` builds hidden cell superfiles today.
  - `prepare_superfile()`, `finish_superfile_entry()`,
    `collect_prepared_superfiles()`, and `persist_commit_async()` are the
    existing publish path to keep.
- `src/supertable/query/vector.rs`
  - `vector_search_global_index_async()` is the query entry point.
  - It currently selects global cells with `select_cells_adaptive()`.
  - It then calls `fanout_vector_clusters()` over selected hidden superfiles.
  - The inner kernel eventually reaches `VectorReader::probe_clusters_async()`.
- `src/supertable/manifest/list.rs`
  - `ManifestList.global_vector_index` stores the user-owned global grid.
  - `PartitionStrategy::VectorCell` is persisted and should remain.
- `src/supertable/manifest/mod.rs`
  - `ClusterCentroids` is the current fp32 centroid container.
  - `select_cells_adaptive()` is the current outer global cell selector.
  - `superfiles_for_routed_cells()` loads/list-filters hidden superfiles for
    routed global cells.
- `src/superfile/vector/layout.rs`
  - Current layouts are `Ivf` and `CellPosting`.
  - Add a new layout for the optimized SPFresh vector blob.
- `src/superfile/vector/builder.rs`
  - `build_subsection_from_materialized()` is the current Sq8-native rebuild
    path.
  - This is the right area to add the new vector blob writer, but the new
    writer should not preserve IVF internals.
- `src/superfile/vector/reader.rs`
  - Current vector reader is IVF-centric.
  - Add a sibling reader path for the new blob rather than stretching
    `probe_clusters_async()` into a non-IVF shape.
- `src/superfile/vector/cell_posting.rs`
  - `MaterializedIvfRow`, `EncodedCellRow`, and
    `materialize_sq8_residual_row_into_cluster_quant()` are useful transition
    helpers while the existing data is IVF.

## Audit Findings

The previous plan was wrong in these ways:

- It treated OPANN as a flat global centroid table. Correct design keeps the
  coarse `VectorCell` layer and adds per-global-cell trees.
- It suggested deleting `VectorCell`. Correct design keeps it.
- It introduced radius/radii. Correct design has no OPANN radius field.
- It introduced a global `OpannIndex` fragment table. Correct design should
  use cell-local tree/run references in the supertable manifest.
- It made K scale with corpus size by replacing `GLOBAL_VECTOR_CELL_COUNT`.
  Correct design keeps `GLOBAL_VECTOR_CELL_COUNT` as the outer cell count.
  Inner cluster/list counts are controlled inside each global cell.
- It implied hidden objects are raw headerless data blobs. Correct design keeps
  hidden superfiles as superfiles.
- It implied user and hidden vector formats differ. Correct design keeps one
  vector blob format shared by both.
- It described drain as append-only delta recording. Correct design says drain
  adds new hidden superfiles and may merge with adjacent/small hidden
  superfiles; compaction handles larger long-lived merges.

## Target Vector Blob

Working name: `VectorLayout::Spfresh` until naming is finalized.

The superfile envelope remains unchanged. The vector subsection changes from
current IVF internals to a cell/tree/run-oriented blob.

Current IVF-like shape to retire:

```text
sub_header
summary_centroid
centroids
cluster_idx
codec_meta
stable_ids
per_cluster_blocks
crc
```

Target shape, conceptually:

```text
vector_subsection:
  subsection header
  global-cell directory [GLOBAL_VECTOR_CELL_COUNT]
  cell segment 0
  cell segment 1
  ...
  crc

cell segment:
  local run directory
  optimized encoded rows grouped by cluster/list

run:
  quantizer for this run
  rows ordered for merge/scan

row:
  rabitq code
  Sq8+epsilon codes/residuals
  stable_id
  optional merge key if needed
```

The exact byte layout belongs in the P1 spec, but the invariant is clear:
candidate vectors for a selected inner cluster/list must be physically
contiguous enough to fetch with a small number of ranges.

The vector blob should support both:

- smaller user superfiles built at ingest time, and
- larger hidden superfiles produced by drain/compaction.

## Manifest Routing Model

The supertable manifest keeps routing metadata for the hidden vector index.

Correct model:

```text
HiddenVectorIndex {
  column
  rot_seed
  metric
  global_cells: Vec<CellTree>   // length == GLOBAL_VECTOR_CELL_COUNT
}

CellTree {
  cell_id
  tree_nodes
  leaves: cluster/run references for that global cell
}

RunRef {
  superfile_uri
  cell_id
  run_id or byte range
}
```

Names are placeholders. The important point is that there is one tree per
global cell, and each tree routes within that cell to runs stored in ordinary
superfiles.

No `radius`. No `fragments` terminology. Use `RunRef`, `ClusterRef`, or another
clear name for physical run locations.

Where this likely lands:

- Add persisted manifest structs in `src/supertable/manifest/list.rs`.
- Add DTO encode/decode next to existing `GlobalVectorIndexDto` and
  `PartitionStrategyDto`.
- Carry the new routing metadata through `Manifest::update()` in
  `src/supertable/manifest/mod.rs`.
- Add `Manifest::with_hidden_vector_trees()` and
  `Manifest::get_hidden_vector_trees()` or equivalent.
- Keep `PartitionStrategy::VectorCell`.

If the routing metadata becomes too large for the JSON list, store it as a
content-addressed manifest blob using the same pattern as other side metadata,
but it is still part of the supertable manifest state. Do not invent an
independent index store.

## Query Shape

Keep the existing outer entry point:

- `SupertableReader::vector_search_global_index_async()` in
  `src/supertable/query/vector.rs`.

Correct query flow:

1. Use the existing hidden vector-index table lookup:
   `self.vector_index_table()`.
2. Load the hidden table reader:
   `vit.reader()`.
3. Use existing global cell routing:
   `ClusterCentroids::select_cells_adaptive()` from
   `src/supertable/manifest/mod.rs`.
4. Use existing cell-to-superfile filtering:
   `filter_superfiles_by_cells()` / `superfiles_for_routed_cells()`.
5. For each selected global cell, use that cell's manifest-maintained tree to
   select inner clusters/runs.
6. Fetch ranges from selected hidden superfiles using existing reader/cache
   infrastructure.
7. Do 1-bit shortlist and Sq8+epsilon rerank.
8. Dedup by stable `_id` and apply the existing hidden deleted-set logic.

What changes:

- The inner call to `fanout_vector_clusters()` should branch by vector layout.
- For `VectorLayout::Ivf`, keep the current behavior.
- For `VectorLayout::Spfresh`, use a new cell-tree/run kernel.

Likely files/methods:

- `src/supertable/query/vector.rs`
  - keep `vector_search_global_index_async()`;
  - add `fanout_vector_spfresh_runs()` or similar;
  - keep `filter_superfiles_by_cells()`;
  - reuse `top_k_ascending()`.
- `src/supertable/query/dispatch.rs`
  - reuse the existing fanout/open-reader shape.
- `src/supertable/query/superfile_reader.rs`
  - reuse reader open/cache path.
- `src/superfile/vector/reader.rs`
  - add a sibling to the IVF query path, e.g.
    `search_spfresh_cell_async()` / `probe_spfresh_runs_async()`.

Do not add a query path that scores one flat global OPANN centroid list before
`VectorCell`. That is the wrong architecture for this plan.

## Drain Shape

Keep the existing drain entry point:

- `drain_user_superfiles_to_hidden_cells()` in `src/supertable/writer.rs`.

Correct drain flow:

1. Keep single-flight through `hidden_inner.compaction_outstanding`.
2. Keep `drained_ranges`.
3. Keep version-aligned batching through `drain_batch_superfiles()`.
4. Keep full-resident user superfile opens and stable-id resolution.
5. Materialize rows from user superfiles.
6. For each global cell, add new user-superfile data into the hidden cell LSM.
7. Drain may produce new hidden superfiles and may merge with adjacent/small
   hidden superfiles in that cell.
8. Publish through the existing `prepare_superfile()` /
   `finish_superfile_entry()` / `collect_prepared_superfiles()` /
   `persist_commit_async()` path.
9. Update the manifest-maintained cell trees for affected global cells in the
   same commit.

Current helper reuse:

- `materialized_ivf_rows_in_doc_order()` is useful during transition from IVF.
- `build_one_shard_from_materialized()` is the current builder path, but the
  new layout should eventually call a new vector blob builder instead of IVF.
- `prepare_superfile()` and `finish_superfile_entry()` stay.
- `PartitionStrategy::VectorCell` stamping stays.

Drain is not a brand-new loop. It is a new branch inside the existing drain
build/maintenance step.

## Compaction Shape

Hidden compaction remains the long-lived maintenance path.

Likely files:

- `src/supertable/compaction/mod.rs`
- `src/supertable/gc/`
- `src/supertable/optimize/`
- `src/supertable/writer.rs` for shared build helpers

Correct behavior:

- Drain handles fresh user data and small/adjacent merges.
- Compaction handles larger, older hidden superfiles.
- Both produce ordinary hidden superfiles in the same vector layout.
- Both update the relevant global-cell trees in the supertable manifest.
- Both use MVCC manifest swap and existing GC, not in-place mutation.

## Phased Implementation

### P(-1): Diagnostic Gate (done)

This was validated at the scale where the problem appears. The recall drop is
not meaningful at 1M docs because routing errors do not yet push candidates far
enough away; the diagnostic has to be interpreted at the 10M-doc regime where
recall starts to fall.

The useful diagnostic answers:

- Did the outer `VectorCell` routing select the true neighbor's global cell?
- Did the inner IVF superfile/cluster selection select the true neighbor's
  cluster?
- Was the candidate fetched but lost during 1-bit/Sq8 rerank?

Files:

- `src/supertable/query/vector.rs`
- `benches/utils/` or a dedicated diagnostic in the existing bench harness

This phase is no longer a blocker for P0.

### P0: Scaffold And Flags

Add a layout flag without behavior change.

Files:

- `src/superfile/vector/layout.rs`
  - add `VectorLayout::Spfresh` and a metadata string.
- `src/supertable/options.rs`
  - ensure `with_vector_layout()` carries the new layout.
- `src/supertable/handle.rs`
  - keep `GLOBAL_VECTOR_CELL_COUNT` unchanged.
  - in `build_vector_index_options()`, allow hidden index layout selection by
    `INFINO_HIDDEN_INDEX=spfresh|nested`, but default remains current IVF.

Do not change global cell count scaling here.

### P1: New Superfile Vector Blob

Implement the optimized vector subsection.

Files:

- `src/superfile/vector/spfresh.rs` or
  `src/superfile/vector/cell_tree.rs`
- `src/superfile/vector/mod.rs`
- `src/superfile/builder.rs`
- `src/superfile/reader.rs`
- `src/superfile/vector/builder.rs`
- `src/superfile/vector/reader.rs`
- `src/supertable/writer.rs`
- `src/supertable/manifest/part.rs`

Work:

- Add layout dispatch for build/read/open.
- Add blob writer from materialized rows.
- Add blob reader that can fetch/scan selected cell-local runs.
- Add `build_subsection_offsets()` support in `writer.rs`.
- Add manifest part roundtrip for the new layout string.
- Add tests for:
  - layout metadata roundtrip,
  - vector blob roundtrip,
  - selected run fetch,
  - rerank vs brute-force on a small corpus.

### P2: Manifest Cell Trees

Add manifest-owned routing trees for hidden vector cells.

Files:

- `src/supertable/manifest/list.rs`
- `src/supertable/manifest/mod.rs`
- `src/supertable/manifest/options_hash.rs`
- `src/supertable/manifest/commit.rs` if side metadata needs commit support

Work:

- Add a hidden-vector routing structure with `GLOBAL_VECTOR_CELL_COUNT` trees.
- Persist it in the manifest list or in a content-addressed manifest blob
  referenced from the list.
- Add `with_*` and `get_*` methods on `Manifest`.
- Add DTO encode/decode and roundtrip tests.

No radius fields.

### P3: Build User Superfiles With The New Blob

User superfiles should use the same new vector blob layout.

Files:

- `src/supertable/writer.rs`
- `src/superfile/vector/builder.rs`
- `src/superfile/builder.rs`

Work:

- Keep `bootstrap_centroids_from_batch()` and
  `ManifestList.global_vector_index` as the source of the global cell grid.
- Build each user superfile with `GLOBAL_VECTOR_CELL_COUNT` global cells.
- Use the new vector blob format inside the superfile.
- Keep source `_id` handling and scalar stats unchanged.

### P4: Drain Into Hidden Cell LSMs

Extend the existing drain, do not fork it.

Files:

- `src/supertable/writer.rs`
- `src/superfile/vector/spfresh.rs` or chosen vector blob module
- `src/supertable/manifest/mod.rs`

Work:

- Branch inside `drain_user_superfiles_to_hidden_cells()`.
- Keep batching, `drained_ranges`, single-flight, and MVCC.
- For each drained batch:
  - materialize rows from user superfiles;
  - group by global cell;
  - merge fresh user data with adjacent/small hidden superfiles where the
    maintenance policy says to;
  - build new hidden superfiles in the same vector layout;
  - update affected cell trees;
  - publish with `persist_commit_async()`.

### P5: Query New Layout

Wire query through the existing global path.

Files:

- `src/supertable/query/vector.rs`
- `src/supertable/query/dispatch.rs`
- `src/supertable/query/superfile_reader.rs`
- `src/superfile/vector/reader.rs`

Work:

- Keep `vector_search_global_index_async()`.
- Keep `select_cells_adaptive()`.
- Keep selected-superfile filtering by `VectorCell`.
- Add inner dispatch for `VectorLayout::Spfresh`.
- Descend selected cell trees to choose cell-local clusters/runs.
- Fetch selected ranges and rerank.
- Dedup stable ids exactly as current hidden-hit remapping expects.

### P6: Hidden Compaction

Extend hidden compaction for the new vector layout.

Files:

- `src/supertable/compaction/mod.rs`
- `src/supertable/gc/`
- `src/supertable/optimize/`

Work:

- Merge larger long-lived hidden superfiles.
- Rebuild affected cell trees.
- Preserve `PartitionStrategy::VectorCell`.
- Preserve MVCC and GC safety.

### P7: Cutover

Only after recall, GET count, and write/compact cost are proven:

- default hidden vector layout can move from IVF to the new layout;
- current inner IVF-specific code can be retired where the new layout fully
  replaces it;
- `VectorCell` stays.

Do not delete the outer `VectorCell` routing system.

## Validation Gates

- Correctness:
  - recall@10 >= 0.99 on the standard vector bench;
  - exact/brute-force oracle on small corpora;
  - hidden/user stable-id remapping unchanged;
  - delete filtering unchanged.
- I/O:
  - object GET count per query;
  - range count per query;
  - bytes fetched per query;
  - distinct hidden superfiles touched per query.
- Maintenance:
  - drain wall time;
  - compaction write amplification;
  - hidden superfile size distribution;
  - per-global-cell tree size and rebuild time.
- Safety:
  - existing `make ci`;
  - crash-safety tests if commit/manifest path changes;
  - no new unsafe expected.

## Names To Avoid

Avoid these in the implementation plan unless explicitly referring to the old,
discarded flat plan:

- `OpannIndex`
- `fragments`
- `radius`
- `radii`
- `radius-aware`
- "delete `VectorCell`"
- "single-level global OPANN"
- "hidden objects are headerless raw blobs"

Prefer:

- `CellTree`
- `RunRef`
- `ClusterRef`
- `VectorLayout::Spfresh` or final chosen layout name
- "global cells + per-cell trees"
- "same superfile format, new vector blob"
