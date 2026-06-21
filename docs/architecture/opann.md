# OPANN — Object-Partitioned Approximate Nearest Neighbor

> Status: design (in progress). This is the canonical reference for the hidden
> vector index. Cells, the rebalancer, and the in-memory routing scorer exist;
> the paged routing tree and its commit/query wiring are being built to this doc.

## Thesis

**SPANN is disk-partitioned ANN. OPANN is object-partitioned ANN.**

OPANN is an object-storage-native ANN architecture:

- **Routing/search layer on compute** — a copy of the partition centroids lives
  in the supertable manifest, searched contiguously with zero object GETs.
- **Vector payload on object storage** — vectors live in immutable,
  object-resident leaf cells.
- **One object GET per probe** — a query descends the routing layer to its
  `n_probe` nearest cells, then fetches one object per cell and scans it.
- **Hot-region rebalancing** — overlap-triggered re-cluster keeps partition
  quality as writes accumulate.

## Pieces

| OPANN term            | What it is                                                        | Code |
| --------------------- | ----------------------------------------------------------------- | ---- |
| Partition (cell)      | Immutable ≤8 MB IVF superfile, vector payload on object storage   | `writer::recluster_cells`, the per-commit cell build |
| Routing layer         | Manifest-resident centroid tree (Sq8residual), searched on compute | `opann::routing` (flat scorer today), the paged tree (to build) |
| Probe                 | One object GET of a routed cell, then whole-cell scan + rerank    | `query::vector` whole-cell scan |
| Hot-region rebalancing | Overlap-degree ≥ τ → re-cluster that region into tight cells      | `spfresh::hot_overlap_groups` + `compaction::overlap_consolidation_jobs` + `writer::recluster_cells` |

## Invariants

- **Cells are immutable and ≤8 MB.** Updates = delete + insert via tombstones;
  hot regions are re-clustered into new cells, never edited in place.
- **One codec: Sq8+residual.** User input is fp32; everything stored — cell
  payloads *and* centroids — is Sq8+residual. There is no plain-Sq8 and no fp32
  storage path.
- **All centroid distances go through `Sq8ResidualKernel`** (code leg + residual
  leg), never plain Sq8, never fp32.

## The routing tree

A hierarchical centroid tree:

- **Leaf node** = one cell. Carries the cell's Sq8residual centroid, its bounding
  radius, and its **cell ref** (the cell superfile's object id + byte range to
  GET).
- **Internal node** = a coarse routing point. Carries the Sq8residual centroid of
  the cells beneath it (a mean), its covering radius, and child links.

The cell's own centroid is also written self-contained inside the cell superfile;
the tree holds the **routing copy**. Nothing reads cell centroids back from the
superfiles for routing — the manifest tree is the routing structure.

### Paged, copy-on-write layout (object storage)

A mutable tree on immutable object storage = copy-on-write paging.

- **Pages.** The tree is cut into content-addressed **pages**, each an S3 object
  of ~a few MB holding a contiguous slice of nodes (a subtree). A page hash is
  its content address.
- **Node → child link = `(page_hash, offset)`.** Content-addressed, so the link
  is also the version of the referenced subtree.
- **Within a page**, node centroids share one Sq8 quantizer (`scale`/`offset`),
  so descent builds one `Sq8ResidualKernel` per page and scores its nodes with no
  per-node quantizer setup.
- **Root.** The manifest stores the current **root page hash** — that single hash
  *is* the tree version.

Why paged: at 10B docs (~2–3M cells, a tens-of-GB tree) a single blob is
impossible to rewrite per commit. Paging makes a commit rewrite only the pages on
its root→leaf path.

## Descent (query, 0 GETs)

1. Load the root page (disk-cached / mmap'd).
2. Build `Sq8ResidualKernel(query, page.scale, page.offset)`; score the page's
   node centroids; keep the best `beam` by distance.
3. Follow the kept nodes' child links into their pages (cached); repeat to the
   leaf level, reaching the `n_probe` nearest **cells**.
4. Pages on the path are disk-cached, so a warm descent issues **0 S3 GETs**.
5. Then issue **one object GET per probed cell**, scan the whole cell, and rerank
   against the Sq8residual payload.

Per-node radius supports best-first pruning (skip a subtree whose bound can't beat
the current `n_probe`-th best). Beam width and `n_probe` are the recall knobs;
acceptance bar is recall@10 ≥ 0.99.

## Commit (write path, partial rewrite)

Every commit that carries vectors updates the hidden manifest:

1. Build the commit's new ≤8 MB cell superfiles (object payload).
2. In memory, over the mmap'd tree pages: descend to the target leaf page(s),
   insert the new cells' Sq8residual centroids.
3. **Split** any page that overflows its size target (B-tree page split).
4. **Copy-on-write the path to root**: each touched page is rewritten with a new
   content hash; parent pointers are updated up the path. Only ~`log_fanout(N)`
   pages are re-PUT (a handful of MB) — never the whole tree.
5. Swap the new **root page hash** into the manifest atomically (the existing OCC
   manifest commit).

At 10B that's ~3–4 pages (~10–16 MB) re-PUT per commit, not tens of GB.

## Rebalance (consolidation)

When a region's cells' bounding spheres overlap with mean degree ≥ τ
(`hot_overlap_groups`), that region is re-clustered: its rows are streamed in
through the mmap reader (one bounded region at a time, never the corpus),
k-means'd (Sq8residual) into tight, disjoint ≤8 MB cells, and swapped in for the
old cells. The tree's affected pages + path are copy-on-write rewritten — same
partial cost as a commit. No split-overflow, no byte-splice merge (which would
mix the cells' independently-trained clusters).

## Distance kernel

One kernel: `Sq8ResidualKernel` (`distance.rs`). fp32 query × Sq8 code leg + i8
residual leg, SIMD via `sq8_dot`. Built once per page (shared quantizer), scores
every node in the page via `distance_with_norm`. The legacy plain-`Sq8Kernel` and
the fp32 rerank path are gone.

## Scale sketch

| docs | ~cells | tree size | per-commit re-PUT | descent |
| ---- | ------ | --------- | ----------------- | ------- |
| 10M  | ~370 (dim 128) – ~2.8K (dim 768) | ~120 KB – ~6 MB | ~log N pages | ~log N cached pages, 0 GETs |
| 10B  | ~2–3M | tens of GB | ~3–4 pages (~10–16 MB) | ~log N cached pages, 0 GETs |

One object GET per probe in all cases.

## Build status / plan

- ✅ Cells: per-commit ≤8 MB IVF superfiles; consolidation re-cluster
  (`recluster_cells`); no eager drain, no split.
- ✅ One codec (Sq8residual), one kernel (`Sq8ResidualKernel`).
- ✅ In-memory routing scorer foundation (`opann::routing::OpannRouting`: flat
  Sq8residual scan → n_probe cells; oracle-tested vs exact fp32 for L2Sq / Cosine
  / NegDot).
- ⬜ Paged tree format: page = content-addressed object; node =
  `[Sq8residual centroid, radius, child links, (leaf) cell ref]`; links =
  `(page_hash, offset)`; manifest holds the root hash. (No backwards compat.)
- ⬜ Commit: write cells + CoW-update the touched tree pages + atomic root swap.
- ⬜ Query: descend the paged tree → n_probe cells → 1 GET each → scan.
- ⬜ Page split + rebalance CoW.
- ⬜ 10M (then 10B) recall@10 ≥ 0.99 + p50 validation.

## Tuning decisions (defaults; revisit via bench)

- **Page size / fanout** — ~4 MB page target; fanout = page capacity ÷ node
  size. A tuning knob, not a design fork.
- **Descent** — beam search scored by `Sq8ResidualKernel`; recall is the
  empirical recall@10 ≥ 0.99 bar (widen beam / `n_probe` if short). Per-node
  radius is a pruning *hint*, not a correctness gate — no admissibility proofs.
- **Quantizer on split** — re-derive per page from that page's own centroids;
  each page is self-contained.
- **Page eviction** — reuse the existing `reader_cache` disk LRU for tree pages,
  same as cells. No separate policy.
