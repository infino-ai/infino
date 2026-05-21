# Superfile architecture

The superfile is infino's on-disk segment format: **one file = a valid
[Apache Parquet](https://parquet.apache.org/docs/) file plus embedded
[BM25](https://en.wikipedia.org/wiki/Okapi_BM25) and vector indexes**.
Vanilla Parquet engines ([DataFusion](https://datafusion.apache.org/),
[DuckDB](https://duckdb.org/), [pyarrow](https://arrow.apache.org/docs/python/))
read it as a regular columnar table; infino-aware tooling
(`SuperfileReader`) unlocks the embedded indexes for full-text and
nearest-neighbor search.

This document is the long-form reference for the format, the
algorithms it embeds, and the rationale for every non-obvious
design decision. Components that compose with the superfile but
are out of scope for this format —
[the supertable](./supertable.md) that joins many superfiles into
one logical table, compaction of small superfiles, update / delete
via tombstones, distributed multi-node operation, SQL UDF / TVF
integration — are deliberately deferred and noted in
[Non-goals](#goals-and-non-goals).

## Table of contents

- [Goals and non-goals](#goals-and-non-goals)
- [Format spec](#format-spec)
  - [Top-level byte layout](#top-level-byte-layout)
  - [`inf.*` KV metadata](#inf-kv-metadata)
  - [FTS blob layout](#fts-blob-layout)
  - [Vector blob layout](#vector-blob-layout)
- [API surface](#api-surface)
  - [Build path](#build-path)
  - [Read path](#read-path)
  - [Top-level search wrappers](#top-level-search-wrappers)
- [Subsystem design](#subsystem-design)
  - [FTS pipeline](#fts-pipeline)
    - [BM25 with Lucene-style IDF](#bm25-with-lucene-style-idf)
    - [Tokenization](#tokenization)
    - [Term dictionary: FST](#term-dictionary-fst)
    - [Posting codec: PFOR-delta + skip table](#posting-codec-pfor-delta--skip-table)
    - [Single-term search: BlockMaxWAND on the skip table](#single-term-search-blockmaxwand-on-the-skip-table)
    - [Multi-term OR search: WAND vs MaxScore vs Block-Max-MaxScore](#multi-term-or-search-wand-vs-maxscore-vs-block-max-maxscore)
    - [Per-doc work optimizations](#per-doc-work-optimizations)
  - [Vector pipeline](#vector-pipeline)
    - [IVF + 1-bit RaBitQ + full-precision rerank](#ivf--1-bit-rabitq--full-precision-rerank)
    - [Why not HNSW or full-precision IVF](#why-not-hnsw-or-full-precision-ivf)
    - [Cluster-contiguous storage](#cluster-contiguous-storage)
  - [Format and footer surgery](#format-and-footer-surgery)
  - [Allocator strategy](#allocator-strategy)
  - [Ingestion durability and `commit()`](#ingestion-durability-and-commit)
    - [The five options considered](#the-five-options-considered)
    - [Multi-threaded ingestion](#multi-threaded-ingestion)
- [Performance](#performance)
- [Test strategy](#test-strategy)
- [Known limitations](#known-limitations)
- [Phase 2 plans](#phase-2-plans)

## Goals and non-goals

**Goals.**

- **Single-file segment.** Parquet body, FTS blob, vector blob, and
  rewritten footer all live in one byte sequence. A query needs at
  most three range fetches against object storage (footer + at most
  one FTS posting region + at most one vector cluster region) — see
  [single-RTT cold path discussion](#cluster-contiguous-storage).
- **Open-format compatibility.** A superfile is a valid Parquet file.
  DataFusion / DuckDB / pyarrow read it as a regular columnar table.
  infino-specific functionality is layered via `inf.*` KV metadata;
  Parquet readers ignore unknown KV keys per spec.
- **Search-native.** BM25 multi-column full-text and IVF-based vector
  kNN are first-class, not a layer above an opaque catalog. Both
  blobs are produced and consumed through the same `SuperfileBuilder`
  / `SuperfileReader` pair.
- **Latency targets.** Single-term BM25 search p50 in the
  microseconds at 1M docs; multi-term OR p50 in the low
  milliseconds even on the worst Zipfian shapes; vector kNN p50
  ≤ 20 ms at 10M × 384 / recall@10 ≥ 0.90.
- **Caller-controlled durability.** A successful `commit()` is the
  durability barrier; lax callers ride on a configurable
  threshold-flush cadence, strict callers call `commit()` per batch.
  No separate write-ahead log. See
  [Ingestion durability and `commit()`](#ingestion-durability-and-commit).
- **Bounded build-time memory.** Same threshold-flush mechanism caps
  the in-memory builder buffer at `commit_threshold_size_mb` MiB.

**Non-goals (deliberately deferred).**

- **Cross-segment query.** A reader that joins many superfiles
  into one logical view lives one layer up in
  [`supertable`](./supertable.md); the single-segment
  `SuperfileReader` stays focused on one file.
- **Compaction.** Merging small superfiles into larger ones is a
  future addition. The superfile format itself is immutable once
  written.
- **Updates and deletes.** Tombstones + compaction-removes (the
  Lucene / [Iceberg](https://iceberg.apache.org/) pattern) is a
  future addition.
- **Distributed, multi-node.** Out of scope.
- **SQL UDFs / TVFs.** First-class BM25 / vector / hybrid SQL
  integration is a future addition.
- **Phrase queries, faceting, snippet generation, query parsers.**
  Out of scope for v1; possible additions later. The format leaves
  room for them (e.g. positions could be added behind a flag bit
  in the FTS metadata header).
- **Buffer-reader path** — querying the in-memory builder buffer
  before `commit()`. Deliberately not shipping; see [the
  ingest-model rationale](#the-five-options-considered).

## Format spec

### Top-level byte layout

```text
[ Parquet body                     ]   ← row groups: scalar columns
[ FTS blob       (optional)        ]   ← inverted index over LargeUtf8 columns
[ Vector blob    (optional)        ]   ← IVF + 1-bit RaBitQ + f32 rerank
[ Parquet footer (rewritten)       ]   ← Thrift-encoded metadata + inf.* KV
[ footer length (u32 LE) ][ "PAR1" ]
```

The footer is rewritten after the blobs are appended so that
`key_value_metadata` (a Parquet primitive any reader exposes) carries
the offsets of the embedded blobs. See
[Format and footer surgery](#format-and-footer-surgery) for the
implementation.

### `inf.*` KV metadata

Every superfile's footer carries these keys; absent keys mean "this
subsystem is not present in this file". Parquet readers ignore
unknown keys per spec, which is what makes the file readable as
plain Parquet.

```text
inf.format            = "infino-superfile"
inf.format_version    = "1.0.0"
inf.id_column         = "<user-chosen UInt64 column>"
inf.n_docs            = "<u64 string>"
inf.builder           = "infino/<crate-version>+<git-sha>"
inf.fts.offset        = "<u64>"   (only if FTS blob present)
inf.fts.length        = "<u64>"   (only if FTS blob present)
inf.fts.columns       = "[ ... per-column FTS config JSON ... ]"
inf.vec.offset        = "<u64>"   (only if vector blob present)
inf.vec.length        = "<u64>"   (only if vector blob present)
inf.vec.columns       = "[ ... per-column vector config JSON ... ]"
```

Verified by [`tests/parquet_compat.rs`](../../tests/parquet_compat.rs)
running DataFusion against a planted-row superfile.

### FTS blob layout

```text
header (56 bytes:
  magic = "INFFTS01"
  version, n_columns, n_docs, n_terms_total
  fst_offset, postings_offset, doc_lens_table_offset)
FST term dictionary           + CRC32C
postings region               + CRC32C
doc-lens directory            + CRC32C
per-column doc-lens arrays    (each + its own CRC32C)
```

- **FST keys** are `<column_name>\x1F<term>` (ASCII Unit Separator
  U+001F as the separator). The single-FST design lets one term
  appear in multiple columns without per-column dictionaries.
- **Posting lists** are PFOR-delta encoded in 128-doc blocks. Each
  `(column, term)` entry has a skip table with per-block max BM25
  upper bound for [BlockMaxWAND](#single-term-search-blockmaxwand-on-the-skip-table)
  pruning.
- **Doc lengths** (token counts) are stored per column as `u16` LE,
  with a directory entry per column carrying that column's `avgdl`
  and offset into the doc-lens region. `avgdl` is the per-column
  `total_tokens / n_docs` and is the one fixed-point value
  (`avgdl_x1000`) we store at build time so reader
  bit-for-bit reproducibility doesn't depend on float ordering.

The CRC32C placement is byte-level: every CRC-protected region has a
trailing `u32` LE checksum computed via the Castagnoli polynomial
([CRC32C](https://datatracker.ietf.org/doc/html/rfc3720#appendix-B.4)).
The default `SuperfileReader::open` / `VectorReader::open` paths
verify every CRC eagerly; callers running on trusted storage
(serverless workers, multi-tenant per-request opens) can pass
`OpenOptions { verify_crc: false }` to skip the scans — at 1M × 384
that's the difference between ~132 ms (default) and ~1 ms cold
open. See
[`tests/crc_corruption.rs`](../../tests/crc_corruption.rs) for the
byte-flip rejection oracle (default path).

### Vector blob layout

```text
outer header (32 bytes:
  magic = "INFVEC01"
  version, n_columns, n_docs, directory_offset)
directory     (n_columns × 64 bytes:
  column_id, dim, n_cent, metric_id, rot_seed,
  sub_offset, sub_length, summary_offset, summary_length)
              + CRC32C
per-column subsections   (each:
  magic = "INFVECC1"
  IVF centroids
  per-cluster summary
  cluster index
  1-bit RaBitQ codes
  full f32 vectors
  doc_ids
  all in cluster-contiguous order
              + CRC32C)
trailing CRC32C of the entire blob
```

**Cluster-contiguous storage** means a query that probes K clusters
reads K hot regions — ~5 KB each at `dim=384` — instead of scattering
DRAM / object-storage requests. The full `f32` vectors are kept
(~1.03× storage overhead vs codes-only) because rerank is what gives
us recall@10 ≥ 0.90 at default tuning; the 1-bit codes are an
estimator-shortcut, not a replacement.

## API surface

### Build path

```rust
use infino::superfile::builder::{
    BuilderOptions, FtsConfig, SuperfileBuilder, VectorConfig,
};

// Configure schema + index columns.
let opts = BuilderOptions::new(schema, "doc_id",
    vec![FtsConfig { column: "title".into() }],
    vec![VectorConfig { column: "embedding".into(), dim: 384,
                        n_cent: 1024, rot_seed: 7,
                        metric: Metric::Cosine }],
    Some(Arc::new(AsciiLowerTokenizer)));

let mut b = SuperfileBuilder::new(opts)?;
b.add_batch(&batch_a, &[&vectors_a])?;
b.add_batch(&batch_b, &[&vectors_b])?;
let segment: Vec<u8> = b.finish()?;   // consumes builder, returns one superfile
```

`SuperfileBuilder` is a single-segment primitive: one builder
produces one superfile blob on `finish()`. Multi-segment +
durability + crash-safety live one layer up in
[`supertable`](./supertable.md), which drives N builders
(one per rayon shard) per commit and atomically swings the
manifest pointer once all segments are durable.

### Read path

```rust
use infino::superfile::SuperfileReader;
use bytes::Bytes;

let bytes: Bytes = std::fs::read("my.superfile")?.into();
let reader = SuperfileReader::open(bytes)?;
//   ↑ validates magic + every CRC eagerly. Once this returns Ok,
//     all queries against the reader are free of per-call I/O
//     besides whatever the underlying Bytes layer (mmap / network)
//     does on first read.
//
// Opt-in fast open for trusted storage:
//   let reader = SuperfileReader::open_with(
//       bytes,
//       OpenOptions { verify_crc: false },
//   )?;

reader.schema();             // &Arc<Schema>
reader.n_docs();             // u64
reader.fts();                // Option<&FtsReader>
reader.vec();                // Option<&VectorReader>
reader.parquet_bytes();      // &Bytes — sub-slice that DataFusion can register as-is
```

### Top-level search wrappers

```rust
use infino::superfile::{bm25_search, bm25_search_multi, vector_search,
                         VectorSearchOptions};
use infino::superfile::fts::reader::BoolMode;

let hits = bm25_search(&reader, "title", "rust async", 10, BoolMode::Or)?;
let hits = bm25_search_multi(&reader,
    &[("title", 1.0), ("body", 0.5)],
    "rust async", 10, BoolMode::Or)?;
let hits = vector_search(&reader, "embedding", &query_f32, 10,
                         VectorSearchOptions::default())?;
```

`VectorSearchOptions::default()` ships `nprobe=8`, `rerank_mult=20`
— measured to recover ≥0.9 recall@10 on typical IVF setups without
caller tuning. Rationale in [Vector pipeline](#vector-pipeline).

## Subsystem design

This section is the technical core. Each subsystem starts with its
problem statement, lists the alternatives considered, and explains
why the chosen design wins on the relevant axis. **Reading these
"alternatives rejected" callouts is the fastest way to understand
why infino's perf numbers look the way they do.**

### FTS pipeline

```text
                       BUILD                                         READ

   add_doc(col, doc_id, text)              SuperfileReader::open(bytes)
        │                                          │
        ▼                                          │  validate magic + CRCs
   AsciiLowerTokenizer                             ▼
        │   per-doc tf-counting HashMap         FtsReader
        │   keyed by &str in bumpalo arena         │
        │   (Bump::reset per add_doc)              │
        ▼                                          │
   postings[col_id]: FxHashMap<                    │
       Box<str>,                                   │
       PostingAcc { Vec<(doc_id, tf)> }            │
   >                                               │
        │                                          │
        ▼  finish()                                ▼  search("col", terms, k, mode)
   per-(col, term):                            (see Multi-term OR dispatch
     PFOR-delta 128-doc blocks                  for Bmm / WandBmw / Exhaustive
     + skip table (block_max_bm25)              routing)
        │                                          │
        ▼                                          ▼
   single FST over <col>\x1F<term>             BM25 top-k heap
   keyed by lex order                              │
        │                                          ▼
        ▼                                      Vec<(local_doc_id, score)>
   FTS blob bytes (= header + FST +
   postings + doc_lens, each CRC-protected)
```

#### BM25 with Lucene-style IDF

We use the BM25 ranking function from Robertson & Zaragoza, [The
Probabilistic Relevance Framework: BM25 and
Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf):

```text
score(d, q) = Σ_{t ∈ q} idf(t) · tf_factor(tf(t, d), dl(d), avgdl)

idf(t)      = ln( 1 + (N - df(t) + 0.5) / (df(t) + 0.5) )       // Lucene form
tf_factor   = tf · (k1 + 1) / (tf + k1 · (1 - b + b · dl/avgdl))
k1, b       = 1.2, 0.75                                          // Lucene defaults
```

The two important per-doc factors are precomputed at index open:

- `dl_norm_k1[d] = k1 · (1 - b + b · dl[d] / avgdl)` (per-doc
  constant that lives in the score denominator).
- `idf_x_k1p1 = idf · (k1 + 1)` (per-cursor constant in the
  numerator).

Inner-loop scoring reduces to `score = idf_x_k1p1 · tf / (tf +
dl_norm_k1[d])` — a multiply, an add, and a divide. The
[`bm25::score_simd_x4`](../../src/superfile/fts/bm25.rs) variant
SIMD-packs four cursors (= four lanes of `(idf_x_k1p1, tf)`) per
scored doc using the [`wide`](https://docs.rs/wide/) crate.

**Rejected alternatives.**

- *BM25F or BM25+ multi-column normalization.* Cross-column TF/IDF
  normalization is real complexity (per-column avgdl, per-column
  field weights interacting with cross-field TF saturation). v1
  ships **`most_fields`** weighted-sum semantics via
  `bm25_search_multi`: each column scores independently, the
  caller-supplied weights blend them. Real BM25F can be added in a
  follow-up if a workload justifies the complexity.
- *Probabilistic IR alternatives (DPH, DLH, language models).*
  BM25 is the de facto baseline and what the brute-force oracle
  expects; switching gains < 1% NDCG on TREC-style benchmarks at
  the cost of a tunable parameter surface. Not worth it.

#### Tokenization

[`AsciiLowerTokenizer`](../../src/superfile/fts/tokenize.rs) splits
on whitespace + ASCII punctuation, lowercases ASCII letters, and
emits each token through a zero-allocation
`tokenize_each(text, &mut FnMut(&str))` callback. No stemming, no
stop-word removal, no Unicode-NFC normalization in v1.

The per-doc tf-counting `HashMap<&str, u32>` is backed by a
[`bumpalo::Bump`](https://github.com/fitzgen/bumpalo) **owned by
the `FtsBuilder` and reused across every `add_doc` call**. At the
top of each call,
[`Bump::reset`](https://docs.rs/bumpalo/latest/bumpalo/struct.Bump.html#method.reset)
invalidates the previous doc's token bytes while keeping the
largest chunk allocated — steady-state docs allocate in place
without a system-allocator round trip. Drop order within `add_doc`
is `tf_per_term` before the borrow of `bump` ends, so the `&str`
keys never outlive their backing bytes; the
[`std::mem::transmute`](https://doc.rust-lang.org/std/mem/fn.transmute.html)
that widens the bump lifetime is the *only* remaining `unsafe`
block in `src/`, and is signed off by both
[`make miri`](../../Makefile) (Stacked Borrows) and `make asan`
(AddressSanitizer).

**Rejected alternatives.**

- *Per-term `String` allocations.* One heap allocation per token per
  doc. Profiled at ~30% of `add_doc` cost; bumpalo eliminates it.
- *Unicode segmentation* via [`unicode-segmentation`](https://crates.io/crates/unicode-segmentation).
  Adds dependency weight and ~3× tokenization cost for synthetic
  ASCII workloads. Deferred to a follow-up tokenizer that can be
  registered alongside the ASCII one (the `Tokenizer` trait
  already supports this).
- *Stemmer (Porter / Snowball).* Adds language coupling and locks
  the index to a stem version. v1 ships unstemmed; stemming is a
  natural per-tokenizer addition.

#### Term dictionary: FST

The term dictionary is a finite-state transducer built via
[BurntSushi's `fst` crate](https://github.com/BurntSushi/fst); see
[burntsushi.net/transducers](https://blog.burntsushi.net/transducers/)
for the data structure walkthrough. FST keys are
`<column>\x1F<term>` (ASCII Unit Separator); FST values are 64-bit
words encoding `(metadata_offset << 8) | flags` so a single FST
lookup gets us straight to the `(column, term)`'s posting metadata.

**Rejected alternatives.**

- *Per-column FSTs.* Multiplies the lookup cost by `O(n_columns)`
  on multi-column queries. The `<column>\x1F<term>` separator on
  one combined FST is `O(1)` per lookup.
- *Hash table (rustc-hash, ahash, etc.).* Loses ordered iteration
  (which the FST gives us for free for prefix queries — not
  shipped in v1, but the format leaves room). Hash-table memory
  is also higher than the FST's compressed transducer.
- *[Marisa-trie](https://github.com/s-yata/marisa-trie) or other
  succinct-trie crates.* Equivalent functionality; `fst` has
  better Rust-native integration and zero-copy mmap support
  (relevant when we move the on-disk format to mmap). No
  performance difference on our workload.

#### Posting codec: PFOR-delta + skip table

Posting lists for each `(column, term)` pair are stored as a chain
of 128-doc blocks. Each block is encoded with **PFOR-delta**
(Patched Frame-of-Reference + delta-coding) per [Zukowski et
al. 2006](https://homepages.cwi.nl/~boncz/snb-challenge/chapter-3.pdf):
doc IDs are delta-coded, the delta width is picked per-block via
the [`bitpacking`](https://github.com/quickwit-oss/bitpacking)
crate, and term frequencies are stored as a parallel
varint-or-bitpacked array.

Each `(column, term)`'s metadata header (`TERM_META_SIZE = 32`
bytes) is followed by a **skip table**: one entry per block carrying
the block's `last_doc_id`, byte offset into the term's posting bytes,
and `block_max_bm25` (the maximum BM25 score any single doc in this
block can produce, fixed-point as `u32 × 1000`). The skip table is
what enables [BlockMaxWAND](#single-term-search-blockmaxwand-on-the-skip-table)
pruning: a block whose `block_max_bm25` is below the kth-best score
in the heap can be discarded without decoding.

**Rejected alternatives.**

- *FOR (Frame-of-Reference) without patching.* PFOR's "patches"
  handle outliers within a block (one large delta among many small
  ones doesn't force the whole block to the worst-case bit width).
  On Zipfian-distributed corpora the patching reduces posting size
  by ~20%.
- *[Roaring bitmaps](https://roaringbitmap.org/).* Designed for
  set-membership operations on sparse `u32` sets — beautifully
  cache-friendly for set intersection. But our posting lists carry
  per-doc `tf` values, not just doc-id sets, and roaring's
  hybrid run/array/bitmap layout makes per-doc decode expensive
  when we're scanning. PFOR-delta wins on streamed scans of (doc,
  tf) pairs.
- *Variable-length-encoded (VLE) deltas only* (no bitpacking). VLE
  is simpler (~30 lines of code) but ~3× slower to decode than
  bitpacked PFOR-delta on modern CPUs because branchy varint loops
  defeat speculation. The
  [`bitpacking`](https://crates.io/crates/bitpacking) crate
  compiles to AVX2 / NEON intrinsics, which is what makes the
  bitpacked path win on the inner loop.

#### Single-term search: BlockMaxWAND on the skip table

[`search_single_term_bmw`](../../src/superfile/fts/reader.rs)
walks the skip table and decodes only the blocks whose
`block_max_bm25` could possibly land in the top-k heap. With Zipfian
distributions this prunes 95–99% of blocks for common terms.

The algorithm is straightforward:

1. Set `threshold = 0`. Walk blocks left to right.
2. If `heap.len() == k && block_max_bm25 ≤ threshold`: **skip the
   whole block** — no doc in it can beat the heap's worst entry.
3. Otherwise: decode the block, score every doc, push into the
   `(score, doc_id)` heap, update `threshold = heap.peek().0` once
   the heap reaches size `k`.

See [BlockMaxWAND, Ding & Suel 2011](http://engineering.nyu.edu/~suel/papers/bmw.pdf)
for the original paper.

#### Multi-term OR search: WAND vs MaxScore vs Block-Max-MaxScore

For multi-term OR queries the key question is **which docs can we
discard without scoring them**. There are three classical algorithms
and we ship the latter two:

- **[WAND](https://www.cse.lehigh.edu/~brian/pubs/2003/CIKM/wand.pdf)
  (Broder et al. 2003):** sort cursors by current `doc_id`; pick a
  *pivot* term-prefix whose `Σ term_max ≥ threshold` (any doc that
  can beat threshold must include at least one of these prefix
  cursors). Advance the pivot cursor; rinse repeat.
- **[BlockMaxWAND (BMW)](http://engineering.nyu.edu/~suel/papers/bmw.pdf)
  (Ding & Suel 2011):** WAND augmented with per-block UBs from the
  skip table. Tightens the pivot bound from "term max" to "block
  max at pivot's would-be position".
- **[MaxScore](https://dl.acm.org/doi/10.1145/215206.215359) (Turtle
  & Flood 1995):** partition cursors into *essential* (those whose
  `term_max` we can't ignore for a top-k hit) and *non-essential*.
  Walk essential cursors only; probe non-essentials lazily.
- **Block-Max-MaxScore (BMM)** ([Petri, Moffat, Mackenzie & Culpepper
  2019](https://dl.acm.org/doi/10.1145/3357384.3358045)): the BMM
  augmentation of MaxScore, same shape as BMW augments WAND.

**What's in the codebase.** Three OR algorithms are implemented,
selectable via the
[`OrAlgo`](../../src/superfile/fts/reader.rs) enum:

| Variant | Function | Production routing |
|---|---|---|
| `Bmm` | [`run_max_score_bmm`](../../src/superfile/fts/reader.rs) | **Default** — `dispatch_multi_term_or` always routes here |
| `WandBmw` | [`run_wand_bmw`](../../src/superfile/fts/reader.rs) | Retained for bench comparison only |
| `Exhaustive` | [`run_exhaustive_union`](../../src/superfile/fts/reader.rs) | Bench-only; see callout below |

The dispatcher
([`dispatch_multi_term_or`](../../src/superfile/fts/reader.rs))
**always routes to BMM** because, with the optimizations below,
BMM beats WAND+BMW on every measured query shape — including the
ones the textbook says BMW should win.

```text
   search(column, terms, k, mode=Or)
     │  terms.is_empty() → []
     │  terms.len() == 1 → search_single_term_bmw  (BMW skip-table walk)
     ▼
   dispatch_multi_term_or(column_id, terms, k)
     │
     ▼  build_term_cursors → Vec<TermCursor>
     │
     └─→  run_max_score_bmm  (always)

   search_with_algo_for_bench(column, terms, k, algo)   # #[doc(hidden)]
     │   bench harness only — lets us compare all three under
     │   identical inputs without touching the dispatcher.
     ▼
     │  algo == Bmm        → run_max_score_bmm
     │  algo == WandBmw    → run_wand_bmw
     │  algo == Exhaustive → run_exhaustive_union
```

The exhaustive walk exists because the supertable bench surfaced
one specific shape where it narrowly wins (prefix expansion over
10+ very-rare terms, in parallel mode: 40.2 ms vs BMM's 54.0 ms at
10M × 8 segments). The dispatcher does not route to it because the
same algorithm regresses mid-rank uniform-UB shapes by 50–80% —
see the cost model in
[`run_exhaustive_union`'s doc comment](../../src/superfile/fts/reader.rs).

**Sub-range entry point for supertable fan-out.** When the
supertable layer has more reader-pool threads than segments, it
slices each segment into doc-id sub-ranges and runs
[`search_or_range_pretokenized`](../../src/superfile/fts/reader.rs)
on each sub-range in parallel. Internally that delegates to
[`run_max_score_bmm_range`](../../src/superfile/fts/reader.rs),
which seeks every cursor to `doc_id_start` and breaks the outer
loop once the next candidate reaches `doc_id_end`. The per-segment
BMM bookkeeping is unchanged; only the cursor seek + termination
condition differ. See
[supertable.md § query fan-out](./supertable.md#query-fan-out)
for how the supertable layer uses this.

The BMM implementation has three layers of perf engineering on top
of textbook MaxScore:

1. **`f_essential == 1` block-batch fast path.** Once the heap is
   filled and threshold rises enough that only one cursor stays
   essential (the steady state for wide-UB queries and for
   heap-warmed similar-UB queries), the inner loop decodes a block
   of the essential cursor's posting list and scores every doc in
   the block inline — no per-doc cursor sort, no per-doc pivot
   logic, just a tight loop. The outer-loop overhead amortizes
   over ~128 docs per block instead of 1 doc per iteration.

2. **Per-doc UB bail.** Bound each candidate doc's max possible
   score by `essential_score + sum_others_term_max`; if even this
   can't beat the heap threshold, skip the non-essential probe +
   heap update entirely. Most rank-100 docs in the
   `three_wide` query (`term00001 + term00050 + term00100`) score
   around 2.7 alone, so with `others_term_ub ≈ 2.5` their max is
   ~5.2 — well below the steady-state threshold of ~6.5, and the
   bail fires for ~70% of docs. This single optimization moved
   `three_wide` from 4× *slower* than a naive multi-term walk to
   3.5× *faster*.

3. **Shallow-advance bail in `f >= 2` paths.** For multi-essential
   regimes the bail uses `inspect_block_max` — the per-block UB at
   the candidate's would-be block in each non-essential cursor —
   which is amortized O(1) via a separate inspect-block pointer
   that tracks block positions without decoding. If the tightened
   UB already can't beat threshold we skip the deep `skip_to`
   advances on non-essentials (which are the dominant per-doc cost
   on the otherwise-tight inner loop).

**Rejected alternatives.**

- *Heap-of-cursors plain WAND without block-max augmentation.*
  Skips by term-max alone. Vastly weaker pruning than BMW: doesn't
  use the skip table's per-block UB at all. Measured ~5× slower
  than BMW on `three_similar`. Easy first thing to ship; we
  iterated past it.
- *Always WAND+BMW.* The straightforward initial choice. BMW's
  pivot bound becomes loose when many query terms have similar
  UBs (the pivot thrashes). Once BMM acquired the f=1 fast path
  + per-doc bail and the shallow-advance UB, it dominated WAND
  on every shape we tested — including the wide-UB queries
  where WAND has the textbook advantage.
- *Conjunctive (AND-only) execution.* AND is supported via the
  full-decode + HashMap intersection path
  ([`run_and`](../../src/superfile/fts/reader.rs)) and stays
  there because AND query cardinality is bounded by the smallest
  posting list — the inner loop fits in L1 and dynamic pruning
  doesn't help much.
- *[Variable-Block-Max WAND](https://dl.acm.org/doi/10.1145/2911451.2914746)
  (Mallia et al. 2017).* Tighter block UBs at the cost of more
  metadata storage. The fixed-128 block size + skip-table format
  is intentionally simple to keep the posting layout
  straightforward to encode and decode. Possible follow-up if a
  workload demands tighter pruning.

#### Per-doc work optimizations

The single biggest perf bug found during the multi-term-OR
optimization pass was that `skip_to` was **re-decoding the current
PFOR block on every call** even when the target was within the
already-decoded block. With ~300K skip_to calls per multi-term
query, that was 300K wasted PFOR decodes.

[`TermCursor::skip_to`](../../src/superfile/fts/reader.rs) now
splits into:

- A small `#[inline(always)]` fast path that detects "target lies in
  the current decoded block" and just scans `pos` forward.
- A `#[cold]` `skip_to_cross_block` helper that handles the actual
  block-boundary case (advance `current_block` via the skip table,
  decode the new block, scan).

Same idea with `inspect_block`, the lightweight block-pointer used
by BMW UB lookups: it tracks the block containing pivot_doc / the
current candidate without decoding — a standard block-cursor /
posting-cursor split that amortizes UB lookups to O(1) for
monotonically-advancing pivots.

This single fix accounts for these post-fix improvements (1M docs):

| query | before | after |
|---|---|---|
| `three_wide` | 2.54 ms | **1.28 ms** |
| `three_similar` | 24.2 ms | **5.94 ms** |
| `five_term` | 14.8 ms | **10.0 ms** |

See `git log -p src/superfile/fts/reader.rs | grep -A40 "skip_to within-block fast path"`
for the diff.

### Vector pipeline

#### IVF + 1-bit RaBitQ + full-precision rerank

Per column:

1. **Random rotation.** A deterministic-seeded Gram-Schmidt rotation
   matrix is applied to all vectors at build time. Rotation
   distributes information across coordinates, making the 1-bit
   sign-coding step in step 3 close-to-uniformly informative across
   dimensions. See [RaBitQ, 2024](https://arxiv.org/abs/2405.12497).
2. **IVF (inverted-file index).** 5 iterations of Lloyd's k-means
   cluster the rotated vectors into `n_cent` clusters via
   [rayon](https://github.com/rayon-rs/rayon)-parallel centroid
   recomputation. `n_cent` is conventionally `~sqrt(n_docs)` —
   `1024` at 1M docs, `4096` at 10M.
3. **1-bit RaBitQ codes.** Each rotated vector becomes a `dim`-bit
   string (sign of each coordinate). Quantization is lossy but
   produces a 32× compression of the vector buffer relative to f32
   storage.
4. **Cluster-contiguous reorganization.** Vectors are reordered so
   that all vectors in cluster `c_0` come before all vectors in
   `c_1`, etc. The 1-bit codes, the full f32 vectors, and the
   per-row `doc_id` array are all reordered identically.
5. **Per-cluster summary.** For each cluster we store a small
   summary (centroid + offset + length) used at query time to do
   the centroid scan.

At query time:

1. **Centroid scan.** Compute the user's chosen distance
   ([L2sq / Cosine / NegDot](../../src/superfile/vector/distance.rs))
   between the query and every centroid; take the top `nprobe`.
2. **1-bit estimator over `nprobe` clusters' codes.** For each
   probed cluster, estimate distance via popcount of the XOR of
   query sign-bits and stored sign-bits. Faster than full-precision
   distance but lossy.
3. **`select_nth` shortlist.** Take the `k * rerank_mult` lowest
   estimated distances.
4. **Full-precision rerank.** Compute the full-f32 distance between
   the query and the shortlisted vectors. The result is the final
   top-k.

#### Why not HNSW or full-precision IVF

[HNSW (Malkov & Yashunin 2018)](https://arxiv.org/abs/1603.09320)
gives slightly higher recall (~0.97 vs ~0.92 for our IVF at default
options) but requires the HNSW graph + edge list alongside the
vectors. Cold-fetch from object storage of an HNSW segment needs
multiple range GETs (graph header → entry-point → traversed nodes
→ vectors). That's incompatible with the
[single-RTT cold path](#cluster-contiguous-storage) goal.

Full-precision IVF (`f32` distance to every vector in probed
clusters, no quantization) trades scan throughput for recall. At
1M × 384 the rerank-shortlist + 1-bit-estimator path measures
**5.3 ms p50** vs ~16 ms for full-precision IVF on the same probe
budget — the 3× saving is the difference between under and over
the 20 ms target.

**Rejected alternatives.**

- *HNSW.* Rejected on cold-path grounds (above). Worth revisiting if
  workloads converge on small (< 100k) corpora where the recall gap
  matters more than cold-fetch latency.
- *FAISS-style PQ (product quantization)* with multiple codebooks.
  Higher recall than 1-bit RaBitQ at the same compression ratio,
  but ~5× more bookkeeping (codebook tables per column, asymmetric
  distance computation). Possible follow-up.
- *Two-stage IVF + IVF (hierarchical).* Recursive clustering for
  very large corpora. Not worth the complexity at < 100M scale; the
  centroid scan on 4096 centroids is already L1-resident.
- *Skip the rotation step.* Sign-coding without rotation works on
  pre-rotated embeddings (e.g.
  [BGE-M3](https://huggingface.co/BAAI/bge-m3)) but degrades
  rapidly on non-rotated ones. The rotation is cheap at build time
  and free at query time (we store rotated vectors directly), so we
  always do it.

#### Cluster-contiguous storage

Each cluster's data — its 1-bit codes, full f32 vectors, and
`doc_ids` — lives in **one contiguous byte range** within the
column subsection. A query that probes K clusters reads K hot
regions, each ~5 KB at `dim=384`. The single-RTT cold path means
that against object storage, a query is **one footer read + one
range GET per probed cluster** — no graph walk, no per-doc
pointer chasing.

Compare to:

- *HNSW* needs the graph (typically ~100 MB at 1M × 384), the
  entry point, and vectors. A cold fetch is graph + multiple
  per-node lookups.
- *PQ.* The codebook + the codes + the vectors. Three regions,
  even if cluster-contiguous, plus the per-codebook lookup table
  must be loaded.

The trade is **~1.03× storage overhead** (we keep the f32 vectors
in addition to the codes, since rerank reads them) for **~3×
faster query** at the cluster-probe step.

### Format and footer surgery

[`format/footer.rs`](../../src/superfile/format/footer.rs) does the
post-write splice that turns a vanilla Parquet write into a
superfile:

1. [`ArrowWriter`](https://docs.rs/parquet/latest/parquet/arrow/arrow_writer/struct.ArrowWriter.html)
   writes the row groups + footer to an in-memory `Vec<u8>`.
2. We parse the footer, locate the byte where it starts, and
   truncate.
3. We append the FTS + vector blobs.
4. We patch `key_value_metadata` with the `inf.*` offsets and
   re-encode the footer via `parquet::format` Thrift types.
5. We append the new footer + footer-length + magic.

The result is byte-for-byte a valid Parquet file with two opaque
runs of bytes between the last row group and the footer, and
`inf.*` KV pointing at them. DataFusion sees the file as a normal
Parquet table and ignores the unknown `inf.*` keys (verified by
[`tests/parquet_compat.rs`](../../tests/parquet_compat.rs)).

**Rejected alternatives.**

- *Parquet `ColumnIndex` / `OffsetIndex` extensions.* Parquet
  doesn't define a way to carry an opaque blob in the footer's
  column-index region. Embedding our blobs as opaque KV bytes
  + body splice is the only reader-compatible path.
- *Sidecar files.* A `my.parquet` + `my.parquet.fts` + `my.parquet.vec`
  trio is what most search-on-Parquet stacks do (Iceberg, Delta).
  Requires the storage layer to atomically write three files
  (or a manifest); cold fetch is three GETs minimum. Embedding
  in one file is one GET to bootstrap and is immune to "the
  sidecar got deleted but the parquet didn't" failure modes.
- *Reverse splice (FTS + vector before the row groups).* Would
  require rewriting the row-group offsets in the footer's
  `RowGroup` metadata. The current footer-only patch is much
  simpler.

### Allocator strategy

Two tiers:

- **[mimalloc](https://github.com/microsoft/mimalloc)** as
  `#[global_allocator]` (gated off under `cfg(miri)` because miri
  can't call into the C runtime). ~5–15% across the board on
  heap-heavy paths. Services the per-column
  [`FxHashMap`](https://docs.rs/rustc-hash/latest/rustc_hash/type.FxHashMap.html)`<Box<str>, PostingAcc>`
  tables in `FtsBuilder` — `postings[column_id]` maps term →
  `PostingAcc { list: Vec<(u32, u32)> }`. Lookups in the
  steady-state per-doc loop hash only the term bytes via
  `FxHashMap::get_mut(&str)` (relying on
  [`Box<str>: Borrow<str>`](https://doc.rust-lang.org/std/borrow/trait.Borrow.html)),
  not the `<col_name>\x1F<term>` byte string the earlier
  single-map layout hashed every time.
- **[`bumpalo::Bump`](https://github.com/fitzgen/bumpalo)** held by
  the `FtsBuilder` and **reused** across every `add_doc` call (see
  [Tokenization](#tokenization) for the per-call `Bump::reset`
  pattern). Holds the per-doc tf-counting `HashMap`'s `&str` keys.

#### Allocator choice: mimalloc-backed Vec per term (chained-arena rejected)

An earlier iteration of the FTS builder shipped a custom
chained-chunk `MemoryArena` for posting accumulation, on the
basis that prior published benchmarks showed a 5× advantage over
per-list `Box::new` on glibc malloc. We re-ran that comparison
against three alternative backends on a synthetic workload that
mirrors `FtsBuilder::add_doc`'s shape — 1M docs, 134.8M
`(doc_id, tf)` pushes — and the custom arena was the **slowest**
of the four:

| Backend                                          | Push + iter | Throughput     |
|--------------------------------------------------|------------:|----------------|
| `MemoryArena` + `PostingList` *(was)*            | 1.24 s      | 109 M elem/s   |
| `bumpalo::Bump` + `BumpVec` per term             | 486 ms      | 277 M elem/s   |
| **`Vec<(u32, u32)>` per term, mimalloc** *(now)* | **353 ms**  | **382 M elem/s** |
| `Vec<(u32, u32)>` per term + capacity freelist   | 354 ms      | 381 M elem/s   |

The freelist variant ties plain `Vec` (mimalloc's small-class
freelist already absorbs the realloc churn). Bumpalo cuts the
custom arena's cost roughly in half but still trails plain `Vec`
because each bump-vec push goes through bump's own bookkeeping.

End-to-end `fts_build` at 1M dropped 12.65 s → 9.91 s (**−21.6%**;
79K → 101K docs/s/core).

**Why the custom arena lost.** Two assumptions in the original
comparison were wrong by the time we re-measured:

1. The "5× over `Box::new`" baseline was glibc malloc, not
   mimalloc. With mimalloc as `#[global_allocator]` per-allocation
   cost collapses to roughly an L1 round trip plus a CAS — the
   per-class freelists eat the page-walk cost.
2. `ArenaPtr` resolution per push (slab-id lookup + offset add +
   bounds check + raw-pointer materialization) is not free; on the
   inner-loop critical path it costs more than mimalloc's per-pair
   amortized allocation.

**What got deleted along with the arena.** ~1500 LOC: the entire
`src/superfile/fts/arena.rs` module (slab manager, `MemoryArena`,
`ArenaPtr`, `Inner`, `PostingChunk`, `PostingList`, `PostingIter`,
intern-bytes helper), `tests/arena_property.rs` (49 property tests),
and `benches/arena_alloc.rs`. Plus ~30 LOC of `unsafe`, a
99%-coverage carve-out for the arena module, and the bulk of the
`make miri` / `make asan` lanes' load — they were aimed almost
entirely at the arena's pointer-arithmetic surface.

**What's left in `unsafe` after the arena removal.** Exactly one
block in `src/`: a
[`std::mem::transmute`](https://doc.rust-lang.org/std/mem/fn.transmute.html)
in
[`FtsBuilder::add_doc`](../../src/superfile/fts/builder.rs) that
extends the lifetime of a [`bumpalo`](https://github.com/fitzgen/bumpalo)-allocated
`&str` to `&'static str` so it can key the per-doc tf
HashMap. The soundness rests on Rust's reverse-declaration drop
order — the HashMap drops before the Bump, so the keys never
outlive their backing bytes. Both `make miri` (Stacked Borrows) and
`make asan` (AddressSanitizer) sign off on this.

### Ingestion durability

The superfile layer is a single-segment primitive: one
`SuperfileBuilder` accumulates batches and `finish()` produces
one self-contained superfile blob. Durability and crash-safety
are the supertable layer's job — see
[`supertable.md` § commit pipeline](./supertable.md) for the
atomic-rename pointer protocol that bounds the durability gap
and survives `kill -9` between PUTs.

```rust
b.add_batch(...)?;        // accumulates in-memory
b.add_batch(...)?;        //   …
let segment = b.finish()?;   // consumes builder, returns one superfile
```

#### Why no in-builder WAL / threshold flush

We evaluated five ingest models against memory / durability /
search-during-ingest / operational complexity before settling on
the current shape. The choice landed at the supertable layer
(threshold-flush in-memory buffer + caller-controlled commits),
not in the builder — but the alternatives are worth recording
because the same trade-offs come up whenever a new ingest
surface is proposed.

- **A. WAL + queryable buffer** (Elasticsearch / Cassandra style).
  Translog or commitlog written per-`add_batch`; in-memory buffer
  stays queryable until the next refresh interval. **Rejected**:
  adds a separate log format and operational surface (sized to
  retention, sync schedule, replay on startup, fsync semantics).
  Buffer-reader path also breaks the
  [open-format moat](#goals-and-non-goals) — a buffer-reader query
  has no Parquet equivalent for DataFusion / DuckDB to call.
- **B. Per-`add_batch` commit** (ClickHouse `INSERT` style).
  Every `add_batch` produces one superfile. Maximum durability,
  worst storage overhead (one Parquet footer + tiny FTS / vector
  blobs per call). **Rejected**: storage-amplification factor of
  100× on small batches; compaction load is impractical.
- **C. Long-running buffer until explicit commit**.
  No threshold flush; memory grows without bound until caller
  explicitly commits. **Rejected**: callers who forget to commit
  hit OOM; observed during e2e validation (50+ GB RAM at 10M
  docs in the no-flush path).
- **D. Threshold-flush in-memory buffer + caller-controlled
  durability. ← CHOSEN, at the supertable layer.** Bounded
  memory (the threshold), bounded durability gap (the threshold
  OR the caller's cadence — whichever fires first), one ingest
  surface for both lax and strict callers. Multi-segment output
  composes with future cross-segment-query and small-segment-
  merging layers. Matches the
  [Iceberg](https://iceberg.apache.org/) / Lucene
  segment-per-commit shape.
- **E. WAL + threshold-flush hybrid.** Both in-memory buffer (with
  threshold) AND a translog. **Rejected**: highest operational
  complexity (everything that A and D have, combined), and the
  durability-window improvement over D is marginal — D's
  caller-controlled mode already gives strict callers per-batch
  durability.

#### Multi-threaded ingestion

The supertable's commit shape composes naturally with
[rayon](https://github.com/rayon-rs/rayon) parallelism: shard
the input doc range across N workers, each running its own
`SuperfileBuilder` and emitting one self-contained superfile.
Per commit, the supertable produces N superfiles, one per shard
worker, then atomically swings the manifest pointer once all N
are durable.

The criterion bench `superfile_fts_build` under `benches/`
measures both single-threaded and multi-threaded build paths;
the multi-threaded shard scales near-linearly with the writer
pool's thread count up to the corpus-bound floor where Parquet
+ FTS encode dominates.

**Rejected alternatives.**

- *Lock-protected shared FtsBuilder.* One global `Arc<Mutex<FtsBuilder>>`
  serialized by all worker threads. Defeats the parallelism — measured
  ~1.05× speedup vs single-threaded due to lock contention on the
  per-term posting Vecs.
- *Sharding by term-hash* across N workers' FtsBuilders. Token
  dispatch cost (per-token hash + send across a channel) eats the
  parallelism gain. The naive doc-range shard is simpler and
  faster.

## Performance

Absolute runtime numbers — single-threaded and multi-threaded
build throughput, search p50 / p99 across the BM25 query shapes
and vector kNN at calibrated recall — are produced by the
in-tree criterion harness under `benches/`. Re-run them with
`cargo bench` after any change to the FTS or vector pipeline.

Where the wins come from architecturally:

- **Search-side FTS** — the perf engineering layered onto WAND+BMW
  + MaxScore+BMM, covered in
  [Multi-term OR search](#multi-term-or-search-wand-vs-maxscore-vs-block-max-maxscore)
  and [Per-doc work optimizations](#per-doc-work-optimizations).
- **FTS ingestion** — the mimalloc-backed `Vec<(u32, u32)>` per
  term (see [Allocator strategy](#allocator-strategy)) and the
  rayon fan-out (see
  [Multi-threaded ingestion](#multi-threaded-ingestion)).
- **Vector** — the cluster-contiguous storage layout (see
  [Cluster-contiguous storage](#cluster-contiguous-storage)) plus
  a zero-copy SIMD rerank that scores candidates directly from
  byte slices into `f32x8` via `bytemuck::try_cast_slice`,
  removing a `Vec<f32>` allocation + scalar byte-decode per
  candidate at `rerank_mult = 1024` (10K candidates per query).
- **Vector build** — returning k-means' final-iter assignments
  alongside its centroids (skipping a redundant assignment pass)
  and parallelizing the k-means update step's per-cluster sums.

## Test strategy

| Suite | Catches |
|---|---|
| [`tests/superfile/pipeline.rs`](../../tests/superfile/pipeline.rs) | end-to-end superfile (FTS + vector + Parquet) build → read |
| [`tests/superfile/fts/pipeline.rs`](../../tests/superfile/fts/pipeline.rs) | FTS sub-component build → search |
| [`tests/superfile/fts/brute_force_oracle.rs`](../../tests/superfile/fts/brute_force_oracle.rs) | textbook BM25 oracle: top-k matches the brute-force formula on a 60-doc planted corpus |
| [`tests/superfile/vector/pipeline.rs`](../../tests/superfile/vector/pipeline.rs) | vector sub-component build → kNN |
| [`tests/superfile/vector/brute_force_oracle.rs`](../../tests/superfile/vector/brute_force_oracle.rs) | exact-NN oracle for L2Sq / Cosine / NegDot |
| [`tests/superfile/format/parquet_compat.rs`](../../tests/superfile/format/parquet_compat.rs) | DataFusion reads superfile as plain Parquet (planted-row counts, GROUP BY, predicate pushdown match) |
| [`tests/superfile/format/crc_corruption.rs`](../../tests/superfile/format/crc_corruption.rs) | byte-flip rejection across every CRC region |
| in-module `#[cfg(test)] mod tests` | per-module unit tests across `superfile::{builder, fts, vector, format}` |

Plus criterion benchmarks under `benches/` and the memory-safety
lanes:

- [`make miri`](../../Makefile) (Stacked Borrows + UB detection)
  — passing on the FTS surface, zero violations.
- [`make asan`](../../Makefile) (LLVM AddressSanitizer) — passing
  on the FTS surface, zero memory errors.

Both lanes target `superfile::fts` since that's where the only
remaining `unsafe` block lives (the bumpalo lifetime extension
in `FtsBuilder::add_doc` — see
[Allocator strategy](#allocator-strategy)).

The brute-force BM25 oracle catches scoring-math bugs that
planted-ground-truth tests can't: a self-consistent BM25 bug
(e.g. wrong avgdl handling) can produce correct relative ranking
on the planted set while disagreeing with the actual BM25
formula. Brute-force computes the formula by direct construction
with no code shared with the optimized BMW / BMM walks.

## Known limitations

- **Single-segment reader** (`SuperfileReader`). Cross-segment
  query lives one layer up in
  [`supertable`](./supertable.md); the superfile itself stays a
  single-segment primitive.
- **No streaming queries during build** (queryable == durable —
  see [the five options considered](#the-five-options-considered)).
  Records are queryable iff they're in a completed superfile.
- **No phrase / position queries.** v1 ships unweighted unordered
  bag-of-words BM25.
- **`AsciiLowerTokenizer` only.** Unicode segmentation, stemming,
  stop words are deferred. The `Tokenizer` trait is open for
  extension.
- **k-means is single-machine and synchronous.** At 10M × 384,
  full-batch Lloyd's takes ~27 minutes per build; mini-batch
  k-means or reservoir-sampled init are possible follow-ups.
- **Build memory peak is governed by the **commit_threshold**.**
  Default 1 GiB; callers running tight on RAM can drop it.
- **Memory-safety lanes.** `make miri` and `make asan` both run
  clean on the FTS surface (106 tests each). See the
  memory-safety lanes table in
  [`README.md`](../../README.md).

## Phase 2 plans

The superfile format is designed to compose with four future
layers on top of the shipping [`supertable`](./supertable.md):

- **Compaction** — small-segment merging into larger superfiles.
- **Updates / deletes** — tombstones plus compaction-removes,
  Lucene / [Iceberg](https://iceberg.apache.org/) style.
- **Distributed scale-out** — multi-node coordination.
- **Extended SQL** — first-class BM25 / vector / hybrid SQL UDFs
  and TVFs.
