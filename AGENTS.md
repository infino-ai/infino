# Infino — notes for AI agents

Read `[CONTRIBUTING.md](CONTRIBUTING.md)` first — it covers prerequisites, build, the demo, test commands, the `make ci` gates, code conventions, and the fork → branch → PR workflow. This file only covers what isn't in there: the repository map, hard boundaries, and traps that aren't obvious from the code.

## Project overview

**Infino is a fast retrieval engine that stores your data on object storage (like Amazon S3) and runs SQL, full-text search, and vector search over it from a single system**. One file (a "superfile") is a valid Apache Parquet file with embedded BM25 + vector indexes spliced into it. The `supertable` layer composes many superfiles into a queryable table with snapshot-isolated reads, append-only writes, and atomic-commit manifest. Object-storage-native; no daemon, no managed service.

For the plain-language tour — what Infino is, the mental model, and how it compares to other systems — see `docs/architecture/overview.md`. For design references, read `docs/architecture/superfile.md` and `docs/architecture/supertable.md` before touching format or manifest code.

## Golden rule

**Correctness > complexity > performance — in that order, always.** A correct, simple solution beats a clever or faster one. Reach for performance only after the code is correct and no more complex than it needs to be. When a change forces a trade-off between these three, the higher priority wins; if you're optimizing at the cost of correctness or simplicity, stop and reconsider.

## Rule precedence

When guidance disagrees, resolve in this order (closest wins):

1. **Explicit user / task instructions** in the current session trump everything below.
2. **Subdirectory `AGENTS.md`** files (when present) take precedence over this file for the scope they cover.
3. **This file** carries the traps and boundaries; `[CONTRIBUTING.md](CONTRIBUTING.md)` carries the general contributor workflow.
4. **Configuration files** (`Cargo.toml`, `Makefile`, `rust-toolchain.toml`, `.github/workflows/`) are the source of truth where they overlap with anything written here — see "Sources of truth" near the end.

## Running a focused test subset

Test binaries are bundled by layer in `Cargo.toml` (`[[test]]` stanzas) to keep link time down — `--test <binary>` picks the layer, the filter narrows within it. Benches are one further step consolidated: a single `[[bench]]` target named `bench`, selected by positional tokens.

```sh
# Run one integration test crate (each binary covers a top-level layer)
cargo test --test superfile fts::
cargo test --test supertable commit::
cargo test --test superfile format::crc_corruption

# Run unit tests in one module
cargo test --lib superfile::vector::

# All benches share one `bench` target; select cells with positional
# tokens after `--`: [tier] [modality] [phase ...] (space-separated)
cargo bench --bench bench -- superfile fts
cargo bench --bench bench -- supertable vector build warm
# Diagnostics are tokens on the same binary:
#   scale | tombstone | update | sql-diag | object-store
cargo bench --bench bench -- tombstone
```

## Code style beyond CONTRIBUTING.md

- **Rayon for CPU, tokio for I/O.** In general this split is the standing concurrency contract for the query and build paths:
  - **tokio owns the I/O waves**: superfile opens, object-store range GETs, sidecar prefetches — `tokio::spawn` / `try_join_all` on the shared multi-thread query runtime so connections pool and fetches overlap. Never build a throwaway per-call runtime on a worker thread (that exact anti-pattern once regressed cold vector search from ~1.1 s to ~3.7–11 s).
  - **rayon owns the CPU waves**: Parquet page decode, BM25 / vector scoring, rerank, encode. Run them on `options.reader_pool` (the configurable pool — not the global rayon pool) via `pool.install(|| … par_iter …)`.
  - **Bridge with a oneshot**: when an async task needs a CPU wave, hand the work to the rayon pool and `await` a `tokio::sync::oneshot` for the result, so tokio workers keep driving I/O instead of blocking under the compute. Don't call `par_iter` (or any long compute) inline in an async fn.
  - If you change where work runs, benchmark before and after (`cargo bench --bench bench -- supertable search` plus the `INFINO_DIAG_QUERY_SQL_OVERHEAD` diagnostic for the SQL resolve path) — a prior change silently moved warm decodes back onto tokio and cost ~5× on `resolve_hits`.
- **No magic numbers.** Numeric (and other opaque) literals that carry semantic meaning must be named `const`s with a short doc-comment, never inlined mid-expression. Declare them at the **top of the file** for runtime code; for test code, at the top of the file or at the top of the relevant test section / module. Trivial values in obvious arithmetic and indexing (`i + 1`, `len - 1`, `x / 2`, index `0`) are exempt.
- **Imports at the top of the file.** All `use` statements live at the top of the file (or, for an inline `#[cfg(test)]` module, at the top of that module) — never function-local, block-scoped, or otherwise inline. A file's full dependency surface should be readable in one place. If you find yourself reaching for a fully-qualified path mid-expression only to avoid an import, add the `use` at the top instead.
- **No code duplication.** Read the surrounding modules *before* writing new code — there is usually an existing helper that already does what you need. Refactor shared logic into one helper rather than copy-pasting; duplicated logic drifts out of sync and is a correctness hazard.
- **No `unsafe` outside the documented surface.** `unsafe` in `src/` is concentrated in four areas: SIMD intrinsic kernels (`superfile/vector/distance.rs`, `vector/sq8_simd.rs`, `vector/quant.rs`, `superfile/fts/tokenize.rs`); memory-mapped / page-advise I/O (`supertable/reader_cache/disk.rs`, `config/`, and the spilled-partition `Mmap` reads in `superfile/fts/builder.rs` + `superfile/vector/spill.rs`); the FTS builder's `bumpalo` lifetime extensions (`FtsBuilder::add_doc` and `intern_term_id_arena`); and bounds-check elisions in the FTS builder's hottest loops (the radix-sort scatter and the `add_doc` dense-tf/partition updates in `superfile/fts/builder.rs`), each guarded by a `SAFETY:` comment proving the index invariant. (Note: `superfile/format/` is *safe* byte parsing — no `unsafe` there.) New `unsafe` requires both `make miri` and `make asan` green plus a clear safety argument in a doc-comment above the block.
- **Visibility hygiene.** Items used only inside the crate are `pub(crate)`, not `pub`. Test-only methods go behind `#[cfg(test)]`, not `#[allow(dead_code)]`. The public API surface is what's re-exported at the crate root (`src/lib.rs`) — see the "Public API surface" section below.
- **State rationale inline in comments.** Don't cite external documents or trackers a reader may not have access to; explain the reasoning directly.
- **Use plain language in source, comments, and commit messages.** Avoid cryptic internal shorthand or tracking tags; describe the change directly.
- **Performance numbers live in `benches/README.md`.** Keep benchmark results there rather than scattered through the codebase.

## Testing instructions

Three lanes beyond `cargo test`:

- **Brute-force oracles** under `tests/` — BM25 top-K is compared against the textbook BM25 formula on planted corpora; full-nprobe IVF is compared against brute-force exact-nearest for L2Sq / Cosine / NegDot. These are the correctness gates; if you touch scoring math or vector distance kernels, the oracles run first.
- **Recall measurement — the acceptance bar is recall@10 ≥ 0.99, full stop.** No change is accepted that drops vector recall@10 below 0.99 on the standard vector bench (10M-row supertable, default config); demonstrate it and report the number in the PR body. The lower floors currently hard-coded in the bench suite (recall@10 ≥ 0.90 / 0.95 in `benches/utils/scale.rs`, and the 0.80 / 0.85 correctness floors) are loose regression tripwires only — **passing them is necessary but not sufficient.** The bar is 0.99.
- `**make miri` + `make asan`** — the memory-safety oracles. Run them. Cost: miri ~100-1000× slower than native; asan 2-3×.
- **Property tests** — `proptest` is in dev-deps; used for round-trip invariants like PFOR encode/decode.

Test deletions require explicit justification.

## Performance

Performance *and* cost are first-class acceptance criteria for every change. A PR that regresses query latency, ingest throughput, or the cost profile (bytes fetched, object-store request count, memory / cache footprint) is rejected unless it buys a correctness fix or a deliberate, documented trade-off. The golden rule still holds — correctness and simplicity come first — but among otherwise-acceptable changes, the one that preserves or improves speed-per-dollar wins.

### What `benches/` covers

One bench target (`[[bench]] name = "bench"`, `harness = false`, custom `main`) drives the whole suite; all measurement logic lives in the `infino-bench-utils` crate under `benches/utils/`. Selection is positional: `cargo bench --bench bench -- [tier] [modality] [phase ...]` with tier `superfile` | `supertable`, modality `fts` | `vector` | `sql`, phase `build` | `warm` | `cold` (omitted ⇒ all). A bare `cargo bench` runs every tier × modality. See `benches/README.md` for the full invocation guide and recorded result tables.

- `**superfile` tier** — single-superfile, in-memory scale (default 1M docs): BM25, IVF + RaBitQ vector, and SQL over one superfile.
- `**supertable` tier** — multi-superfile table over object storage (default 10M docs; backend chosen by `INFINO_BENCH_STORE`, in-process `s3s-fs` emulator by default): the warm/cold table paths for FTS, vector, and SQL.

Diagnostics are standalone programs sharing the same binary (tokens, not separate targets): `scale` (release-profile recall gates), `tombstone`, `update`, `sql-diag`, `object-store`. Scale knobs: `INFINO_BENCH_SUPERFILE_DOCS` / `INFINO_BENCH_SUPERTABLE_DOCS` (plain integers, per tier) and `INFINO_BENCH_WRITERS`.

Recorded numbers live in `benches/README.md`; the structured source of truth is `target/infino-bench/<bench>.json`, written by the report layer after each run (the previous run's file is the delta baseline).

### Running the suite

- **Before any material change to the codebase, run the full bench suite** (`make bench`, i.e. `cargo bench`) and keep the baseline. After your change, re-run and diff against `main` — confirm there is no latency, throughput, *or* cost (bytes / requests / memory) regression. `make bench-quick` (a 100K-doc `superfile fts warm` run) is for fast inner-loop iteration only, never for the final gate.
- This is the same bar the PR checklist enforces: the comparison against `main`, and any intentional trade-off, goes in the PR body.
- Treat a bench run as mandatory — not optional — when you touch scoring math, distance / SIMD kernels, the quantization codecs, the commit / manifest path, or the reader cache.

## Security considerations

- **Crash safety contract.** Committed superfiles must survive `SIGABRT` mid-flight. Verified by tests in `tests/supertable_commit_crash_localfs.rs` (parent spawns aborting child; assertions check superfiles persist). Don't break this contract; if your change touches the commit path, run that test specifically.
- **Don't add new dependencies casually.** Supply-chain surface is part of the public crate's risk profile. New deps require justification in the PR body — what they enable, why no existing dep covers it, and the maintainer / license picture.
- **No secrets in commits.** Agents have committed `.env` files before; the rule applies to them too.
- **Object-store credentials** in tests use mock servers (`s3s` + `s3s-fs`); don't introduce tests that require live cloud credentials.

## Commit message guidelines

- **No `Co-Authored-By: Claude ...` trailer (or any other AI-attribution trailer).** Commit metadata reflects the human author only, even when an agent drafts the message or runs `git commit`.
- **Subject line under ~70 characters.** Body explains *what and why*, not *how* (the diff already shows how).
- **Reference the layer or subsystem in the subject** when reasonable. The recent history is good reference: `fts: leapfrog AND intersection over the skip table`, `superfile/vector: gate scalar/fake test helpers behind target_arch="x86_64"`, `WAL-driven updates + deletes`.
- **No `--no-verify`, no `--no-gpg-sign`, no `-c commit.gpgsign=false`** unless the human author has explicitly requested it. If a pre-commit hook fails, fix the cause; don't bypass.
- **Prefer new commits over amending.** If a hook fails or a review change is needed, create a new commit; do not `git commit --amend` unless explicitly requested.

## Pull request guidelines

- **For non-trivial changes, open an issue first.** Describe the problem and proposed approach so the design can be discussed before code review starts.
- **Bug fixes need a regression test** that fails before the fix and passes after. New features need coverage proportional to the surface added.
- **Run `benches/` against current `main` before submitting.** Every PR must run the benchmark suite against `main` and confirm there are no performance *or* cost regressions. Report the comparison in the PR body, and call out any intentional trade-off explicitly (see `benches/README.md` for where results live).
- **Recall@10 ≥ 0.99 is non-negotiable.** A PR that drops vector recall@10 below 0.99 on the standard bench is rejected regardless of any other merit. The sanity checks baked into the bench code are not the acceptance bar — 0.99 is.
- **Don't force-push to `main` or shared branches.** Force-push your own feature branches if you need to clean up history, but never to `main`.
- **Don't merge with red CI** or with unanswered review comments.
- **PR title and body follow the commit-message conventions above.** No AI-attribution trailers.

## Repository layout

Three core layers (`storage`, `superfile`, `supertable`), the public
`catalog` layer on top, plus a few small support modules:

```
src/
├── lib.rs                 ← crate root (small; declares modules + curated re-exports)
├── catalog/               ← public entry point (connect → Connection → tables, search TVFs)
├── error.rs               ← the single public InfinoError (coarse, #[non_exhaustive])
├── storage/               ← byte-level I/O (StorageProvider trait, LocalFs, S3, Azure)
├── superfile/             ← single-file format (immutable superfiles)
│   ├── builder.rs         ← write path
│   ├── reader.rs          ← read path
│   ├── format/            ← binary layout (footer, CRC)
│   ├── fts/               ← BM25 + posting lists + tokenizer
│   ├── vector/            ← IVF + rotation + quantization codecs
│   └── lazy_source.rs     ← byte-source abstraction for object-store reads
├── supertable/            ← table layer (composes many superfiles)
│   ├── handle.rs          ← Supertable, ArcSwap<Manifest>, single-writer slot
│   ├── writer.rs          ← append + commit
│   ├── manifest/          ← superfile list + Bloom + min/max + partition strategies
│   ├── query/             ← cross-superfile fanout (fts, vector, sql)
│   ├── tombstones/        ← runtime tombstone cache (filtering deleted rows)
│   ├── wal/               ← write-ahead log for update/delete pipeline
│   └── reader_cache/      ← per-process superfile cache
├── config/                ← runtime tuning knobs (StorageSettings + defaults)
├── runtime_bridge.rs      ← sync ↔ async runtime glue
└── test_helpers/          ← shared test fixtures (pub for integration tests)

tests/                     ← integration tests; the two main binaries
                             (superfile, supertable) are bundled by layer
                             in [[test]] stanzas, plus a few standalone
                             top-level test files (e.g. the crash /
                             concurrent-process tests)
benches/                   ← one custom-harness [[bench]] target (`bench`); all
                             measurement logic in benches/utils (infino-bench-utils)
docs/architecture/         ← canonical design references
examples/                  ← runnable examples (start with `cargo run --example demo`)
```

Rule of thumb for landing a change in the right place:


| If the change is about…                     | Edit here                                                             |
| ------------------------------------------- | --------------------------------------------------------------------- |
| BM25 scoring                                | `src/superfile/fts/bm25.rs`                                           |
| Posting list iteration / skip table         | `src/superfile/fts/posting.rs`                                        |
| Vector quantization codec                   | `src/superfile/vector/quant.rs`                                       |
| Vector distance kernel (incl. SIMD)         | `src/superfile/vector/distance.rs`                                    |
| Tokenizer                                   | `src/superfile/fts/tokenize.rs`                                       |
| Partition strategy                          | `src/supertable/manifest/partition.rs`                                |
| Skip pruning (Bloom / min-max / term range) | `src/supertable/manifest/{bloom,aggregates,term_range,list_prune}.rs` |
| Commit / writer slot / handle               | `src/supertable/writer.rs` + `src/supertable/handle.rs`               |
| Tombstones (delete-path / query-filter)     | `src/supertable/{wal,tombstones}/`                                    |
| New storage backend                         | `src/storage/`                                                        |
| File-format byte layout                     | `src/superfile/format/`                                               |
| Catalog / `connect` / `Connection`          | `src/catalog/`                                                        |
| Public error mapping                        | `src/error.rs`                                                        |


## Boundaries

### ✅ Safe to propose and PR (with tests)

- Bug fixes with regression tests
- Documentation, error-message, and example improvements
- Performance optimizations localized to one subsystem
- New implementations of an existing public trait — the trait is the extension contract
- Test additions and refactors confined to one module

### ⚠️ Ask first (open an issue before writing code)

- Changes to the public API surface (anything re-exported at the crate root / tracked in `public-api.txt`)
- Adding a new top-level module under `src/`
- Adding a new direct dependency to `Cargo.toml`
- Changes to the on-disk format (`superfile/format/`, footer layout, blob layout, CRC discipline)
- Changes to commit / manifest semantics (`supertable/manifest/commit.rs`, `supertable/handle.rs`, `supertable/writer.rs`)
- Anything adding `unsafe` outside the existing documented areas (SIMD kernels, mmap/page-advise, the `bumpalo` extension)

### 🚫 Never propose (measured and rejected; don't bring back without genuinely new evidence)

- **Non-Parquet file format** (e.g. a proprietary columnar layout like Lance). Search-on-Parquet is the thesis; ecosystem reuse outweighs a 30-50% storage win.
- **WAL-based ingest** (per-row durability before commit). Rejected as a different architectural model; commit-as-durability-boundary is deliberate. Note: a WAL *does* exist in `src/supertable/wal/` for the **updates/deletes** pipeline — that's orchestration state, not ingest durability. Don't conflate.
- **HNSW graph inside each IVF partition.** Memory cost is 80 MB / 1M docs for an 18% warm-search win; not worth it given our high-`n_cent` + 1-bit-code shape.
- **SPFresh-style in-place IVF rebalance.** Superfiles are immutable by design. Updates = delete + insert via tombstones.
- **Multi-vector / ColBERT-style per-token vectors.** Niche; better as a sidecar pattern than a format primitive.
- `**range_concurrent(&[Range])` storage API.** `LazyByteSource::range` is already `async fn`; callers parallelize with `try_join_all` or `FuturesUnordered`.

If you have a strong reason to revisit any 🚫 item, open an issue with new evidence first; don't open a PR cold.

## Public API surface

The stability boundary is the set of items re-exported at the **crate
root** (`src/lib.rs`) — everything a user reaches as `infino::*`. The
surface is deliberately small: a connection-and-table API over the
storage/superfile/supertable layers, which are themselves internal.

- **Entry points** — `connect(uri)` and `connect_with(uri, ConnectOptions)`, returning a `Connection`.
- `**Connection`** — `create_table`, `open_table`, `drop_table` (logical by default; `purge = true` also deletes the table's storage subtree), `list_tables`, `query_sql`.
- `**Supertable`** (the table handle) — `append`, `update`, `delete`, `schema`, plus the sync search surface. All four search methods (`bm25_search`, `vector_search`, and the unranked `token_match` / `exact_match`) return Arrow rows (`Vec<RecordBatch>`) and take a `projection: Option<&[&str]>` naming the output columns (`_id`, any visible scalar column, or the trailing `score`); `None` returns the engine-native `_id` + `score` pair (no scalar decode — `_id` reads from its dedicated id pages), and materializing row data is an explicit opt-in by naming the columns to decode. The async kernels and the superfile-local hit representation stay on the internal `SupertableReader`; the public methods resolve to the stable `_id` before returning.
- **Supporting types** — `ConnectOptions`, `ColdFetchMode`, `IndexSpec`, `Metric`, `BoolMode`, `VectorSearchOptions`, `MutationStats`, the `InfinoError` enum, and `BUILDER_ID`.

Everything else — `SupertableReader`/`SupertableWriter`, the manifest
and summary types, the storage providers and `StorageProvider` trait,
the whole `SuperfileReader`/builder/byte-source surface — is
`pub(crate)` or test-only and **not** part of the public API. Default
new items to `pub(crate)`; if you add a crate-root `pub` re-export,
justify why it needs to be reachable outside the crate.

The surface is **machine-guarded**: `public-api.txt` is a checked-in
`cargo-public-api` snapshot. `make public-api` fails if the live
surface drifts from it; run `make public-api-update` to regenerate the
snapshot intentionally and review the diff in the PR. Test-only
visibility is handled by the `test_visible!` macro (flips an item
between `pub(crate)` and `pub` under the `test-helpers` feature) so
integration tests can reach internals without widening the shipped
surface.

## Sources of truth

When this file and a config file disagree, the config file wins. Authoritative sources:

- `**Cargo.toml`** — dependencies, lint config (`#![deny(clippy::unwrap_used)]` lives in `lib.rs`), test/bench target declarations (`[[test]]` / `[[bench]]` stanzas), feature flags.
- `**Makefile`** — canonical command set (`check`, `test`, `ci`, `coverage`, `miri`, `asan`, `bench`, `bench-quick`, `clean`).
- `**rust-toolchain.toml`** — the exact stable Rust version pinned for this crate.
- `**.github/workflows/**` — what CI actually runs and fails on.
- `**docs/architecture/superfile.md**` + `**docs/architecture/supertable.md**` — design-level invariants and the rationale behind major choices.

If a section here drifts out of sync with one of those, the config wins and this file is wrong.

## When something doesn't fit any of these notes

Ask. Filing an issue describing the problem before writing code is always welcome — a short written proposal to react to beats a surprise PR.

For project overview and quick-start, see `[README.md](README.md)`.