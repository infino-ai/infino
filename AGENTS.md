# Infino

## Project overview

Infino is a Rust embedded retrieval engine. One file (a "superfile") is a valid Apache Parquet file with embedded BM25 + vector indexes spliced into it. The `supertable` layer composes many superfiles into a queryable table with snapshot-isolated reads, append-only writes, and atomic-commit manifest. Object-storage-native; no daemon, no managed service.

For design references, read `docs/architecture/superfile.md` and `docs/architecture/supertable.md` before touching format or manifest code.

## Rule precedence

When guidance disagrees, resolve in this order (closest wins):

1. **Explicit user / task instructions** in the current session trump everything below.
2. **Subdirectory `AGENTS.md`** files (when present) take precedence over this file for the scope they cover.
3. **This file** carries Infino-specific rules.
4. **Workspace `CLAUDE.md`** (one directory up, outside this repo) carries workspace-wide conventions — no `Co-Authored-By` trailers, public-vs-private hygiene, the umbrella rules. Read when an agent opens at the workspace root; not visible when this repo is cloned standalone.
5. **Configuration files** (`Cargo.toml`, `Makefile`, `rust-toolchain.toml`, `.github/workflows/`) are the source of truth where they overlap with anything written here — see "Sources of truth" near the end.

## Build and test commands

| What | Command |
|---|---|
| Build | `cargo build` |
| Run the end-to-end demo | `cargo run --example demo` |
| Format + lint check (CI gate) | `make check` |
| Run all tests | `make test` (= `cargo test`) |
| Coverage summary | `make coverage` |
| Pre-PR check (format + lint + coverage ≥ 90%) | `make ci` |
| Bench (criterion) | `make bench` |
| Quick bench (sanity only) | `make bench-quick` |
| Memory-safety, Rust UB model (FTS surface) | `make miri` (nightly) |
| Memory-safety, real allocator (FTS surface) | `make asan` (nightly) |
| Clean | `make clean` |

Toolchain is pinned by `rust-toolchain.toml` — `rustup` installs the right stable Rust automatically on first build. For miri/asan: `rustup toolchain install nightly` and `rustup +nightly component add miri`. For coverage locally: `cargo install cargo-llvm-cov --locked`.

### Running a focused subset

```sh
# Run one integration test crate (each binary covers a top-level layer)
cargo test --test superfile fts::
cargo test --test supertable commit::
cargo test --test superfile format::crc_corruption

# Run unit tests in one module
cargo test --lib superfile::vector::

# Single test, with stdout
cargo test bm25_oracle -- --nocapture

# Single bench
cargo bench --bench fts
cargo bench --bench vector
```

Test binaries are bundled by layer in `Cargo.toml` (`[[test]]` stanzas) to keep link time down. Same applies to benches.

## Code style guidelines

- **No `.unwrap()` anywhere — including tests and benches.** The crate sets `#![deny(clippy::unwrap_used)]`. Use `?` to propagate; `.expect("invariant: ...")` for paths that are infallible by construction; `.expect("description")` in tests and benches so a failing message tells you which step broke without counting line numbers.

  ```rust
  // ✗ Never:
  let v = fallible_call().unwrap();

  // ✓ Production — propagate:
  let v = fallible_call()?;

  // ✓ Production — infallible by construction:
  let v = fallible_call().expect("invariant: builder has at least one batch before finish()");

  // ✓ Tests and benches:
  let v = fallible_call().expect("setup: build superfile for vector_search_oracle test");
  ```
- **No `unsafe` outside the documented surface.** The only `unsafe` sites in `src/` are one `bumpalo` lifetime extension in `FtsBuilder::add_doc` plus small pockets of byte parsing in `superfile/format/`. New `unsafe` requires both `make miri` and `make asan` green plus a clear safety argument in a doc-comment above the block.
- **Visibility hygiene.** Items used only inside the crate are `pub(crate)`, not `pub`. Test-only methods go behind `#[cfg(test)]`, not `#[allow(dead_code)]`. The public API surface is what's re-exported from `superfile/mod.rs` and `supertable/mod.rs` — see the "Public API surface" section below.
- **No private-repo citations in source comments.** Source comments must not reference any external private repo or document. State the rationale inline.
- **No internal milestone jargon** (`M1`–`M16`, `003 M5`-style tags) in source, comments, or commit messages. Rewrite to plain language describing the change.
- **`infino/` ships publicly. No references to sibling private repos or predecessor projects** anywhere in source, comments, doc-comments, READMEs, or commit messages. Write `infino/` as if it were the only thing that exists.
- **Performance numbers live in `benches/README.md` and are Infino-only** — no third-party engine mentioned. Head-to-head comparisons against other engines do not belong in this repo.
- **Match surrounding style** for naming, comment density, and idiom. Consistency over personal preference.
- **Comments document the *what* and the *why-when-non-obvious*.** Don't restate the code; explain hidden invariants, subtle constraints, or surprising behavior.

## Testing instructions

Three lanes beyond `cargo test`:

- **Brute-force oracles** under `tests/` — BM25 top-K is compared against the textbook BM25 formula on planted corpora; full-nprobe IVF is compared against brute-force exact-nearest for L2Sq / Cosine / NegDot. These are the correctness gates; if you touch scoring math or vector distance kernels, the oracles run first.
- **Recall measurement** — recall@10 must stay ≥ 0.90 at default options on the standard test corpus.
- **`make miri` + `make asan`** — the memory-safety oracles. Run them when you touch FTS or format code (`src/superfile/fts/` or `src/superfile/format/`). Cost: miri ~100-1000× slower than native; asan 2-3×.
- **Property tests** — `proptest` is in dev-deps; used for round-trip invariants like PFOR encode/decode.

Tests are required on any non-trivial change. Test deletions require explicit justification. CI runs the same gates as `make ci`; PRs do not merge with red CI.

## Security considerations

- **Crash safety contract.** Committed superfiles must survive `SIGABRT` mid-flight. Verified by tests in `tests/supertable_commit_crash_localfs.rs` (parent spawns aborting child; assertions check segments persist). Don't break this contract; if your change touches the commit path, run that test specifically.
- **Don't add new dependencies casually.** Supply-chain surface is part of the public crate's risk profile. New deps require justification in the PR body — what they enable, why no existing dep covers it, and the maintainer / license picture.
- **No secrets in commits.** Agents have committed `.env` files before; the rule applies to them too.
- **Object-store credentials** in tests use mock servers (`s3s` + `s3s-fs`); don't introduce tests that require live cloud credentials.

## Commit message guidelines

- **No `Co-Authored-By: Claude ...` trailer (or any other AI-attribution trailer).** Commit metadata reflects the human author only, even when an agent drafts the message or runs `git commit`. Workspace-wide rule.
- **Subject line under ~70 characters.** Body explains *what and why*, not *how* (the diff already shows how).
- **Reference the layer or subsystem in the subject** when reasonable. The recent history is good reference: `fts: leapfrog AND intersection over the skip table`, `superfile/vector: gate scalar/fake test helpers behind target_arch="x86_64"`, `WAL-driven updates + deletes`.
- **No `--no-verify`, no `--no-gpg-sign`, no `-c commit.gpgsign=false`** unless the human author has explicitly requested it. If a pre-commit hook fails, fix the cause; don't bypass.
- **Prefer new commits over amending.** If a hook fails or a review change is needed, create a new commit; do not `git commit --amend` unless explicitly requested.

## Pull request guidelines

- **Run `make ci` locally first.** It must be green. Don't open PRs expecting CI to be the gate of first resort.
- **Tests required for non-trivial changes.** Bug fixes need a regression test that fails before the fix and passes after. New features need coverage proportional to the surface added.
- **For non-trivial changes, link a design doc.** The team's plan-first workflow expects a design doc to exist for any meaningful change before code review starts. See "Plan-first workflow" below.
- **Don't force-push to `main` or shared branches.** Force-push your own feature branches if you need to clean up history, but never to `main`.
- **Don't merge with red CI** or with unanswered review comments.
- **PR title and body follow the commit-message conventions above.** No AI-attribution trailers. No private-repo references.

## Repository layout

Three-layer architecture; everything in the crate lives in one of these layers:

```
src/
├── lib.rs                 ← crate root (small; declares modules)
├── storage/               ← byte-level I/O (StorageProvider trait, LocalFs, S3)
├── superfile/             ← single-file format (immutable segments)
│   ├── builder.rs         ← write path
│   ├── reader.rs          ← read path
│   ├── format/            ← binary layout (footer, CRC)
│   ├── fts/               ← BM25 + posting lists + tokenizer
│   ├── vector/            ← IVF + rotation + quantization codecs
│   └── lazy_source.rs     ← byte-source abstraction for object-store reads
└── supertable/            ← table layer (composes many superfiles)
    ├── handle.rs          ← Supertable, ArcSwap<Manifest>, single-writer slot
    ├── writer.rs          ← append + commit
    ├── manifest/          ← segment list + Bloom + min/max + partition strategies
    ├── query/             ← cross-segment fanout (fts, vector, sql)
    ├── tombstones/        ← runtime tombstone cache (filtering deleted rows)
    ├── wal/               ← write-ahead log for update/delete pipeline
    └── reader_cache/      ← per-process segment cache

tests/                     ← integration tests, bundled by layer in [[test]] stanzas
benches/                   ← criterion benches, bundled by topic in [[bench]] stanzas
docs/architecture/         ← canonical design references
examples/                  ← runnable examples (start with `cargo run --example demo`)
```

Rule of thumb for landing a change in the right place:

| If the change is about… | Edit here |
|---|---|
| BM25 scoring | `src/superfile/fts/bm25.rs` |
| Posting list iteration / skip table | `src/superfile/fts/posting.rs` |
| Vector quantization codec | `src/superfile/vector/quant.rs` |
| Vector distance kernel (incl. SIMD) | `src/superfile/vector/distance.rs` |
| Tokenizer | `src/superfile/fts/tokenize.rs` |
| Partition strategy | `src/supertable/manifest/partition.rs` |
| Skip pruning (Bloom / min-max / term range) | `src/supertable/manifest/{bloom,aggregates,term_range,list_prune}.rs` |
| Commit / writer slot / handle | `src/supertable/writer.rs` + `src/supertable/handle.rs` |
| Tombstones (delete-path / query-filter) | `src/supertable/{wal,tombstones}/` |
| New storage backend | `src/storage/` |
| File-format byte layout | `src/superfile/format/` |

## Boundaries

### ✅ Safe to propose and PR (with tests)

- Bug fixes with regression tests
- Documentation, error-message, and example improvements
- Performance optimizations localized to one subsystem
- New implementations of an existing public trait — the trait is the extension contract
- Test additions and refactors confined to one module

### ⚠️ Ask first (raise as a plan or an issue before writing code)

- Changes to the public API surface (anything re-exported from `superfile/mod.rs` or `supertable/mod.rs`)
- Adding a new top-level module under `src/`
- Adding a new direct dependency to `Cargo.toml`
- Changes to the on-disk format (`superfile/format/`, footer layout, blob layout, CRC discipline)
- Changes to commit / manifest semantics (`supertable/manifest/commit.rs`, `supertable/handle.rs`, `supertable/writer.rs`)
- Anything touching `unsafe` outside the two documented sites

### 🚫 Never propose (measured and rejected; don't bring back without genuinely new evidence)

- **GPU acceleration (build or search).** Rejected on cost-first grounds. The substring `gpu` / `cuda` / `cublas` should appear nowhere in `src/`, `benches/`, `tests/`, or `Cargo.toml`.
- **Non-Parquet file format** (e.g. a proprietary columnar layout like Lance). Search-on-Parquet is the thesis; ecosystem reuse outweighs a 30-50% storage win.
- **WAL-based ingest** (per-row durability before commit). Rejected as a different architectural model; commit-as-durability-boundary is deliberate. Note: a WAL *does* exist in `src/supertable/wal/` for the **updates/deletes** pipeline — that's orchestration state, not ingest durability. Don't conflate.
- **HNSW graph inside each IVF partition.** Memory cost is 80 MB / 1M docs for an 18% warm-search win; not worth it given our high-`n_cent` + 1-bit-code shape.
- **SPFresh-style in-place IVF rebalance.** Segments are immutable by design. Updates = delete + insert via tombstones.
- **Multi-vector / ColBERT-style per-token vectors.** Niche; better as a sidecar pattern than a format primitive.
- **`range_concurrent(&[Range])` storage API.** `LazyByteSource::range` is already `async fn`; callers parallelize with `try_join_all` or `FuturesUnordered`.

If you have a strong reason to revisit any 🚫 item, write a plan with new evidence; don't open a PR cold.

## Plan-first workflow

For any change that isn't a small bugfix or doc edit:

1. Draft a design doc in the team's plan tracker (separate from this repo). Naming convention: `NNN_short_slug.md` with the next free `NNN`.
2. Get review on the plan before implementation begins.
3. Reference the plan number in the commit message subject or body. Do not include private-repo paths inside source comments.

Trivial bug fixes, doc fixes, and one-file refactors are exempt and can go straight to PR.

## Public API surface

The stability boundary is what's re-exported from these two module roots:

- **`src/superfile/mod.rs`** — `SuperfileReader`, `SuperfileBuilder`, `VectorSearchOptions`, `OpenOptions`, `LazyByteSource`, error types, and the free functions `bm25_search` / `vector_search`.
- **`src/supertable/mod.rs`** — `Supertable`, `SupertableReader`, `SupertableWriter`, `SupertableOptions`, `Manifest`, `SuperfileEntry`, `SuperfileUri`, `FtsSummary`, `VectorSummary`, storage providers (`LocalFsStorageProvider`, `S3StorageProvider`, `StorageProvider` trait), and error types.

Anything not re-exported through one of those module roots is internal. Default new items to `pub(crate)`; if you add a `pub` item, justify why it needs to be reachable outside the crate.

## Sources of truth

When this file and a config file disagree, the config file wins. Authoritative sources:

- **`Cargo.toml`** — dependencies, lint config (`#![deny(clippy::unwrap_used)]` lives in `lib.rs`), test/bench target declarations (`[[test]]` / `[[bench]]` stanzas), feature flags.
- **`Makefile`** — canonical command set (`check`, `test`, `ci`, `coverage`, `miri`, `asan`, `bench`, `bench-quick`, `clean`).
- **`rust-toolchain.toml`** — the exact stable Rust version pinned for this crate.
- **`.github/workflows/`** — what CI actually runs and fails on.
- **`docs/architecture/superfile.md`** + **`docs/architecture/supertable.md`** — design-level invariants and the rationale behind major choices.

If a section here drifts out of sync with one of those, the config wins and this file is wrong.

## When something doesn't fit any of these notes

Ask. The team has a strong design-doc culture and would rather have a one-page plan to react to than a surprise PR. Filing an issue describing the problem before writing code is always welcome.

For human-contributor-facing guidance (the linear "clone → build → test → PR" narrative), see [`CONTRIBUTING.md`](CONTRIBUTING.md). For project overview and quick-start, see [`README.md`](README.md).
