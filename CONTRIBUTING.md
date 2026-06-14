# Contributing to infino

Thanks for your interest in infino. This guide gets you from a fresh
clone to a green pull request.

## Prerequisites

- **Rust** — install via [rustup](https://www.rust-lang.org/tools/install).
  You do not need to pick a version: `rust-toolchain.toml` pins the
  exact stable toolchain and `rustup` installs it automatically the
  first time you build.
- **Nightly Rust** — only required for the memory-safety lanes
  (`make miri`, `make asan`). Install with
  `rustup toolchain install nightly` and, for miri,
  `rustup +nightly component add miri`.
- **cargo-llvm-cov** — only required to run the coverage gate locally:
  `cargo install cargo-llvm-cov --locked`.
- **pre-commit** — optional but recommended for local development. Install via
  [pre-commit.com](https://pre-commit.com/#install) and run `pre-commit install`
  in the repository root. This sets up git hooks to catch style and lint issues
  before they reach CI.

## Build

```bash
git clone git@github.com:infino-ai/infino.git
cd infino
cargo build
```

## Run the demo

The fastest way to see the whole stack work end to end — a superfile
built in memory, queried by BM25 and vector kNN, then read back as a
plain Parquet table:

```bash
cargo run --example demo
```

## Test

```bash
cargo test --features test-helpers          # full suite (integration tests need this feature)
cargo test --features test-helpers <name_substring>   # a single test or module
cargo test --features test-helpers bm25_oracle -- --nocapture   # ...with stdout
```

## Pre-commit hooks (optional but recommended)

After running `pre-commit install`, git hooks will automatically run before each commit to catch:
- Code formatting issues (`rustfmt`)
- Compiler errors (`cargo check`)
- Lint warnings (`clippy`)
- File hygiene (trailing whitespace, line endings, merge conflicts, YAML syntax)

To run them manually without committing:

```bash
pre-commit run --all-files        # check all files
pre-commit run --files <file>     # check specific file(s)
```

## Before you open a pull request

Run the same gates CI runs:

```bash
make ci
```

`make ci` is `make check` + `make coverage`; it must pass before a PR
is merged. The individual targets:

| Target | What it does |
|---|---|
| `make check` | `cargo fmt --check` + `cargo clippy --all-targets -- -D warnings` |
| `make test` | `cargo test` (full suite) |
| `make ci` | `check` + `coverage` — the pre-PR gate |
| `make coverage` | `cargo llvm-cov --summary-only` (≥90% overall) |
| `make bench` / `make bench-quick` | custom harness benches (`bench-quick` for a fast pass) |
| `make miri` | language-level UB detection on the FTS surface (nightly) |
| `make asan` | LLVM AddressSanitizer on the FTS surface (nightly) |

If your change touches any `unsafe` code, also run `make miri` and
`make asan`.

## Code conventions

- **No `.unwrap()`.** The crate is `#![deny(clippy::unwrap_used)]`
  crate-wide (reasserted in `tests/` and `benches/`). Use `?` to
  propagate, or `.expect("invariant: ...")` on paths that are
  infallible by construction. Tests and benches use
  `.expect("description")`.
- **Match the surrounding code** — naming, comment density, and idiom.
- Source comments document the *what*; keep them focused and accurate.

## Workflow

1. Fork the repository and create a branch off `main`.
2. Make your change with tests — we do not merge code without tests.
3. Run `make ci` until it is green.
4. Open a pull request against `main`; CI runs the same gates.

For the design rationale behind a subsystem, read the architecture
references in [`docs/architecture/`](docs/architecture/) —
[`superfile.md`](docs/architecture/superfile.md) (the on-disk format
and single-file reader/builder) and
[`supertable.md`](docs/architecture/supertable.md) (the in-memory
cross-superfile query and manifest layer).
