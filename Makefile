.PHONY: check fmt test doctest doc \
        coverage coverage-summary \
        bench bench-quick miri asan ci clean \
        public-api public-api-update \
        python-test python-typecheck python-wheel python-examples-test \
        node-test node-build node-verify node-example

# Import layout: group into std / external / crate blocks and merge each
# crate into one `use` tree.
RUSTFMT_OPTS := imports_granularity=Crate,group_imports=StdExternalCrate

check:
	cargo fmt --all -- --check --config $(RUSTFMT_OPTS)
	cargo clippy --all-targets --features test-helpers -- -D warnings

# Apply formatting, including the import-layout rules above.
fmt:
	cargo fmt --all -- --config $(RUSTFMT_OPTS)

# Public-API surface guard. Regenerates the curated public surface and
# fails if it drifts from the committed `public-api.txt` snapshot. The
# surface is taken WITHOUT `test-helpers`, so the internal modules — which
# are `pub` only under that feature — stay off the contract. Any intended
# surface change must land alongside a `make public-api-update` in the
# same commit, so the diff is reviewed like any other contract change.
# Requires the nightly toolchain and `cargo install cargo-public-api`.
public-api:
	cargo public-api --simplified > /tmp/infino-public-api.current
	diff -u public-api.txt /tmp/infino-public-api.current \
	  || { echo "Public API drifted. Review, then run 'make public-api-update'."; exit 1; }

public-api-update:
	cargo public-api --simplified > public-api.txt

test:
	cargo test --features test-helpers

# Coverage (cargo-llvm-cov; install: cargo install cargo-llvm-cov)
coverage:                      # CI gate: ≥90% lines/functions/regions + lcov.info for codecov upload
	cargo llvm-cov --summary-only --features test-helpers --fail-under-lines 90 --fail-under-functions 90 --fail-under-regions 90 --ignore-filename-regex "test_helpers/"

coverage-summary:              # quick terminal summary
	cargo llvm-cov --summary-only --features test-helpers

# Note: an earlier `coverage-arena` gate was retired when the
# custom MemoryArena it covered was deleted. The remaining
# `unsafe` surface in the FTS stack is one bumpalo lifetime
# extension in `FtsBuilder::add_doc` plus small pockets in
# `format/` byte parsing — covered by the regular `coverage`
# gate plus the `miri` and `asan` lanes below.

# Benchmarks
bench:
	cargo bench --features test-helpers

bench-quick:
	INFINO_BENCH_SUPERFILE_DOCS=100000 cargo bench --features test-helpers -- superfile fts warm

# Memory safety oracles for the FTS / format `unsafe` surface.
# The remaining `unsafe` surface is one bumpalo lifetime
# extension in `FtsBuilder::add_doc` plus byte parsing inside
# `format/`. We run miri + asan to validate both.

# miri: Rust's MIR interpreter. Catches LANGUAGE-level UB — bugs that are wrong
#   by Rust's rules even if they happen to work on this hardware. Specifically:
#     * Stacked/Tree Borrows aliasing violations (pointer aliasing model)
#     * Pointer provenance bugs (int-to-ptr round-trips losing metadata)
#     * Reads of uninitialized memory
#     * Misaligned reads/writes (UB on ARM even if they work on x86)
#     * Data races
#   Cost: 100-1000× slower than native; --lib filter keeps it manageable.
#   Install once: `rustup +nightly component add miri`
miri:
	# --lib skips integration-test crates so we don't pay miri's
	# compile-the-world tax on dev-deps. Targets the FTS surface
	# (builder + reader byte handling, format parsing).
	cargo +nightly miri test --lib superfile::fts

# asan: LLVM AddressSanitizer. Catches HARDWARE-level memory errors at execution
#   time — instrumented allocator + shadow memory. Specifically:
#     * Use-after-free
#     * Heap buffer overflow/underflow
#     * Stack buffer overflow / use-after-return / use-after-scope
#     * Memory leaks (LSan bundled in)
#   Cost: 2-3× native; usable on wider surfaces than miri.
#
# Why the cryptic --target flag: sanitizers must be applied at the TARGET level,
#   not host. Without an explicit --target, cargo skips recompiling std with the
#   sanitizer and ASAN misses bugs in std-allocated buffers (which is most of
#   them). The `rustc -vV | sed ...` extracts the host triple (e.g.
#   aarch64-apple-darwin) and forces cargo to rebuild std under instrumentation.
#
# miri vs asan are complementary, not redundant — miri catches Rust-rule
#   violations the CPU is fine with; asan catches real-hardware memory errors
#   miri can't simulate (FFI, real-allocator behavior). Run both.
asan:
	RUSTFLAGS="-Z sanitizer=address" \
	cargo +nightly test --lib \
	  --target $$(rustc -vV | sed -n 's|host: ||p') superfile::fts

# Doctests — runs the README quick example (the crate doc via
# `include_str!`) and any rustdoc examples. No `test-helpers`, so it
# exercises the same curated public API a downstream user sees.
doctest:
	cargo test --doc

# Build the API docs locally, exactly as docs.rs renders them: crate only
# (`--no-deps`), default features, opened in a browser. The landing page is
# the README (lib.rs pulls it in via `include_str!`); the rest is rustdoc
# from the public items' doc comments. Output: target/doc/infino/index.html.
doc:
	cargo doc --no-deps --open

# Python bindings (PyO3 + maturin). Built standalone — `infino-python` is
# excluded from the cargo workspace, so the core crate never needs a
# Python toolchain. These targets are self-contained: they create a
# throwaway venv under `infino-python/.venv` with the build + test deps.

# Build the extension into the venv and run the smoke tests.
python-test:
	python3 -m venv infino-python/.venv
	infino-python/.venv/bin/pip install -q --upgrade pip
	infino-python/.venv/bin/pip install -q maturin pytest pyarrow pandas
	VIRTUAL_ENV=$(CURDIR)/infino-python/.venv infino-python/.venv/bin/maturin develop --locked -m infino-python/Cargo.toml
	infino-python/.venv/bin/python -m pytest infino-python/tests/ -v

# Type-check the package and a sample consumer under `mypy --strict`,
# against the source stubs (no extension build needed). Checking
# `__init__.py` verifies its re-exports match the `_infino` stub; checking
# the sample fails the run if the surface drifts or a `Literal` argument
# widens to plain `str`.
python-typecheck:
	python3 -m venv infino-python/.venv
	infino-python/.venv/bin/pip install -q --upgrade pip mypy
	MYPYPATH=infino-python/python infino-python/.venv/bin/mypy \
		--config-file infino-python/pyproject.toml \
		infino-python/python/infino/__init__.py \
		infino-python/tests/typing/quickstart.py

# Build a release abi3 wheel for the current platform into
# `infino-python/dist/` (one wheel covers CPython >= 3.9).
python-wheel:
	python3 -m venv infino-python/.venv
	infino-python/.venv/bin/pip install -q --upgrade pip maturin
	infino-python/.venv/bin/maturin build --release --locked --out infino-python/dist -m infino-python/Cargo.toml

# Concurrent example notebooks; lower it on smaller runners.
CONCURRENT_EXAMPLE_TESTS ?= 4

# Build the bindings from source and execute every example notebook (a failing
# cell fails the target). Notebooks run in parallel — each uses a distinct
# scratch dir. The venv is a reused throwaway (gitignored).
python-examples-test:
	python3 -m venv infino-python/.venv
	infino-python/.venv/bin/pip install -q --upgrade pip maturin
	VIRTUAL_ENV=$(CURDIR)/infino-python/.venv infino-python/.venv/bin/maturin develop --locked -m infino-python/Cargo.toml
	# Drop infino from the requirements; the from-source build above is what runs.
	grep -v '^[[:space:]]*infino' infino-python/examples/requirements.txt \
		| infino-python/.venv/bin/pip install -q -r /dev/stdin
	infino-python/.venv/bin/pip install -q nbconvert ipykernel
	# Warm the shared embedding model so parallel workers don't race the download.
	PYTHONPATH=infino-python/examples infino-python/.venv/bin/python \
		-c "from _shared.embedding import _get_model; _get_model()" >/dev/null
	@ls infino-python/examples/*/[0-9]*.ipynb | \
	PY=infino-python/.venv/bin/python xargs -P $(CONCURRENT_EXAMPLE_TESTS) -I {} \
		sh -c 'echo "executing {}"; "$$PY" -m nbconvert --to notebook --execute \
			--stdout --ExecutePreprocessor.timeout=900 "{}" >/dev/null'; \
	status=$$?; \
	rm -rf infino-python/examples/*/*_data infino-python/examples/_shared/__pycache__; \
	exit $$status

# Node bindings (napi-rs). Built standalone — `infino-node` is excluded
# from the cargo workspace, so the core crate never needs a Node
# toolchain. These targets require npm + a Rust toolchain on PATH.

# Build the addon (debug) + run the node:test smoke suite.
node-test:
	cd infino-node && npm install && npm run build:debug && npm test

# Build a release addon for the current platform.
node-build:
	cd infino-node && npm install && npm run build

# Run the Node examples as end-to-end smoke tests. Assumes the addon is already
# built (run `make node-test` or `make node-build` first); each example's
# `file:../..` dependency links that build. The hybrid-search-api example runs
# with SMOKE=1 so it self-checks and exits instead of serving forever.
node-example:
	cd infino-node/examples/agent-memory && npm install && node index.mjs
	cd infino-node/examples/hybrid-search-api && npm install && SMOKE=1 node index.mjs

# Verify the published package shape: pack the thin main package + the
# host platform package, install them into a throwaway project, and run a
# roundtrip. Reuses an existing build (REBUILD=1 forces a fresh one).
node-verify:
	cd infino-node && ./scripts/verify-pack.sh

# Local "pre-PR" check — same gates CI runs
ci: check doctest coverage
	@echo "✓ ready to PR"

clean:
	cargo clean
	rm -rf target/llvm-cov
	rm -f lcov.info
