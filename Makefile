.PHONY: check test \
        coverage coverage-summary \
        bench bench-quick miri asan ci clean

check:
	cargo fmt --check
	cargo clippy --all-targets -- -D warnings

test:
	cargo test

# Coverage (cargo-llvm-cov; install: cargo install cargo-llvm-cov)
coverage:                      # CI gate: ≥90% overall + lcov.info for codecov upload
	cargo llvm-cov --summary-only

coverage-summary:              # quick terminal summary
	cargo llvm-cov --summary-only

# Note: an earlier `coverage-arena` gate was retired when the
# custom MemoryArena it covered was deleted. The remaining
# `unsafe` surface in the FTS stack is one bumpalo lifetime
# extension in `FtsBuilder::add_doc` plus small pockets in
# `format/` byte parsing — covered by the regular `coverage`
# gate plus the `miri` and `asan` lanes below.

# Benchmarks
bench:
	cargo bench

bench-quick:
	INFINO_BENCH_DOC_COUNT=100000 cargo bench --bench superfile_fts

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

# Local "pre-PR" check — same gates CI runs
ci: check coverage
	@echo "✓ ready to PR"

clean:
	cargo clean
	rm -rf target/llvm-cov
	rm -f lcov.info
