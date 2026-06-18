# Versioning

Infino ships several artifacts from this repository, plus ecosystem packages in
their own repos. **Each artifact is versioned independently** with standard
SemVer; version numbers do **not** correlate across languages.

## Artifacts

| Artifact | Registry | Version source of truth |
| --- | --- | --- |
| `infino` — the engine | crates.io | root `Cargo.toml` |
| `infino` — Python binding | PyPI | `infino-python/pyproject.toml` |
| `infino` — Node binding | npm | `infino-node/package.json` (the per-platform `infino-<triple>` packages share this version) |

## Independent versioning

- Each artifact bumps its own version when **it** changes, following SemVer.
- **Pre-1.0 (`0.x`):** a **minor** bump may include breaking changes; a
  **patch** is fixes only. At **`1.0`** an artifact switches to strict SemVer
  (major = breaking). The 1.0 transition happens **per-artifact**, when that
  surface has stabilized — the engine, the Python binding, and the Node binding
  do not have to reach 1.0 together.
- Numbers are per-artifact and **do not align**. `infino` (npm) `0.3.0`,
  `infino` (PyPI) `0.1.2`, and the `infino` crate `0.2.0` are unrelated.

## Why this is safe here

The bindings **statically embed the engine**: they build against the in-repo
core through a path dependency and bundle the compiled library. A published
binding is therefore **self-contained** — there is no separate engine version it
resolves at runtime, and so **no cross-artifact compatibility matrix to
maintain**. (The binding crates are `publish = false`; only the engine crate is
published to crates.io.)

## Traceability

Because the published numbers don't correlate, use these to answer "what engine
build is inside this binding?":

- **`BUILDER_ID`** — exposed by each binding (and the crate); it carries the
  engine's build identifier, independent of the package version.
- The per-package **CHANGELOG**, which records what changed in each release.

## Bindings may differ in features

Bindings are **not** required to be at feature parity. A capability can land in
one binding before another (e.g. Node before Python). Each binding's README /
CHANGELOG is the source of truth for its surface at a given version.

## Releasing

- **One artifact per release.** Tag with a per-artifact prefix so CI publishes
  the right thing:
  - `crate-vX.Y.Z` → `cargo publish` (crates.io)
  - `py-vX.Y.Z` → maturin wheels (PyPI)
  - `node-vX.Y.Z` → napi prebuilds (npm)
- Each publish workflow reads **its own** version file. There is **no**
  "versions must match across artifacts" check.
- A binding's own `Cargo.toml` version tracks **its** published version
  (cosmetic), not the engine crate's.
