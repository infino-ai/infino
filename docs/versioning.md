<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright The Infino Authors -->

# Versioning & release policy

infino ships as three published artifacts built from one engine:

| Artifact            | Source                | Registry  | Version field                  |
| ------------------- | --------------------- | --------- | ------------------------------ |
| Rust crate `infino` | `Cargo.toml` (root)   | crates.io | `Cargo.toml` `version`         |
| Node binding        | `infino-node/`        | npm       | `infino-node/package.json`     |
| Python binding      | `infino-python/`      | PyPI      | `infino-python/Cargo.toml`     |

All three use [semantic versioning](https://semver.org). This document is the
contract for how their versions relate.

## The rules

1. **Pre-1.0 for now.** The engine stays on major version `0` for roughly the
   first year while the public API settles. Under semver, while major is `0` a
   **breaking change bumps the minor** (`0.1 → 0.2`), not the major. We do not
   bump to `1.0` until we deliberately declare the API stable.

2. **`major.minor` is locked in sync across all three packages.** A release line
   is `0.<minor>`, and every package shares it: Rust `0.3.x`, Node `0.3.y`,
   Python `0.3.z` all sit on the `0.3` line. Never publish a binding on a
   different `major.minor` than the engine.

3. **`patch` is independent per package.** Each package bumps its own patch for
   fixes that only affect it — a Node loader fix, a Python wheel fix, a
   Rust-only bug fix — without touching the others. So patch numbers will
   diverge (e.g. Rust `0.3.4`, Node `0.3.2`, Python `0.3.5`). That is expected
   and fine; only `major.minor` must agree.

4. **The Rust crate is the source of truth for `major.minor`.** The engine
   defines the release line; the bindings follow it. When the crate moves to a
   new minor, the bindings move with it.

## What bumps which number

- **Patch** (`0.3.4 → 0.3.5`) — bug fix, no public API change. Bumped per
  package, independently, whenever that package needs a fix. Registries are
  immutable (you can never republish a version), so **every publish needs a
  fresh patch.**
- **Minor** (`0.3.x → 0.4.0`) — a new feature **or** a breaking API change
  (breaking changes are minor bumps while we are pre-1.0). This is a
  **coordinated** event: bump the Rust crate's minor first, then bring Node and
  Python onto the same minor (resetting their patch to `0`). All three publish
  on the new line.
- **Major** (`0 → 1`) — deferred until we declare the API stable (~a year out).
  A single coordinated release across all three.

## How to release

For a **patch**, release only the package that needs it:

- **Rust** — bump `version` in the root `Cargo.toml`, then push a matching
  `v<version>` tag (e.g. `v0.1.1`). The `Publish crate` workflow
  (`.github/workflows/crate-publish.yml`) asserts the tag matches `Cargo.toml`
  and publishes to crates.io. The engine is the **only** artifact released by a
  tag; a bare `v*` tag is therefore unambiguous. (You can also run the workflow
  manually from the Actions tab — the default is a dry run.)
- **Node** — bump `version` in `infino-node/package.json`, then run the
  `Node publish` workflow (`.github/workflows/node-publish.yml`). `napi
  prepublish` derives the per-platform package versions and rewrites the
  `optionalDependencies` pins from that single field, so the version lives in
  one place.
- **Python** — bump `version` in `infino-python/Cargo.toml`, then run the
  `publish-python` workflow.

For a **minor** (feature or breaking change), do all three in the same release
cycle so `major.minor` never drifts: land the engine change first, then **update
the Node and Python binding code to cover any new or changed engine surface**,
bump the crate and the two bindings to the matching `0.<minor>.0`, and publish
all three. The bindings must never be released on a new minor before their code
actually exposes that minor's engine changes.

## Worked example

| Event                                   | Rust  | Node  | Python |
| --------------------------------------- | ----- | ----- | ------ |
| Initial release                         | 0.1.0 | 0.1.0 | 0.1.0  |
| Node loader bug fix (Node only)         | 0.1.0 | 0.1.1 | 0.1.0  |
| Python wheel fix (Python only)          | 0.1.0 | 0.1.1 | 0.1.1  |
| Engine bug fix (Rust only)              | 0.1.1 | 0.1.1 | 0.1.1  |
| New feature → coordinated minor         | 0.2.0 | 0.2.0 | 0.2.0  |
| Breaking API change (still pre-1.0)     | 0.3.0 | 0.3.0 | 0.3.0  |

Patches diverge between coordinated minors; a minor bump realigns everything on
`0.<minor>.0`.

## Don'ts

- **Don't bump major** before the deliberate `1.0` stability declaration.
- **Don't publish a binding on a `major.minor` the engine isn't on.** Bindings
  never lead the engine's release line.
- **Don't let bindings differ on `major.minor`** from each other.
- **Don't try to republish a version** — registries are immutable. Bump the
  patch instead.
- **Don't add a commit-message-driven release bot** (semantic-release and
  friends). They compute each package's version independently and will break the
  `major.minor` lockstep. Version selection stays a deliberate, coordinated step.

## Open gaps

- **Rust publish automation: done.** The `Publish crate` workflow
  (`.github/workflows/crate-publish.yml`) publishes the engine to crates.io on a
  `v<version>` tag; Node and Python still publish via their own manual workflows.
  Still worth naming a clear owner for the engine's version bumps so the crate
  doesn't fall behind the bindings on the shared release line.
- **No drift guard yet.** A small CI check that asserts the `major.minor` of the
  root `Cargo.toml`, `infino-node/package.json`, and `infino-python/Cargo.toml`
  all agree (patch ignored) would enforce rule 2 cheaply. Recommended.
