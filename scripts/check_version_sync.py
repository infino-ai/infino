#!/usr/bin/env python3
"""Guard that the three published packages stay on one release line.

infino ships three artifacts from this repo — the Rust crate (root
`Cargo.toml`), the Node binding (`infino-node/package.json`), and the
Python binding (`infino-python/Cargo.toml`). Per `docs/versioning.md`,
their `major.minor` is locked in sync (the crate defines the line) while
each package bumps its own patch independently.

The check fails (non-zero exit) when:
  (a) a binding sits on a different `major.minor` than the crate
  (b) an `infx-*` platform pin in `infino-node/package.json`
      `optionalDependencies` differs from the package's own version
  (c) a Cargo.lock records a different version for its own package than
      the matching Cargo.toml — a bump that skipped the lockfile breaks
      the `--locked` publish at release time

Usage:
  check_version_sync.py                          # verify; non-zero exit on drift
  check_version_sync.py --release-version 0.2.1  # also assert a version about
                                                 # to publish sits on the
                                                 # crate's release line
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_VERSION = re.compile(r'version\s*=\s*"([^"]+)"')


def manifest_version(toml_path):
    """The `[package]` version of a Cargo.toml (regex; stdlib-only on 3.9)."""
    section = None
    for raw in toml_path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("["):
            section = line
        elif section == "[package]":
            m = _VERSION.match(line)
            if m:
                return m.group(1)
    return None


def locked_version(lock_path, name):
    """The version Cargo.lock records for package `name`."""
    lines = [line.strip() for line in lock_path.read_text().splitlines()]
    for i, line in enumerate(lines[:-1]):
        if line == f'name = "{name}"':
            m = _VERSION.match(lines[i + 1])
            if m:
                return m.group(1)
    return None


def release_line(version):
    """major.minor — the release line a version sits on."""
    return ".".join(version.split(".")[:2])


def check(root, release_version=None):
    """Verify the version contract under `root`; return error strings.

    `release_version` is a version about to be published (e.g. the value a
    publish workflow was dispatched with); it must sit on the crate's
    release line.
    """
    root = Path(root)
    errors = []

    crate = manifest_version(root / "Cargo.toml")
    if crate is None:
        errors.append("no [package] version found in Cargo.toml")
    else:
        if (release_version is not None
                and release_line(release_version) != release_line(crate)):
            errors.append(
                f"release version {release_version} is on the "
                f"{release_line(release_version)} line but the crate is on "
                f"{release_line(crate)} ({crate}); major.minor is locked "
                f"across the three packages (docs/versioning.md)"
            )
        crate_locked = locked_version(root / "Cargo.lock", "infino")
        if crate_locked != crate:
            errors.append(
                f"Cargo.lock records infino {crate_locked} but Cargo.toml "
                f"says {crate}; regenerate the lockfile (releases publish "
                f"with --locked)"
            )

    node_manifest = json.loads((root / "infino-node" / "package.json").read_text())
    node = node_manifest.get("version")
    if node is None:
        errors.append("no version field found in infino-node/package.json")
    else:
        if crate is not None and release_line(node) != release_line(crate):
            errors.append(
                f"infino-node/package.json is on the {release_line(node)} "
                f"line ({node}) but the crate is on {release_line(crate)} "
                f"({crate}); major.minor is locked across the three "
                f"packages (docs/versioning.md)"
            )
        for pin, pinned in sorted(node_manifest.get("optionalDependencies", {}).items()):
            if pin.startswith("infx-") and pinned != node:
                errors.append(
                    f"infino-node/package.json pins {pin} at {pinned} but "
                    f"the package version is {node}; the platform pins "
                    f"track the package version"
                )
        # The node crate is unpublished (`publish = false`), but its version
        # must track package.json so no committed version file lies.
        node_crate = manifest_version(root / "infino-node" / "Cargo.toml")
        if node_crate != node:
            errors.append(
                f"infino-node/Cargo.toml records {node_crate} but "
                f"infino-node/package.json says {node}; the node crate "
                f"manifest tracks the npm package version"
            )
        else:
            node_crate_locked = locked_version(
                root / "infino-node" / "Cargo.lock", "infino-node"
            )
            if node_crate_locked != node_crate:
                errors.append(
                    f"infino-node/Cargo.lock records infino-node "
                    f"{node_crate_locked} but infino-node/Cargo.toml says "
                    f"{node_crate}; regenerate the lockfile"
                )

    python = manifest_version(root / "infino-python" / "Cargo.toml")
    if python is None:
        errors.append("no [package] version found in infino-python/Cargo.toml")
    else:
        if crate is not None and release_line(python) != release_line(crate):
            errors.append(
                f"infino-python/Cargo.toml is on the {release_line(python)} "
                f"line ({python}) but the crate is on {release_line(crate)} "
                f"({crate}); major.minor is locked across the three "
                f"packages (docs/versioning.md)"
            )
        python_locked = locked_version(
            root / "infino-python" / "Cargo.lock", "infino-python"
        )
        if python_locked != python:
            errors.append(
                f"infino-python/Cargo.lock records infino-python "
                f"{python_locked} but infino-python/Cargo.toml says {python}; "
                f"regenerate the lockfile (releases publish with --locked)"
            )

    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-version",
        help="a version about to be published; must sit on the crate's "
             "major.minor release line",
    )
    args = parser.parse_args()
    errors = check(ROOT, release_version=args.release_version)
    if errors:
        for error in errors:
            print(f"version drift: {error}", file=sys.stderr)
        return 1
    crate = manifest_version(ROOT / "Cargo.toml")
    print(f"version sync OK: all packages on the {release_line(crate)} line")
    return 0


if __name__ == "__main__":
    sys.exit(main())
