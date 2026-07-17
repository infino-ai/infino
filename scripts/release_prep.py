#!/usr/bin/env python3
"""Stamp every version file for a release, per docs/versioning.md.

One engine, three published packages, four kinds of release:

  release_prep.py --package crate  --version 0.1.5   # engine patch
  release_prep.py --package node   --version 0.1.5   # Node-only patch
  release_prep.py --package python --version 0.1.6   # Python-only patch
  release_prep.py --package each                     # every package, own next patch
  release_prep.py --package all    --version 0.2.0   # coordinated minor/major

A single-package bump must stay on the crate's `major.minor` release line
and move that package strictly forward (registries are immutable — every
publish needs a fresh version). `each` bumps every package to its own next
patch — patch counters are independent, so no version argument applies. A
coordinated release must have patch 0 (the binding publish workflows key on
the `.0` tag suffix) and moves all three packages onto the new line together.

Each scope stamps every file that carries that package's version — manifest,
lockfile, and for Node the `infx-*` platform pins — then re-runs the
version-sync guard as a self-check, and prints the follow-up step (which tag
to push or which workflow to dispatch).

Plain `X.Y.Z` versions only; pre-releases are rare enough to stamp by hand.
Typical use, from the repo root:

  make release-prep PACKAGE=node VERSION=0.1.5
"""

import argparse
import json
import re
import sys
from pathlib import Path

from check_version_sync import check, manifest_version, release_line

ROOT = Path(__file__).resolve().parent.parent

_STRICT_VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

FOLLOW_UP = {
    "crate": "merge the release PR — the 'Tag release' workflow pushes v{v}"
             " and kicks off the crate publish (bindings skip a patch)",
    "node": "merge the release PR, then dispatch the 'Node publish' workflow"
            " (uncheck dry_run to publish)",
    "python": "merge the release PR, then dispatch the 'Publish Python"
              " package' workflow (it reads the version from the tree)",
    "all": "merge the release PR — the 'Tag release' workflow pushes v{v}"
           " and kicks off all three publish workflows",
    "each": "merge the release PR — 'Tag release' tags and publishes the"
            " crate; then dispatch 'Node publish' (uncheck dry_run) and"
            " 'Publish Python package' for the bindings",
}


class ReleasePrepError(Exception):
    """A requested release violates the versioning contract."""


def parse_version(version):
    m = _STRICT_VERSION.match(version)
    if not m:
        raise ReleasePrepError(
            f"version must be plain X.Y.Z (got {version!r}); stamp"
            f" pre-releases by hand"
        )
    return tuple(int(part) for part in m.groups())


def stamp_manifest(path, version):
    """Rewrite the `[package]` version of a Cargo.toml."""
    text, count = re.subn(
        r'(?m)^version = "[^"]*"', f'version = "{version}"',
        path.read_text(), count=1,
    )
    if count != 1:
        raise ReleasePrepError(f"no [package] version found in {path}")
    path.write_text(text)


def stamp_lock(path, name, version):
    """Rewrite the version Cargo.lock records for package `name`."""
    text, count = re.subn(
        rf'(?ms)(\[\[package\]\]\nname = "{name}"\nversion = ")[^"]*(")',
        rf"\g<1>{version}\g<2>", path.read_text(), count=1,
    )
    if count != 1:
        raise ReleasePrepError(f"no {name} entry found in {path}")
    path.write_text(text)


def stamp_node_package(path, version):
    """Rewrite package.json's version and its infx-* platform pins."""
    manifest = json.loads(path.read_text())
    manifest["version"] = version
    for pin in manifest.get("optionalDependencies", {}):
        if pin.startswith("infx-"):
            manifest["optionalDependencies"][pin] = version
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def current_versions(root):
    return {
        "crate": manifest_version(root / "Cargo.toml"),
        "node": json.loads(
            (root / "infino-node" / "package.json").read_text()
        )["version"],
        "python": manifest_version(root / "infino-python" / "Cargo.toml"),
    }


def next_patch(version):
    major, minor, patch = parse_version(version)
    return f"{major}.{minor}.{patch + 1}"


def validate(root, package, version):
    """Check the bump obeys the versioning rules; return {package: version}
    targets to stamp. Raises ReleasePrepError on any violation."""
    current = current_versions(root)
    crate = current["crate"]

    if package == "each":
        if version is not None:
            raise ReleasePrepError(
                "a version does not apply to 'each' — every package moves"
                " to its own next patch (their counters are independent)"
            )
        return {name: next_patch(cur) for name, cur in current.items()}

    if version is None:
        raise ReleasePrepError(f"a version is required for '{package}'")
    requested = parse_version(version)

    if package == "all":
        if requested[2] != 0:
            raise ReleasePrepError(
                f"a coordinated release needs patch 0 (got {version}); the"
                f" binding publish workflows fire only on a vX.Y.0 tag"
            )
        for name, cur in current.items():
            if requested <= parse_version(cur):
                raise ReleasePrepError(
                    f"coordinated version {version} does not move {name}"
                    f" forward (currently {cur})"
                )
        return {name: version for name in current}

    if release_line(version) != release_line(crate):
        raise ReleasePrepError(
            f"a {package} patch must stay on the crate's release line"
            f" {release_line(crate)} (got {version}); changing the line"
            f" is a coordinated release (--package all)"
        )
    cur = current[package]
    if requested <= parse_version(cur):
        raise ReleasePrepError(
            f"{package} is already at {cur}; the new version must move"
            f" it strictly forward (got {version})"
        )
    return {package: version}


def prepare(root, package, version=None):
    """Validate, then stamp every file carrying each target's version.

    Returns the repo-relative paths that were rewritten. Nothing is written
    if validation fails.
    """
    root = Path(root)
    targets = validate(root, package, version)

    changed = []
    if "crate" in targets:
        stamp_manifest(root / "Cargo.toml", targets["crate"])
        stamp_lock(root / "Cargo.lock", "infino", targets["crate"])
        changed += ["Cargo.toml", "Cargo.lock"]
    if "node" in targets:
        stamp_node_package(root / "infino-node" / "package.json",
                           targets["node"])
        stamp_manifest(root / "infino-node" / "Cargo.toml", targets["node"])
        stamp_lock(root / "infino-node" / "Cargo.lock", "infino-node",
                   targets["node"])
        changed += ["infino-node/package.json", "infino-node/Cargo.toml",
                    "infino-node/Cargo.lock"]
    if "python" in targets:
        stamp_manifest(root / "infino-python" / "Cargo.toml",
                       targets["python"])
        stamp_lock(root / "infino-python" / "Cargo.lock", "infino-python",
                   targets["python"])
        changed += ["infino-python/Cargo.toml", "infino-python/Cargo.lock"]

    drift = check(root)
    if drift:
        raise ReleasePrepError(
            "stamping left the tree inconsistent (bug in this script):\n  "
            + "\n  ".join(drift)
        )
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True,
                        choices=["crate", "node", "python", "each", "all"])
    parser.add_argument("--version",
                        help="new version, plain X.Y.Z (omit for 'each')")
    args = parser.parse_args()

    try:
        changed = prepare(ROOT, args.package, args.version)
    except ReleasePrepError as error:
        print(f"release-prep: {error}", file=sys.stderr)
        return 1

    stamped = current_versions(ROOT)
    markers = {"crate": "Cargo.toml", "node": "infino-node/package.json",
               "python": "infino-python/Cargo.toml"}
    print("stamped:")
    for name, marker in markers.items():
        if marker in changed:
            print(f"  {name} -> {stamped[name]}")
    for path in changed:
        print(f"  {path}")

    slug = args.version if args.version else stamped["crate"]
    print("\nnext steps:")
    print(f"  1. commit on a branch and open the release PR, e.g.:")
    print(f"       git checkout -b release-{args.package}-{slug}")
    print(f"       git commit -am 'release: {args.package} {slug}'")
    print(f"  2. {FOLLOW_UP[args.package].format(v=slug)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
