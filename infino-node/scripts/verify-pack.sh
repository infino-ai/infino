#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Infino Authors
#
# Verify the *published package shape* locally, end to end.
#
# `npm test` loads the native addon as a sibling file, so it never exercises
# how an installed package finds its binary. This script does: it builds the
# release addon, stages it into its per-platform package, packs exactly what
# npm would ship, installs those tarballs into a throwaway project, and runs a
# real query. The installed main package contains no binary, so a successful
# roundtrip proves the loader resolved the addon from its optional
# per-platform dependency — the one thing that can break on a real install.
#
# Only the host platform is checked (no cross-compilation). Usage:
#
#     ./scripts/verify-pack.sh
set -euo pipefail

# Package root (this script lives in <root>/scripts).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Read names from package.json so this works for any package name
# (e.g. infino, infino-node): PKG_NAME is the npm package, BIN_NAME the
# napi binary prefix.
PKG_NAME="$(node -p "require('./package.json').name")"
BIN_NAME="$(node -p "require('./package.json').napi.name")"
# Per-platform packages are named after napi.package.name (e.g.
# infx-linux-x64-gnu), NOT the npm package name — that's what the
# generated loader requires, so it's what the reuse guard must grep for.
PLAT_PKG="$(node -p "const p=require('./package.json'); (p.napi.package && p.napi.package.name) || p.name")"

# Host platform in napi's node-platform naming.
PLATFORM="$(node -p 'process.platform')"
ARCH="$(node -p 'process.arch')"
case "$PLATFORM" in
  darwin) TRIPLE="darwin-${ARCH}" ;;
  linux)
    # glibc vs musl (Alpine) — napi names the binary/package per libc. Detect it
    # the same way napi's loader does: glibcVersionRuntime is absent under musl.
    # exit 0 = glibc, 1 = musl, 2 = couldn't tell (don't guess — that would
    # stage the wrong per-platform binary and fail confusingly).
    if node -e 'try{process.exit(process.report.getReport().header.glibcVersionRuntime?0:1)}catch{process.exit(2)}'; then
      TRIPLE="linux-${ARCH}-gnu"
    else
      status=$?
      if [ "$status" = "1" ]; then
        TRIPLE="linux-${ARCH}-musl"
      else
        # process.report unavailable — fall back to an authoritative libc probe.
        if ! command -v ldd >/dev/null 2>&1; then
          echo "verify-pack: could not determine libc for linux-${ARCH}: ldd is unavailable" >&2
          exit 1
        fi
        # Capture ldd's output first: musl's `ldd --version` exits non-zero, which
        # `set -o pipefail` would otherwise treat as "no match".
        ldd_out="$(ldd --version 2>&1 || true)"
        if printf '%s' "$ldd_out" | grep -qi musl; then
          TRIPLE="linux-${ARCH}-musl"
        elif printf '%s' "$ldd_out" | grep -qi 'glibc\|gnu libc'; then
          TRIPLE="linux-${ARCH}-gnu"
        else
          echo "verify-pack: could not determine libc (glibc vs musl) for linux-${ARCH}" >&2
          exit 1
        fi
      fi
    fi ;;
  *) echo "verify-pack: unsupported host platform '$PLATFORM'" >&2; exit 1 ;;
esac
PKG_DIR="npm/${TRIPLE}"
NODE_FILE="${BIN_NAME}.${TRIPLE}.node"

# Clean up the staged binary and the throwaway project on any exit.
WORK=""
cleanup() {
  rm -f "$PKG_DIR/$NODE_FILE"
  [ -n "$WORK" ] && rm -rf "$WORK"
}
trap cleanup EXIT

# Reuse an existing build by default — this script verifies packaging, not
# the binary's contents, so a debug build is fine and a rebuild is slow.
# Force a fresh release build with REBUILD=1.
NEED_BUILD=0
[ "${REBUILD:-0}" = "1" ] && NEED_BUILD=1
for f in "infino/${NODE_FILE}" infino/index.js infino/index.d.ts infino/native.js infino/native.d.ts; do
  [ -f "$f" ] || NEED_BUILD=1
done
# A reused loader from a previous package name (after a napi.name change)
# requires a different platform package and would fail at load — rebuild if
# native.js doesn't reference the current platform package name.
if [ "$NEED_BUILD" = "0" ] && ! grep -q "${PLAT_PKG}-${TRIPLE}" infino/native.js 2>/dev/null; then
  NEED_BUILD=1
fi
if [ "$NEED_BUILD" = "1" ]; then
  echo "==> [1/5] building addon + wrapper"
  npm install
  npm run build
else
  echo "==> [1/5] reusing existing infino/${NODE_FILE} (set REBUILD=1 to rebuild)"
fi

# npm/ is generated (gitignored) — regenerate the per-platform package dirs
# from package.json, the same way the publish workflow does.
npx napi create-npm-dir -t . >/dev/null
if [ ! -d "$PKG_DIR" ]; then
  echo "verify-pack: host triple $TRIPLE not in napi.triples" >&2
  exit 1
fi

echo "==> [2/5] staging $NODE_FILE into $PKG_DIR"
cp "infino/${NODE_FILE}" "$PKG_DIR/"

echo "==> [3/5] dry-run publish (validates manifests, no upload)"
# --tag latest: the name's history holds a higher version, so npm won't move
# the latest tag onto this one implicitly (see the publish workflow).
npm publish --dry-run --tag latest
( cd "$PKG_DIR" && npm publish --dry-run )

echo "==> [4/5] packing the tarballs npm would ship"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/infino-verify-pack.XXXXXX")"
MAIN_TGZ="$(npm pack "$ROOT" --pack-destination "$WORK" 2>/dev/null | tail -1)"
PLAT_TGZ="$(npm pack "$ROOT/$PKG_DIR" --pack-destination "$WORK" 2>/dev/null | tail -1)"

echo "==> [5/5] installing into a throwaway project + running a roundtrip"
cd "$WORK"
npm init -y >/dev/null
npm install "$WORK/$MAIN_TGZ" "$WORK/$PLAT_TGZ" 'apache-arrow@^17' >/dev/null
INFINO_PKG="$PKG_NAME" node --input-type=module <<'JS'
const { connect, IndexSpec } = await import(process.env.INFINO_PKG);
const { Schema, Field, LargeUtf8 } = await import("apache-arrow");

const db = connect("memory://");
const docs = db.createTable(
  "docs",
  new Schema([new Field("title", new LargeUtf8(), false)]),
  new IndexSpec().fts("title"),
);
docs.append([{ title: "the quick brown fox" }, { title: "a lazy dog" }]);
const hits = docs.bm25Search("title", "fox", 10);
if (hits.length !== 1 || typeof hits[0]._id !== "bigint") {
  throw new Error("roundtrip failed: unexpected hits");
}
console.log("    roundtrip ok — _id:", hits[0]._id.toString(), "score:", hits[0].score);
JS

echo ""
echo "✓ verify-pack passed: the published package shape resolves the native"
echo "  binary from its optional per-platform dependency and works end to end."
