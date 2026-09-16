#!/usr/bin/env bash
# Rebuild the corpus tables the shape tests read.
#
# Each generator pins one published engine release, and the release decides
# the superfile shape its builder writes. The tables are a few megabytes of
# fixture bytes, so they are generated rather than committed; the generators
# and this script are what is version-controlled.
#
# Usage: tests/corpus/generate.sh [shape ...]   (default: all)
set -euo pipefail

cd "$(dirname "$0")"
tables="$PWD/tables"

# shape : generator directory : expected FTS blob version
shapes=(
  "v1_positionless:v0_1_5:1"
  "v2_positions_region:v0_5_4:2"
  "v4_bitset_blocks:v0_5_12:4"
  "v5_positionless:v0_8_0:5"
  "v5_positional:v0_8_2:5"
)

wanted=("$@")
for entry in "${shapes[@]}"; do
  IFS=: read -r shape gen version <<<"$entry"
  if [ ${#wanted[@]} -gt 0 ] && [[ ! " ${wanted[*]} " =~ " ${shape} " ]]; then
    continue
  fi

  echo "==> $shape (engine ${gen#v}, expecting blob version $version)"
  bin="generators/$gen/target/release/corpus-gen-$(echo "${gen#v}" | tr '_' '-')"
  ( cd "generators/$gen" && cargo build --release --quiet )

  rm -rf "${tables:?}/$shape"
  mkdir -p "$tables/$shape"
  "$bin" "$tables/$shape" corpus >/dev/null

  # The shape is content-dependent, not just a property of the writer: a
  # corpus too sparse to produce a dense block, or too small for a
  # multi-entry coarse table, makes an older builder stamp a lower version.
  # Assert here so a silently weaker corpus fails at generation rather than
  # passing a test that then proves less than it claims.
  python3 - "$tables/$shape" "$version" <<'PY'
import glob, struct, sys

root, expected = sys.argv[1], int(sys.argv[2])
files = sorted(glob.glob(f"{root}/**/data/*.sf.parquet", recursive=True))
if not files:
    sys.exit(f"no superfiles written under {root}")
for path in files:
    blob = open(path, "rb").read()
    at = blob.find(b"INFFTS01")
    if at < 0:
        sys.exit(f"{path}: no FTS blob")
    version, _, n_docs, _ = struct.unpack_from("<IIII", blob, at + 8)
    if version != expected:
        sys.exit(f"{path}: blob version {version}, expected {expected}")
print(f"    {len(files)} superfile(s), blob version {expected}")
PY
done
