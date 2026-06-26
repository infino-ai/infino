#!/usr/bin/env bash
set -euo pipefail

# Resolve requested cpuset, or auto-pick stable default:
# - <=2 CPUs: use all
# - >2 CPUs: reserve CPU0 for OS noise, use 1..N-1
requested="${1:-}"
if [ -n "$requested" ]; then
  cpuset="$requested"
else
  ncpu="$(nproc)"
  if [ "${ncpu}" -le 2 ]; then
    cpuset="0-$((ncpu - 1))"
  else
    cpuset="1-$((ncpu - 1))"
  fi
fi

if ! command -v taskset >/dev/null 2>&1; then
  echo "error: taskset unavailable (resolved cpuset '$cpuset')" >&2
  exit 1
fi
if ! taskset -c "$cpuset" true >/dev/null 2>&1; then
  echo "error: invalid cpuset '$cpuset' for this VM" >&2
  exit 1
fi

printf '%s\n' "$cpuset"
