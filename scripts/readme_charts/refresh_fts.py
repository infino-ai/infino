#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Infino Authors

"""Refresh SBG_ROWS in bench_data.py from the published benchmark run.

Fetches the latest Search Benchmark, the Game results and recomputes the
full-text comparison chart's rows: the geometric mean, across all queries,
of each engine's best-of-N latency relative to Lucene's. The engines are
the newest published infino crate, the Lucene baseline, and the newest
Tantivy — the same roster the chart has always shown. Run generate.py
afterwards to redraw the SVG.
"""

from __future__ import annotations

import json
import math
import re
import sys
import urllib.request
from pathlib import Path

RESULTS_URL = "https://infino-ai.github.io/search-benchmark-game/results.json"
# Chart columns: "search (top-k)" and "count".
COMMANDS = ("TOP_10", "COUNT")
BENCH_DATA = Path(__file__).resolve().parent / "bench_data.py"


def version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def newest(engines: list[str], prefix: str) -> str:
    """The highest plain `<prefix>-X.Y[.Z]` engine; variants like `-bp` or
    `-main` are excluded so every row is a released build."""
    pattern = re.compile(rf"^{prefix}-(\d+(?:\.\d+)*)$")
    versions = [m.group(1) for name in engines if (m := pattern.match(name))]
    if not versions:
        raise SystemExit(f"no released {prefix} engine in the results")
    return f"{prefix}-{max(versions, key=version_key)}"


def geomean_ratio(results: dict, command: str, engine: str, baseline: str) -> float:
    base = {q["query"]: min(q["duration"]) for q in results[command][baseline]}
    ratios = [
        min(q["duration"]) / base[q["query"]]
        for q in results[command][engine]
        if base.get(q["query"], 0) > 0 and min(q["duration"]) > 0
    ]
    if len(ratios) < len(base) * 0.9:
        raise SystemExit(f"{engine} covers too few of {baseline}'s queries in {command}")
    return math.exp(sum(math.log(r) for r in ratios) / len(ratios))


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else RESULTS_URL
    with urllib.request.urlopen(url) as response:
        results = json.load(response)["results"]

    engines = sorted(results[COMMANDS[0]].keys())
    baseline = newest(engines, "lucene")
    roster = [
        ("Infino", newest(engines, "infino"), True),
        ("Lucene", baseline, False),
        ("Tantivy", newest(engines, "tantivy"), False),
    ]

    rows = []
    for name, engine, is_infino in roster:
        search, count = (
            geomean_ratio(results, command, engine, baseline) for command in COMMANDS
        )
        version = engine.split("-", 1)[1]
        rows.append(f'    ("{name}", "{version}", {search:.2f}, {count:.2f}, {is_infino}),')

    source = BENCH_DATA.read_text()
    block = "SBG_ROWS: list[tuple[str, str, float, float, bool]] = [\n" + "\n".join(rows) + "\n]"
    updated, substitutions = re.subn(
        r"SBG_ROWS: list\[tuple\[str, str, float, float, bool\]\] = \[\n(?:.*?\n)*?\]",
        block,
        source,
        count=1,
    )
    if substitutions != 1:
        raise SystemExit("SBG_ROWS block not found in bench_data.py")
    BENCH_DATA.write_text(updated)
    print(block)


if __name__ == "__main__":
    main()
