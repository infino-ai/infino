#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Infino Authors

"""Render the README performance charts from bench_data.py.

Latency spans three or more orders of magnitude between a warm hit and a cold
first query, so the charts use a base-10 log axis with a labelled gridline on
every decade. A linear axis collapses the warm bar to a stub and hides the
number the chart exists to show.
"""

from __future__ import annotations

import math
from pathlib import Path

from bench_data import (
    CLICKBENCH_SEARCH_META,
    CLICKBENCH_SEARCH_ROWS,
    EMBED_META,
    EMBED_ROWS,
    SBG_META,
    SBG_ROWS,
    SQL_EXT_META,
    SQL_EXT_ROWS,
    VDB_META,
    VDB_ROWS,
    CROSSOVER,
    FTS,
    INGEST,
    MODES_LATENCY,
    MODES_MEMORY,
    SQL,
    SQL_PUSHDOWN,
    VECTOR,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "readme"

WARM = "#D6400F"
COLD = "#B7B4AB"
BG = "#FBF9F3"
INK = "#17130F"
MUTED = "#5E5749"
GRID = "#DCD6C7"
FONT = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

WIDTH = 880
PAD = 26
LABEL_X = 170
TRACK_X = 184
TRACK_W = 600
VALUE_X = WIDTH - PAD

PLOT_TOP = 56
BAR_H = 9
ROW_PITCH = 22
GROUP_GAP = 12
# A group's header line: its own row, so a long group name (an SQL shape)
# can never collide with the first member's label.
GROUP_HEADER_H = 20

# A bar whose value sits on the axis minimum would render as zero width.
MIN_BAR_W = 3
# Headroom above the largest value so its bar stops short of the frame.
AXIS_HEADROOM = 1.15
# Linear axes need more: the max bar is proportionally longer there.
LINEAR_HEADROOM = 1.18
# Start the axis a decade lower when the smallest value sits near a decade
# boundary, otherwise its bar is a stub against the axis origin.
LOW_DECADE_MARGIN = 2


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt_ms(ms: float) -> str:
    if ms < 1:
        return f"{ms * 1000:.0f} µs"
    if ms < 10:
        return f"{ms:g} ms"
    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms / 1000:g} s"


def fmt_mib(mib: float) -> str:
    if mib < 1024:
        return f"{mib:.0f} MiB"
    return f"{mib / 1024:.2f} GiB"


def fmt_kps(kps: float) -> str:
    return f"{kps:g} K docs/s"


FORMATTERS = {"ms": fmt_ms, "mib": fmt_mib, "kps": fmt_kps}


def log_axis(values: list[float]) -> tuple[float, float, list[float]]:
    """Return (axis min, axis max, decade tick values) for a log scale."""
    lo_v, hi_v = min(values), max(values)
    lo_dec = math.floor(math.log10(lo_v))
    if lo_v / 10.0**lo_dec < LOW_DECADE_MARGIN:
        lo_dec -= 1
    lo = 10.0**lo_dec
    hi = hi_v * AXIS_HEADROOM
    ticks = []
    dec = lo_dec
    while 10.0**dec <= hi:
        ticks.append(10.0**dec)
        dec += 1
    return lo, hi, ticks


def text(x, y, s, *, size=12, fill=MUTED, anchor="start", weight=None) -> str:
    w = f' font-weight="{weight}"' if weight else ""
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
        f'fill="{fill}"{w}{a}>{esc(s)}</text>'
    )


def frame(height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img">',
        f'<rect width="{WIDTH}" height="{height}" rx="10" fill="{BG}" stroke="{GRID}"/>',
        text(PAD, 32, title, size=13, fill=INK, weight=600),
        text(VALUE_X, 32, subtitle, size=11, anchor="end"),
    ]


def latency_chart(spec: dict, filename: str) -> None:
    bars = spec["bars"]
    groups: list[tuple[str, list]] = []
    for bar in bars:
        if not groups or groups[-1][0] != bar.group:
            groups.append((bar.group, []))
        groups[-1][1].append(bar)

    fmt_value = FORMATTERS[spec.get("unit", "ms")]
    linear = spec.get("scale") != "log"
    if linear:
        # Memory (and anything else spanning ~one order of magnitude)
        # reads on a linear axis: log is a rescue for latency's decades,
        # and here it would visually flatten exactly the gap the chart
        # exists to show.
        # Wider headroom than the log axis: a linear max-value bar ends
        # flush with the track, and the value label starts ~70px later —
        # 18% keeps the longest bar clear of its own number.
        hi = max(b.ms for b in bars) * LINEAR_HEADROOM
        lo = 0.0
        # Memory ticks step in powers of two so gridlines land on round
        # GiB values instead of 1.95/3.91; other units keep decimal steps.
        if spec.get("unit") == "mib":
            step = next(2.0**e for e in range(4, 20) if hi / 2.0**e <= 4)
        else:
            step_base = 10.0 ** math.floor(math.log10(hi / 2))
            step = next(m * step_base for m in (1, 2, 5, 10) if hi / (m * step_base) <= 4)
        ticks = [step * i for i in range(1, int(hi / step) + 1)]

        def bar_w(v: float) -> float:
            return max((v / hi) * TRACK_W, MIN_BAR_W)

    else:
        lo, hi, ticks = log_axis([b.ms for b in bars])
        span = math.log10(hi / lo)

        def bar_w(v: float) -> float:
            return max((math.log10(v / lo) / span) * TRACK_W, MIN_BAR_W)

    n_headers = 0 if len(groups) == 1 else len(groups)
    plot_h = len(bars) * ROW_PITCH + n_headers * GROUP_HEADER_H + (len(groups) - 1) * GROUP_GAP
    axis_y = PLOT_TOP + plot_h + 4
    has_cold = any(b.cold for b in bars)
    height = int(axis_y + 24 + (30 if has_cold else 26))

    lines = frame(height, spec["title"], spec["subtitle"])

    for tick in ticks:
        gx = TRACK_X + (
            (tick / hi) * TRACK_W if linear else (math.log10(tick / lo) / span) * TRACK_W
        )
        lines.append(
            f'<line x1="{gx:.1f}" y1="{PLOT_TOP - 6}" x2="{gx:.1f}" y2="{axis_y}" '
            f'stroke="{GRID}"/>'
        )
        lines.append(text(gx, axis_y + 16, fmt_value(tick), size=10, anchor="middle"))
    if not linear:
        lines.append(text(PAD, axis_y + 16, "log scale", size=10))

    y = PLOT_TOP
    lone_group = len(groups) == 1
    for name, members in groups:
        # The header owns its line: drawn inline with the first bar it
        # collides with any member label once the group name passes the
        # label gutter (every SQL shape does). A lone group's name is
        # already the subtitle's job, so it draws no header at all.
        if not lone_group:
            lines.append(text(PAD, y + 10, name, size=12, fill=INK, weight=600))
            y += GROUP_HEADER_H
        baseline = next((b.ms for b in members if b.cold), None)
        for bar in members:
            color = COLD if bar.cold else WARM
            lines.append(text(LABEL_X, y + BAR_H, bar.label, size=11, anchor="end"))
            lines.append(
                f'<rect x="{TRACK_X}" y="{y}" width="{bar_w(bar.ms):.1f}" '
                f'height="{BAR_H}" rx="1" fill="{color}"/>'
            )
            value = fmt_value(bar.ms)
            # State the saving, don't make the reader eyeball it: measured
            # rows carry their factor against the group's baseline bar.
            if spec.get("ratio_vs_cold") and not bar.cold and baseline:
                value = f"{value} · {baseline / bar.ms:.1f}x less"
            lines.append(
                text(
                    VALUE_X,
                    y + BAR_H,
                    value,
                    size=11,
                    fill=MUTED if bar.cold else INK,
                    anchor="end",
                    weight=600,
                )
            )
            y += ROW_PITCH
        y += GROUP_GAP

    foot_y = height - 12
    if has_cold:
        legend_warm = spec.get("legend_warm") or "warm cache"
        legend_cold = spec.get("legend_cold") or "cold · first query on an idle table"
        lines.append(
            f'<rect x="{PAD}" y="{foot_y - 21}" width="9" height="9" rx="1" fill="{WARM}"/>'
        )
        lines.append(text(PAD + 16, foot_y - 13, legend_warm, size=11))
        cold_x = PAD + 16 + 8 * len(legend_warm) + 24
        lines.append(
            f'<rect x="{cold_x}" y="{foot_y - 21}" width="9" height="9" rx="1" fill="{COLD}"/>'
        )
        lines.append(text(cold_x + 16, foot_y - 13, legend_cold, size=11))
    lines.append(text(VALUE_X, foot_y, spec["footnote"], size=10, anchor="end"))
    lines.append("</svg>")
    (OUT / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_chart(meta: dict, rows: list, filename: str, *, log: bool = False) -> None:
    """One row per engine: name, latency bar, then the data columns.

    Everything that decides a choice — the measured value and each extra
    column (recall, resident memory, deployment) — renders at full size in
    aligned columns with headers. Sublabel-sized annotations hide exactly
    the numbers a reader needs to weigh the bar, so there are none.
    """
    columns: tuple[str, ...] = meta.get("columns", ())
    # Lane widths follow their content (monospace: ~6.7 px/char at 11px,
    # plus a gutter) so a long deployment string can never crowd its
    # neighbor; the bar track absorbs whatever width the lanes release.
    def lane_w(i: int) -> float:
        longest = max([len(columns[i])] + [len(r.cols[i]) for r in rows])
        return longest * 6.7 + 22
    widths = [lane_w(i) for i in range(len(columns))]
    label_x = 218
    track_x = 232
    value_x = WIDTH - PAD - sum(widths)
    track_w = value_x - track_x - 78
    col_right = []
    edge = float(VALUE_X)
    for w in reversed(widths):
        col_right.insert(0, edge)
        edge -= w

    if log:
        lo, hi, ticks = log_axis([r.value for r in rows])
        span = math.log10(hi / lo)

        def bar_w(v: float) -> float:
            return max((math.log10(v / lo) / span) * track_w, MIN_BAR_W)

    else:
        hi = max(r.value for r in rows)
        ticks = []

        def bar_w(v: float) -> float:
            return max(track_w * (v / hi), MIN_BAR_W)

    header_y = PLOT_TOP - 4
    row_top = PLOT_TOP + 12
    plot_h = len(rows) * 28
    axis_y = row_top + plot_h - 6
    height = int(axis_y + (26 if log else 10) + 16)
    lines = frame(height, meta["title"], meta["subtitle"])

    for tick in ticks:
        gx = track_x + (math.log10(tick / lo) / span) * track_w
        lines.append(
            f'<line x1="{gx:.1f}" y1="{row_top - 6}" x2="{gx:.1f}" y2="{axis_y}" stroke="{GRID}"/>'
        )
        lines.append(text(gx, axis_y + 16, f"{tick:g}", size=10, anchor="middle"))
    if log:
        lines.append(text(PAD, axis_y + 16, "log scale", size=10))

    # Column headers, right-aligned over their lanes.
    value_header = meta.get("value_header", "")
    if value_header:
        lines.append(text(value_x, header_y, value_header, size=10, anchor="end"))
    for i, name in enumerate(columns):
        lines.append(text(col_right[i], header_y, name, size=10, anchor="end"))

    y = row_top
    for row in rows:
        color = WARM if row.self_row else COLD
        ink = INK if row.self_row else MUTED
        lines.append(
            text(
                label_x,
                y + 10,
                row.name,
                size=12,
                fill=ink,
                weight=600 if row.self_row else None,
                anchor="end",
            )
        )
        lines.append(
            f'<rect x="{track_x}" y="{y + 2}" width="{bar_w(row.value):.1f}" '
            f'height="{BAR_H}" rx="1" fill="{color}"/>'
        )
        lines.append(text(value_x, y + 10, row.label, size=11, fill=ink, anchor="end", weight=600))
        for i, cell in enumerate(row.cols):
            lines.append(
                text(
                    col_right[i],
                    y + 10,
                    cell,
                    size=11,
                    fill=ink,
                    anchor="end",
                    weight=600 if row.self_row else None,
                )
            )
        y += 28
    lines.append("</svg>")
    (OUT / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def ratio_chart(meta: dict, rows: list, filename: str) -> None:
    """Two bars per engine (search, count), as a ratio against a 1.00 baseline."""
    label_x, track_x, track_w = 250, 264, 420
    hi = max(max(search, count) for _, _, search, count, _ in rows)
    height = PLOT_TOP + len(rows) * 44 + 40
    lines = frame(height, meta["title"], meta["subtitle"])

    base_x = track_x + (1.0 / hi) * track_w
    lines.append(
        f'<line x1="{base_x:.1f}" y1="{PLOT_TOP - 6}" x2="{base_x:.1f}" '
        f'y2="{PLOT_TOP + len(rows) * 44 - 8}" stroke="{GRID}"/>'
    )

    y = PLOT_TOP
    for name, version, search, count, is_self in rows:
        lines.append(
            text(
                label_x,
                y + 10,
                name,
                size=12,
                fill=INK if is_self else MUTED,
                weight=600 if is_self else None,
                anchor="end",
            )
        )
        lines.append(text(label_x, y + 24, version, size=10, anchor="end"))
        for idx, val in enumerate((search, count)):
            if is_self:
                fill = WARM if idx == 0 else "#E8A090"
            else:
                fill = COLD if idx == 0 else "#CFCBC2"
            yy = y + idx * 14
            lines.append(
                f'<rect x="{track_x}" y="{yy}" width="{max(track_w * (val / hi), MIN_BAR_W):.1f}" '
                f'height="8" rx="1" fill="{fill}"/>'
            )
            lines.append(
                text(
                    VALUE_X,
                    yy + 7,
                    f"{val:.2f}×",
                    size=10,
                    fill=INK if is_self else MUTED,
                    anchor="end",
                    weight=600,
                )
            )
        y += 44

    foot_y = height - 14
    lines.append(f'<rect x="{PAD}" y="{foot_y - 7}" width="9" height="9" rx="1" fill="{COLD}"/>')
    lines.append(text(PAD + 16, foot_y, "search (top-k)", size=11))
    lines.append(f'<rect x="150" y="{foot_y - 7}" width="9" height="9" rx="1" fill="#CFCBC2"/>')
    lines.append(text(166, foot_y, "count", size=11))
    lines.append("</svg>")
    (OUT / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def architecture(filename: str) -> None:
    """The how-it-works diagram, in the charts' own visual language.

    One left-to-right read: your app calls the engine in-process, the
    engine answers from its cache, misses become byte-range reads of
    superfiles on object storage — and a superfile is one valid Parquet
    file other readers open as a plain table. The previous version drew
    query kinds pointing INTO the file with no cache layer, which read
    as a write pipeline and hid the thing that explains warm latency.
    """
    H = 456
    lines = frame(H, "One Parquet copy, every retrieval path",
                  "columns and search indexes travel in one file, write to query")

    def box(x, y, w, h, fill=BG, stroke=GRID, rx=8, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}"{d}/>'
        )

    def arrow(x1, y1, x2, y2):
        lines.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{MUTED}" '
            f'stroke-width="1.4" marker-end="url(#arr)"/>'
        )

    lines.insert(2,
        f'<defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" '
        f'markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{MUTED}"/>'
        f"</marker></defs>")

    # ── your app ─────────────────────────────────────────────────────
    box(PAD, 96, 178, 168, fill="#FFFFFF")
    lines.append(text(PAD + 14, 122, "YOUR APP", size=10, fill=MUTED))
    for i, call in enumerate(
        ["bm25_search()", "vector_search()", "hybrid_search()", "query_sql()"]
    ):
        lines.append(text(PAD + 14, 148 + i * 22, call, size=12, fill=INK))
    lines.append(text(PAD + 14, 248, "one in-process library", size=10))

    arrow(PAD + 178, 180, 262, 180)
    lines.append(text((PAD + 178 + 262) / 2, 168, "calls", size=10, anchor="middle"))

    # ── engine ───────────────────────────────────────────────────────
    box(262, 96, 266, 250, fill="#FFFFFF")
    lines.append(text(276, 122, "INFINO ENGINE", size=10, fill=WARM, weight=600))
    for i, s_ in enumerate([
        "every result is a relation:",
        "filter, join, aggregate over",
        "ranked hits in one statement",
    ]):
        lines.append(text(276, 148 + i * 18, s_, size=11, fill=INK))
    for i, s_ in enumerate([
        "reads pin a snapshot;",
        "writers never block them",
    ]):
        lines.append(text(276, 216 + i * 18, s_, size=11, fill=MUTED))
    box(276, 262, 238, 66, fill=BG)
    lines.append(text(290, 284, "cache: RAM + local disk", size=11, fill=INK, weight=600))
    lines.append(text(290, 302, "filled by queries, shrinks", size=10))
    lines.append(text(290, 316, "when idle — warm reads stop here", size=10))

    lines.append(text(561, 146, "on a miss:", size=10, anchor="middle"))
    lines.append(text(561, 160, "byte ranges,", size=10, anchor="middle"))
    lines.append(text(561, 174, "not files", size=10, anchor="middle"))
    arrow(528, 188, 596, 188)

    # ── object storage with superfiles ──────────────────────────────
    box(596, 84, 258, 286, dash="4 3")
    lines.append(text(610, 106, "OBJECT STORAGE — S3 / Azure / GCS / disk", size=9, fill=MUTED))
    # ghost superfiles behind the front card
    box(632, 122, 206, 226, fill="#FFFFFF")
    box(622, 130, 206, 222, fill="#FFFFFF")
    box(612, 138, 206, 216, fill="#FFFFFF", stroke=MUTED)
    fx, fw = 612, 206
    lines.append(text(fx + 12, 158, "SUPERFILE", size=10, fill=WARM, weight=600))
    lines.append(text(fx + 12, 172, "one valid Parquet file", size=10, fill=MUTED))
    region = [
        ("Parquet columns", "your rows: text, numbers", "#FFFFFF"),
        ("BM25 index", "dictionary + postings", "#F7E8DC"),
        ("vector index", "clusters + codes + rerank", "#EDE9F3"),
    ]
    ry = 182
    for name, sub, fill in region:
        box(fx + 10, ry, fw - 20, 40, fill=fill, rx=5)
        lines.append(text(fx + 20, ry + 17, name, size=11, fill=INK, weight=600))
        lines.append(text(fx + 20, ry + 31, sub, size=9, fill=MUTED))
        ry += 46
    box(fx + 10, ry, fw - 20, 22, fill=BG, rx=5)
    lines.append(text(fx + 20, ry + 15, "standard Parquet footer", size=9, fill=MUTED))

    # ── ecosystem claim, drawn not implied ───────────────────────────
    arrow(fx + 100, 356, fx + 100, 396)
    lines.append(
        text(fx + 103, 416, "DuckDB, pyarrow, DataFusion", size=11, fill=INK, anchor="middle")
    )
    lines.append(
        text(fx + 103, 431, "open the same file as a plain table", size=10, anchor="middle")
    )
    lines.append(
        text(PAD, 416, "no daemon, no cluster: commit swaps a manifest naming", size=10)
    )
    lines.append(text(PAD, 431, "immutable superfiles — all of a batch appears, or none", size=10))
    lines.append("</svg>")
    (OUT / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    architecture("one-parquet-copy.svg")
    latency_chart(VECTOR, "vector.svg")
    latency_chart(FTS, "fts.svg")
    latency_chart(SQL, "sql.svg")
    latency_chart(MODES_MEMORY, "vector-modes-memory.svg")
    latency_chart(MODES_LATENCY, "vector-modes-latency.svg")
    latency_chart(SQL_PUSHDOWN, "sql-pushdown.svg")
    latency_chart(INGEST, "ingest.svg")
    latency_chart(CROSSOVER, "vector-crossover.svg")
    compare_chart(VDB_META, VDB_ROWS, "compare-vdb.svg")
    compare_chart(EMBED_META, EMBED_ROWS, "compare-embedded.svg")
    compare_chart(SQL_EXT_META, SQL_EXT_ROWS, "compare-sql.svg")
    compare_chart(
        CLICKBENCH_SEARCH_META, CLICKBENCH_SEARCH_ROWS, "compare-sql-search.svg", log=True
    )
    ratio_chart(SBG_META, SBG_ROWS, "compare-fts.svg")
    print(f"wrote charts to {OUT}")


if __name__ == "__main__":
    main()
