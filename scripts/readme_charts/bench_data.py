# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Infino Authors

"""Numbers and provenance for the README performance charts.

Two measurement windows feed these charts and they are *not* directly
comparable — different commits, different hardware:

  1M  — CI Benchmark (Azure Blob, 4 cores pinned), run 33245831329 on
        commit 3aaffb64, 2026-08-29.
  10M — the default supertable scale on an 8-vCPU AMD EPYC 9V74 (AVX-512,
        62 GiB) against Azure Blob, commit 339e621. See benches/README.md.

Every chart therefore labels its scale groups and carries both provenances in
the footnote. Footnote text is drawn straight into SVG, so it must stay plain:
no markdown, no angle brackets.
"""

from __future__ import annotations

from dataclasses import dataclass

# Milliseconds is the single internal unit; the renderer picks µs/ms/s per value.
US = 1 / 1000


@dataclass(frozen=True)
class Bar:
    """One horizontal bar. `group` heads a run of consecutive bars."""

    group: str
    label: str
    ms: float
    cold: bool = False


@dataclass(frozen=True)
class CompareRow:
    """One engine in a comparison: the bar value, its printed label, and
    the extra data columns (matched positionally to the chart meta's
    `columns` headers). Columns render full-size — recall and resident
    memory decide choices and must not hide in sublabel text."""

    name: str
    value: float
    label: str
    cols: tuple[str, ...] = ()
    self_row: bool = False


CI_RUN = "33245831329"
CI_RUN_URL = f"https://github.com/infino-ai/infino/actions/runs/{CI_RUN}"
CI_COMMIT = "3aaffb64"

# Both windows in one line, plain text, for the chart footnotes.
PROVENANCE = (
    f"1M: Azure Blob, 4 cores pinned, CI run {CI_RUN} ({CI_COMMIT}) "
    "· 10M: EPYC 9V74 8 vCPU, Azure Blob, 339e621"
)

# ── Internal latency ────────────────────────────────────────────────────────

VECTOR = {
    "title": "Vector search",
    "subtitle": "1024-d cosine · top-10 · post-drain · recall@10 0.992",
    "footnote": PROVENANCE,
    "bars": [
        Bar("1M", "warm p50", 591 * US),
        Bar("1M", "warm p99", 687 * US),
        Bar("1M", "cold p50", 114, cold=True),
        Bar("10M", "warm p50", 5),
        Bar("10M", "warm p99", 12),
        Bar("10M", "cold p50", 314, cold=True),
    ],
}

FTS = {
    "title": "Full-text search (BM25)",
    "subtitle": "top-10 including row fetch · 1M: single_rare · 10M: median shape",
    "footnote": PROVENANCE,
    "bars": [
        Bar("1M", "warm p50", 125 * US),
        Bar("1M", "warm p99", 143 * US),
        Bar("1M", "cold p50", 16.4, cold=True),
        Bar("10M", "warm p50", 2),
        Bar("10M", "warm p99", 7),
        Bar("10M", "cold p50", 275, cold=True),
    ],
}

SQL = {
    "title": "SQL query shapes",
    "subtitle": "warm p50 · supertable on object storage",
    "footnote": PROVENANCE,
    "bars": [
        Bar("1M", "metadata", 186 * US),
        Bar("1M", "lookup", 4.41),
        Bar("1M", "scan", 5.16),
        Bar("1M", "crosstab", 7.63),
        Bar("10M", "metadata", 260 * US),
        Bar("10M", "lookup", 2.74),
        Bar("10M", "scan", 41.14),
        Bar("10M", "crosstab", 75.14),
    ],
}


# ── Vector modes: memory and latency (dbpedia-1536 over Azure Blob) ────────
#
# Measured on main (3aaffb64) on an 8-vCPU AMD EPYC 9V74 (AVX-512, 62 GiB):
# the flat/ivf serving figures are the bench's own RSS decomposition
# (pinned index + manifest, plus the ivf working set). The float32 bars are
# arithmetic — rows x dims x 4 bytes — the size of the raw vectors before
# any engine touches them, drawn muted as the baseline.
MODES_PROVENANCE = (
    "dbpedia-1536, Azure Blob, EPYC 9V74 8 vCPU, 3aaffb64 "
    "· float32 = rows x 1536 x 4B"
)

MODES_MEMORY = {
    "title": "RAM to serve vector search",
    "subtitle": "all-in: index + manifest + working set · dbpedia-1536",
    "footnote": MODES_PROVENANCE,
    "unit": "mib",
    "ratio_vs_cold": True,
    "legend_warm": "measured, serving",
    "legend_cold": "float32 vectors, for scale",
    "bars": [
        Bar("100K", "float32 vectors", 586, cold=True),
        Bar("100K", "ivf (default)", 423),
        Bar("100K", "flat_ivf", 153),
        Bar("1M", "float32 vectors", 5860, cold=True),
        Bar("1M", "ivf (default)", 3160),
        Bar("1M", "flat_ivf", 841),
    ],
}

MODES_LATENCY = {
    "title": "Vector modes: warm p50 at the recall each serves",
    "subtitle": "top-10 · post-compact · dbpedia-1536",
    "footnote": MODES_PROVENANCE,
    "unit": "ms",
    "bars": [
        Bar("100K", "flat_ivf · recall 0.944", 1.63),
        Bar("100K", "ivf · recall 0.998", 2.28),
        Bar("1M", "flat_ivf · recall 0.938", 20.1),
        Bar("1M", "ivf · recall 0.988", 6.2),
    ],
}


# ── SQL: the same query with and without the search indexes ────────────────
#
# Pairs from the recorded 1M warm battery in benches/README.md ("Plain Scan
# (DataFusion only)" vs "FTS-pushdown"): identical SQL over identical files,
# the only difference being whether the text index turns the predicate into a
# row selection before the scan. The last pair is the honest one: a predicate
# that selects every row gains nothing from an index and pays for it.
SQL_PUSHDOWN = {
    "title": "SQL: the same query, with and without the search indexes",
    "subtitle": "1M rows · warm · identical files · benches/README battery",
    "footnote": "recorded 1M battery, benches/README.md · unsorted key column, min/max defeated",
    "unit": "ms",
    "legend_warm": "with FTS pushdown",
    "legend_cold": "DataFusion scan, same files",
    "bars": [
        Bar("WHERE key = ?", "DataFusion scan", 21.90, cold=True),
        Bar("WHERE key = ?", "FTS pushdown", 1.44),
        Bar("COUNT(*) WHERE key = ?", "DataFusion scan", 22.55, cold=True),
        Bar("COUNT(*) WHERE key = ?", "FTS pushdown", 1.69),
        Bar("AVG(rating) WHERE key = ?", "DataFusion scan", 22.56, cold=True),
        Bar("AVG(rating) WHERE key = ?", "FTS pushdown", 1.84),
    ],
}


# ── Ingest throughput (recorded battery blocks, benches/README.md) ─────────
#
# From the per-battery recorded blocks on main `3aaffb64`, NOT the summary
# table at the top of that file — the summary still carries a pre-#665/#671
# run where vector ingest measured 7.8 K/s. The recorded blocks postdate
# the n_cent, pack-scratch, and pipelined-upload fixes.
INGEST = {
    "title": "Ingest throughput",
    "subtitle": "1M docs · 16 commits · object storage",
    "footnote": "recorded battery blocks, benches/README.md at 3aaffb64",
    "unit": "kps",
    "bars": [
        Bar("1M docs", "vector", 40.6),
        Bar("1M docs", "FTS", 38.7),
        Bar("1M docs", "SQL", 24.2),
    ],
}

# ── flat_ivf vs ivf warm p50 across scales (five-point sweep) ───────────────
CROSSOVER = {
    "title": "flat_ivf vs ivf: warm p50 by table size",
    "subtitle": "top-10 · post-compact · dbpedia-1536",
    "footnote": MODES_PROVENANCE,
    "unit": "ms",
    "legend_warm": "flat_ivf (exhaustive 4-bit scan)",
    "legend_cold": "ivf (routed)",
    "bars": [
        Bar("106K rows", "flat_ivf", 1.63),
        Bar("106K rows", "ivf", 2.28, cold=True),
        Bar("141K rows", "flat_ivf", 2.41),
        Bar("141K rows", "ivf", 1.39, cold=True),
        Bar("189K rows", "flat_ivf", 3.47),
        Bar("189K rows", "ivf", 1.59, cold=True),
        Bar("336K rows", "flat_ivf", 6.90),
        Bar("336K rows", "ivf", 2.51, cold=True),
        Bar("992K rows", "flat_ivf", 20.1),
        Bar("992K rows", "ivf", 6.2, cold=True),
    ],
}


# ── External comparisons ────────────────────────────────────────────────────

VDB_ROWS: list[CompareRow] = [
    CompareRow("Infino", 1.1, "1.1 ms", self_row=True),
    CompareRow("Zilliz Cloud", 2.0, "2.0 ms"),
    CompareRow("Qdrant Cloud", 6.4, "6.4 ms"),
    CompareRow("OpenSearch", 7.2, "7.2 ms"),
    CompareRow("Elastic Cloud", 9.5, "9.5 ms"),
    CompareRow("Pinecone", 13.7, "13.7 ms"),
]

VDB_META = {
    "title": "Vector search vs vector databases",
    "subtitle": "VectorDBBench · Cohere 1M · 768-d · top-100 · serial p99 · lower is faster",
    "url": "https://zilliz.com/vdbbench-leaderboard?dataset=vectorSearch",
    "value_header": "p99",
    # Deployment tier strings carry no signal a reader can act on; the
    # leaderboard link in the README is the place for configuration detail.
}

# (name, version, search ratio, count ratio, is_infino)
SBG_ROWS: list[tuple[str, str, float, float, bool]] = [
    ("Infino", "0.1", 1.19, 0.74, True),
    ("Lucene", "10.5.0", 1.0, 1.0, False),
    ("Tantivy", "0.26", 1.15, 0.80, False),
]

SBG_META = {
    "title": "Full-text vs search libraries",
    "subtitle": "Search Benchmark, the Game · latency vs Lucene = 1.00 · lower is faster",
    "url": "https://tantivy-search.github.io/bench/",
}

SQL_EXT_ROWS: list[CompareRow] = [
    CompareRow("ClickHouse", 6.8, "6.8"),
    CompareRow("DuckDB", 9.8, "9.8"),
    CompareRow("Infino", 12.7, "12.7", self_row=True),
    CompareRow("DataFusion", 17.0, "17.0"),
]

SQL_EXT_META = {
    "title": "SQL vs analytic engines",
    "subtitle": "ClickBench · vCPU-sec per query · hot · c6a.4xlarge · lower is faster",
    "url": (
        "https://benchmark.clickhouse.com/#system=+ClickHouse%7CDuckDB%7CInfino"
        "%7CDataFusion%20%28Parquet%2C%20single%29"
        "&machine=+c6a.4xlarge&cluster_size=-&type=-&metric=hot"
    ),
}

# ── ClickBench, search-engine peer group ────────────────────────────────────
#
# Same suite, same metric, same machine as the analytic chart — one
# number system across both peer groups (Infino prints identically in
# each). SigLens is omitted (no recognition to anchor a comparison) and
# Quickwit is omitted (fails roughly half the suite, so any per-query
# figure over the queries it completes is not comparable); Tantivy's
# library tier is covered by the SBG chart.
CLICKBENCH_SEARCH_ROWS: list[CompareRow] = [
    CompareRow("Infino", 12.7, "12.7", self_row=True),
    CompareRow("ParadeDB", 20.2, "20.2"),
    CompareRow("Elasticsearch", 171.1, "171.1"),
    CompareRow("MongoDB", 7229.8, "7230"),
]

CLICKBENCH_SEARCH_META = {
    "title": "SQL vs search engines",
    "subtitle": "ClickBench · vCPU-sec per query · hot · c6a.4xlarge · lower is faster",
}

EMBED_ROWS: list[CompareRow] = [
    CompareRow("turbovec 2-bit", 0.42, "0.42 ms", ("0.835", "38 MiB")),
    CompareRow("Infino flat_ivf (4-bit)", 1.51, "1.51 ms", ("0.934", "75 MiB"), self_row=True),
    CompareRow("turbovec 4-bit", 1.55, "1.55 ms", ("0.944", "75 MiB")),
    CompareRow("FAISS PQ fastscan", 4.38, "4.38 ms", ("0.672", "74 MiB")),
    CompareRow("FAISS PQ 8-bit", 45.3, "45.3 ms", ("0.943", "76 MiB")),
]

EMBED_META = {
    "title": "Quantized vector indexes vs embedded libraries",
    "subtitle": "dbpedia-1536 · 100K · top-10 · same queries, same ground truth",
    "url": "https://github.com/infino-ai/retrievalbench",
    "value_header": "warm p50",
    "columns": ("recall@10", "resident"),
}
