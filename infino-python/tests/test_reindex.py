"""The index-repair surface: `index_staleness`, `reindex_plan` and `reindex`."""

import pathlib
import shutil

import infino
import pyarrow as pa
import pytest

# A three-superfile table whose full-text index an older engine wrote, so it
# is genuinely behind what this engine writes. Shared with the crate's tests.
OLD_FORMAT_FIXTURE = (
    pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "old_format_fts_table"
)
# Superfiles in that fixture.
OLD_FORMAT_SUPERFILES = 3
# A term every row of the fixture carries.
SHARED_TERM = "shared"
# Upper bound on hits for a fixture-wide search; above its row count.
ALL_HITS = 1000
# A seal-takeover age past what 32 bits of milliseconds can hold (~49 days).
LONG_SEAL_TIMEOUT_MS = 2**40
# Where the catalog keeps its own records, beside each table's subtree.
CATALOG_DIR = "_catalog"


def _title_schema() -> pa.Schema:
    return pa.schema([pa.field("title", pa.large_utf8(), nullable=False)])


def _old_format_table(root: pathlib.Path) -> infino.Table:
    """Register a `title` table under `root`, then swap its subtree for a copy
    of the old-format fixture, so the catalog opens the stale table."""
    db = infino.connect(str(root))
    db.create_table("legacy", _title_schema(), infino.IndexSpec().fts("title"))
    [location] = [p for p in root.iterdir() if p.name != CATALOG_DIR]
    shutil.rmtree(location)
    shutil.copytree(OLD_FORMAT_FIXTURE, location)
    # A fresh connection, so no handle cached over the replaced subtree.
    return infino.connect(str(root)).open_table("legacy")


def _hits(table: infino.Table, term: str) -> int:
    return table.bm25_search("title", term, k=ALL_HITS).num_rows


@pytest.mark.parametrize("call", ["reindex", "reindex_plan", "index_staleness"])
def test_reindex_refuses_a_memory_table(call):
    db = infino.connect("memory://")
    table = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    with pytest.raises(ValueError, match="requires durable storage"):
        getattr(table, call)()


def test_an_unknown_mode_is_rejected(tmp_path):
    table = infino.connect(str(tmp_path)).create_table(
        "docs", _title_schema(), infino.IndexSpec().fts("title")
    )
    with pytest.raises(ValueError, match="unknown reindex mode"):
        table.reindex_plan(infino.ReindexOptions(mode="rebuild"))


def test_a_table_this_engine_wrote_is_current(tmp_path):
    table = infino.connect(str(tmp_path)).create_table(
        "docs", _title_schema(), infino.IndexSpec().fts("title")
    )
    table.append([{"title": "a fresh row"}])
    table.append([{"title": "another fresh row"}])

    staleness = table.index_staleness()
    assert staleness.is_current
    assert staleness.superfiles == 2
    assert staleness.needing_rewrite == 0
    assert staleness.awaiting_reanalysis == 0
    assert staleness.bytes_to_rewrite == 0
    assert staleness.unrepairable_columns == []
    assert staleness.inconsistent_footers == []

    assert table.reindex_plan() == []
    report = table.reindex()
    assert report.rewritten == 0
    assert report.already_current == 2


def test_the_plan_names_exactly_what_the_run_repairs(tmp_path):
    table = _old_format_table(tmp_path)
    before = _hits(table, SHARED_TERM)
    assert before > 0

    staleness = table.index_staleness()
    assert not staleness.is_current, staleness
    assert staleness.superfiles == OLD_FORMAT_SUPERFILES

    # Every fixture superfile holds terms an older analysis produced, so the
    # default mode resolves each one to a re-analysis.
    plan = table.reindex_plan()
    assert len(plan) == OLD_FORMAT_SUPERFILES == staleness.awaiting_reanalysis
    assert len({p.superfile_id for p in plan}) == len(plan)
    assert all(p.mode == "reanalyze" for p in plan)
    assert all(p.live_bytes > 0 for p in plan)

    report = table.reindex()
    assert report.rewritten == len(plan)
    assert report.already_current == OLD_FORMAT_SUPERFILES - len(plan)
    assert report.held_by_another_run == 0
    assert report.awaiting_reanalysis == 0

    # Repaired in place: nothing left to plan, and every row still answers.
    assert table.index_staleness().is_current
    assert table.reindex_plan() == []
    assert _hits(table, SHARED_TERM) == before


def test_a_rewrite_leaves_analysis_stale_superfiles_reported(tmp_path):
    # The fixture's layout is current and its terms are not, so a layout-only
    # run has nothing it can repair and must say so rather than rewrite.
    table = _old_format_table(tmp_path)
    rewriting = infino.ReindexOptions(mode="rewrite")

    staleness = table.index_staleness(rewriting)
    assert staleness.needing_rewrite == 0
    assert staleness.bytes_to_rewrite == 0
    assert staleness.awaiting_reanalysis == OLD_FORMAT_SUPERFILES

    assert table.reindex_plan(rewriting) == []
    report = table.reindex(rewriting)
    assert report.rewritten == 0
    assert report.awaiting_reanalysis == OLD_FORMAT_SUPERFILES
    assert not table.index_staleness().is_current


def test_a_long_seal_timeout_is_passed_through(tmp_path):
    table = _old_format_table(tmp_path)
    report = table.reindex(infino.ReindexOptions(stale_seal_timeout_ms=LONG_SEAL_TIMEOUT_MS))
    assert report.rewritten == OLD_FORMAT_SUPERFILES


def test_a_negative_seal_timeout_is_rejected():
    with pytest.raises(OverflowError):
        infino.ReindexOptions(stale_seal_timeout_ms=-1)
