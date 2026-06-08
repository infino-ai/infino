"""End-to-end smoke tests for the infino Python bindings.

Run after `maturin develop`:

    cd infino-python
    maturin develop
    pip install pytest pyarrow
    pytest tests/
"""

import infino
import pyarrow as pa


def _title_schema() -> pa.Schema:
    # Matches the core's user schema (title only; `_id` is auto-injected).
    return pa.schema([pa.field("title", pa.large_utf8(), nullable=False)])


def _title_batch(titles: list[str]) -> pa.RecordBatch:
    # Build from the exact schema so nullability matches what
    # `create_table` declared (append requires an exact schema match).
    return pa.record_batch([pa.array(titles, type=pa.large_utf8())], schema=_title_schema())


def test_memory_roundtrip():
    db = infino.connect("memory://")
    spec = infino.IndexSpec().fts("title")
    table = db.create_table("docs", _title_schema(), spec)
    table.append(_title_batch(["the quick brown fox", "a lazy dog"]))

    assert db.list_tables() == ["docs"]

    # Re-open by name and search.
    reopened = db.open_table("docs")
    hits = reopened.bm25_search("title", "fox", 10)
    assert len(hits) == 1
    assert "_id" in hits[0] and "score" in hits[0]

    db.drop_table("docs")
    assert db.list_tables() == []


def test_query_sql_returns_pyarrow_table():
    db = infino.connect("memory://")
    table = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    table.append(_title_batch(["alpha", "beta", "gamma"]))

    out = db.query_sql("SELECT COUNT(*) AS n FROM docs")
    assert out.num_rows == 1
    assert out.column("n")[0].as_py() == 3


def test_query_sql_bm25_tvf():
    db = infino.connect("memory://")
    table = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    table.append(_title_batch(["the quick brown fox", "a lazy dog"]))

    out = db.query_sql("SELECT _id, score FROM bm25_search('docs', 'title', 'fox', 10)")
    assert out.num_rows == 1


def test_localfs_persists_across_reconnect(tmp_path):
    uri = str(tmp_path / "catalog")
    db = infino.connect(uri)
    table = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    table.append(_title_batch(["a lazy sleeping fox"]))
    del table
    del db

    db2 = infino.connect(uri)
    assert db2.list_tables() == ["docs"]
    hits = db2.open_table("docs").bm25_search("title", "fox", 10)
    assert len(hits) == 1


def test_unknown_table_raises():
    db = infino.connect("memory://")
    try:
        db.open_table("nope")
        assert False, "expected KeyError"
    except KeyError:
        pass
