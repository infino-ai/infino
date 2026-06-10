"""End-to-end smoke tests for the infino Python bindings.

Run after `maturin develop`:

    cd infino-python
    maturin develop
    pip install pytest pyarrow
    pytest tests/
"""

import infino
import pyarrow as pa
import pytest


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
    assert hits.num_rows == 1
    assert "_id" in hits.column_names and "score" in hits.column_names

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


def _count(db, table: str) -> int:
    out = db.query_sql(f"SELECT COUNT(*) AS n FROM {table}")
    return out.column("n")[0].as_py()


def test_append_accepts_pyarrow_table():
    db = infino.connect("memory://")
    t = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    table = pa.Table.from_batches([_title_batch(["alpha", "beta"]), _title_batch(["gamma"])])
    t.append(table)  # a multi-chunk Table → one commit
    assert _count(db, "docs") == 3


def test_append_accepts_list_of_dicts():
    db = infino.connect("memory://")
    t = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    t.append([{"title": "the quick brown fox"}, {"title": "a lazy dog"}])
    assert _count(db, "docs") == 2
    assert t.bm25_search("title", "fox", 10).num_rows == 1


def test_append_accepts_pandas_dataframe():
    pd = pytest.importorskip("pandas")
    db = infino.connect("memory://")
    t = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    t.append(pd.DataFrame({"title": ["hello world", "goodbye world"]}))
    assert _count(db, "docs") == 2


def test_vector_search_end_to_end():
    db = infino.connect("memory://")
    dim = 16  # infino requires vector dim in [16, 4096]

    def onehot(i: int) -> list[float]:
        v = [0.0] * dim
        v[i] = 1.0
        return v

    schema = pa.schema([pa.field("emb", pa.list_(pa.float32(), dim), nullable=False)])
    t = db.create_table("vecs", schema, infino.IndexSpec().vector("emb", dim, 1, "cosine"))
    vecs = [onehot(0), onehot(1), onehot(2)]
    t.append(pa.record_batch([pa.array(vecs, type=pa.list_(pa.float32(), dim))], schema=schema))

    hits = t.vector_search("emb", onehot(0), 10)
    assert hits.num_rows >= 1
    assert "_id" in hits.column_names and "score" in hits.column_names
