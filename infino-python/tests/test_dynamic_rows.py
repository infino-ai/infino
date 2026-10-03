"""Rows as documents: nested dicts, lists and new keys grow the schema, and the
document the engine freezes for the shared corpus is the one the Rust and
Node suites freeze."""

import json
import pathlib

import infino
import pyarrow as pa
import pytest

FIXTURES = pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "dynamic"


def _title_schema() -> pa.Schema:
    return pa.schema([pa.field("title", pa.large_utf8(), nullable=False)])


def _corpus() -> list[dict]:
    lines = (FIXTURES / "rows.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_the_corpus_freezes_the_shared_document():
    db = infino.connect("memory://")
    docs = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    docs.append(_corpus())
    frozen = json.loads((FIXTURES / "schema.json").read_text())
    assert db.schema("docs") == frozen
    rows = db.query_sql('SELECT title, views, "author.name" FROM docs ORDER BY _id').to_pylist()
    assert [r["views"] for r in rows] == [10, 25, None, 7]
    assert [r["author.name"] for r in rows] == ["ann", "bob", "cy", None]


def test_documents_grow_the_schema_and_disagreements_are_refused():
    db = infino.connect("memory://")
    docs = db.create_table("docs", _title_schema(), infino.IndexSpec())
    docs.append([{"title": "a", "n": 1, "tags": ["x", "y"], "who": {"name": "ann"}}])
    doc = db.schema("docs")
    by_name = {f["name"]: f for f in doc["fields"]}
    assert by_name["n"]["type"] == "i64"
    assert by_name["tags"]["type"] == "list"
    assert by_name["who.name"]["type"] == "large_utf8"

    # A Python float literal does not fit the Int64 column, nor does 5.0.
    with pytest.raises(ValueError, match="Int64"):
        docs.append([{"title": "b", "n": 1.5}])
    with pytest.raises(ValueError, match="Int64"):
        docs.append([{"title": "b", "n": 5.0}])
    # An omitted nullable key is null; a new key adds a column.
    docs.append([{"title": "c", "extra": True}])
    rows = db.query_sql("SELECT title, n, extra FROM docs ORDER BY _id").to_pylist()
    assert rows == [
        {"title": "a", "n": 1, "extra": None},
        {"title": "c", "n": None, "extra": True},
    ]
