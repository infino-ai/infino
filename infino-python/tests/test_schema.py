"""The schema document: read, grown from data, and changed through one write."""

import infino
import pyarrow as pa
import pytest


def _title_schema() -> pa.Schema:
    return pa.schema([pa.field("title", pa.large_utf8(), nullable=False)])


def _names(doc: dict) -> list[tuple[str, int]]:
    return [(f["name"], f["id"]) for f in doc["fields"]]


def test_schema_grows_from_data_and_reads_back():
    db = infino.connect("memory://")
    docs = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    before = db.schema("docs")
    assert _names(before) == [("title", 1)]
    assert before["schema_id"] == 1
    assert before["fields"][0]["type"] == "large_utf8"
    assert before["fields"][0]["index"]["kind"] == "fts"

    docs.append(pa.table({"title": ["a"]}, schema=_title_schema()))
    docs.append(
        pa.table(
            {"title": pa.array(["b"], pa.large_utf8()), "score": pa.array([7], pa.int64())}
        )
    )
    after = db.schema("docs")
    assert _names(after) == [("title", 1), ("score", 2)]
    assert after["fields"][1] == {"id": 2, "name": "score", "type": "i64", "nullable": True}
    assert after["schema_id"] == 2

    rows = db.query_sql("SELECT title, score FROM docs ORDER BY title").to_pylist()
    assert rows == [{"title": "a", "score": None}, {"title": "b", "score": 7}]


def test_schema_write_adds_renames_drops_and_guards_with_the_expected_id():
    db = infino.connect("memory://")
    db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))

    doc = db.schema("docs", {"fields": [{"name": "score", "type": "i64"}]})
    assert _names(doc) == [("title", 1), ("score", 2)]

    # The read document applies as a no-op.
    assert db.schema("docs", doc) == doc

    renamed = db.schema("docs", {"fields": [{"id": 2, "name": "points"}]}, doc["schema_id"])
    assert _names(renamed) == [("title", 1), ("points", 2)]

    with pytest.raises(infino.ConflictError):
        db.schema("docs", {"fields": [{"name": "tag", "type": "large_utf8"}]}, doc["schema_id"])

    dropped = db.schema("docs", {"fields": [{"name": "points", "dropped": True}]})
    assert _names(dropped) == [("title", 1)]
    assert dropped["tombstoned"] == [2]

    with pytest.raises(ValueError, match="no live column has id 9"):
        db.schema("docs", {"fields": [{"id": 9, "name": "x", "type": "i64"}]})


def test_schema_write_creates_an_absent_table():
    db = infino.connect("memory://")
    doc = db.schema(
        "fresh",
        {
            "fields": [
                {"name": "title", "type": "large_utf8", "nullable": False},
                {"name": "score", "type": "i64"},
            ],
            "max_fields": 50,
        },
    )
    assert _names(doc) == [("title", 1), ("score", 2)]
    assert doc["max_fields"] == 50
    assert db.list_tables() == ["fresh"]
    with pytest.raises(ValueError, match="fresh"):
        db.create_table("fresh", _title_schema(), infino.IndexSpec())
