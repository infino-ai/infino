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


def test_declared_columns_keep_the_types_json_cannot_carry():
    """Rows that stay inside the declared schema are typed by that schema.

    `json.dumps` raises on bytes, Decimal and datetime, and the engine's own
    document mapper has no arm for binary or decimal, so routing every
    list[dict] through JSON lost column types the table had declared. Rows
    naming only declared columns go through pyarrow instead, which carries
    them.
    """
    import datetime
    import decimal

    db = infino.connect("memory://")
    schema = pa.schema(
        [
            pa.field("title", pa.large_utf8(), nullable=False),
            pa.field("payload", pa.binary(), nullable=True),
            pa.field("price", pa.decimal128(12, 2), nullable=True),
            pa.field("seen", pa.timestamp("ms"), nullable=True),
        ]
    )
    t = db.create_table("docs", schema, infino.IndexSpec())
    t.append(
        [
            {
                "title": "a",
                "payload": b"\x00\x01",
                "price": decimal.Decimal("12.34"),
                "seen": datetime.datetime(2026, 1, 2, 3, 4, 5),
            }
        ]
    )

    out = db.query_sql('SELECT payload, price, seen FROM docs').to_pylist()
    assert out[0]["payload"] == b"\x00\x01"
    assert out[0]["price"] == decimal.Decimal("12.34")
    assert out[0]["seen"] == datetime.datetime(2026, 1, 2, 3, 4, 5)

    # A row naming a column the table does not have still grows the schema,
    # through the document path.
    t.append([{"title": "b", "extra": 7}])
    names = [f["name"] for f in db.schema("docs")["fields"]]
    assert "extra" in names

    # But a new column cannot arrive carrying a value no document can spell:
    # the binding would have to invent its type, and by pyarrow's rules
    # rather than the engine's. Declare it first.
    with pytest.raises(ValueError, match="is new"):
        t.append([{"title": "c", "blob": b"\x09"}])

    # And a declared column still refuses a value that would be truncated,
    # rather than quietly changing it.
    with pytest.raises(ValueError, match="(?i)data loss|truncat"):
        t.append([{"title": "d", "price": decimal.Decimal("1.234"), "payload": b"\x01"}])


def test_a_missing_float_appends_as_null():
    """A float column with a gap is the ordinary way to meet NaN, and `NaN` is
    not JSON. The document path has to hand those rows to the typed route
    rather than serialize something the engine cannot parse back."""
    db = infino.connect("memory://")
    t = db.create_table(
        "docs",
        pa.schema(
            [
                pa.field("title", pa.large_utf8(), nullable=False),
                pa.field("score", pa.float64(), nullable=True),
            ]
        ),
        infino.IndexSpec(),
    )

    t.append([{"title": "a", "score": 1.0}, {"title": "b", "score": float("nan")}])
    rows = db.query_sql("SELECT title, score FROM docs ORDER BY _id").to_pylist()
    assert [r["title"] for r in rows] == ["a", "b"]
    assert rows[0]["score"] == 1.0
    assert rows[1]["score"] is None or rows[1]["score"] != rows[1]["score"]
