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


def test_a_schema_refusal_names_its_kind():
    """Every schema refusal raises one class carrying the variant on `kind`.

    The message names the column, cap or version at fault and is written to
    be read; `kind` is what a program matches on, so a caller can tell a cap
    breach from a type mismatch without parsing prose.
    """
    db = infino.connect("memory://")
    docs = db.create_table("docs", _title_schema(), infino.IndexSpec())
    docs.append([{"title": "a", "n": 1}])

    with pytest.raises(infino.SchemaError) as caught:
        docs.append([{"title": "b", "n": "x"}])
    assert caught.value.kind == "TypeMismatch"
    assert "n" in str(caught.value)

    with pytest.raises(infino.SchemaError) as caught:
        docs.append([{"title": "c", "xs": [1, "a"]}])
    assert caught.value.kind == "MixedArray"


def test_a_schema_refusal_is_still_a_value_error():
    """`SchemaError` is based on `ValueError`, which is what these refusals
    raised before they had a class of their own, so code catching the old
    thing keeps working."""
    db = infino.connect("memory://")
    docs = db.create_table("docs", _title_schema(), infino.IndexSpec())
    docs.append([{"title": "a", "n": 1}])
    with pytest.raises(ValueError):
        docs.append([{"title": "b", "n": "x"}])
    assert issubclass(infino.SchemaError, ValueError)


def test_a_frames_missing_value_is_stored_as_null():
    """A pandas frame marks a missing number with NaN, which JSON cannot
    spell. It means the row carries nothing there, so it is written as null —
    the same as a key a dict leaves out."""
    import pandas as pd

    db = infino.connect("memory://")
    docs = db.create_table("docs", _title_schema(), infino.IndexSpec())
    docs.append(pd.DataFrame({"title": ["a", "b"], "score": [1.5, float("nan")]}))
    rows = db.query_sql("SELECT title, score FROM docs ORDER BY _id").to_pylist()
    assert rows[0]["score"] == 1.5
    assert rows[1]["score"] is None, "NaN is the absence of a value, so null"


def test_only_a_frames_nan_is_read_as_missing():
    """An infinity is a number a Float64 column holds, and a NaN a caller
    wrote in a list of dicts is a value rather than pandas' marker for one
    they left out. Neither is rewritten; both reach the column as sent."""
    import math

    import pandas as pd
    import pyarrow as pa

    scored = pa.schema(
        [
            pa.field("title", pa.large_utf8(), nullable=False),
            pa.field("score", pa.float64(), nullable=True),
        ]
    )

    db = infino.connect("memory://")
    docs = db.create_table("docs", scored, infino.IndexSpec())
    docs.append([{"title": "a", "score": math.inf}, {"title": "b", "score": float("nan")}])
    rows = db.query_sql("SELECT score FROM docs ORDER BY _id").to_pylist()
    assert rows[0]["score"] == math.inf, "an infinity is stored, not nulled"
    assert math.isnan(rows[1]["score"]), "a NaN a caller wrote is stored too"

    # Including in a frame: only the missing marker is a missing value.
    db = infino.connect("memory://")
    docs = db.create_table("docs", scored, infino.IndexSpec())
    docs.append(pd.DataFrame({"title": ["a", "b"], "score": [math.inf, float("nan")]}))
    rows = db.query_sql("SELECT score FROM docs ORDER BY _id").to_pylist()
    assert rows[0]["score"] == math.inf
    assert rows[1]["score"] is None


def test_a_value_json_cannot_spell_on_a_new_column_is_refused():
    """The typed path fills columns the table declares and invents none, so a
    value it carries cannot also bring a column with it. Before, the whole
    column was silently dropped."""
    import math

    db = infino.connect("memory://")
    docs = db.create_table("docs", _title_schema(), infino.IndexSpec())
    with pytest.raises(ValueError, match="score"):
        docs.append([{"title": "a", "score": math.inf}])


def test_a_frame_with_a_nan_is_serialized_twice_not_three_times():
    """The strict dump that detects the NaN is unavoidable, and the sanitized
    one that follows is the bytes we parse. A third, thrown away only to
    prove the result serializes, is a full pass over the frame for nothing —
    and a million-row frame pays for each one."""
    import json as json_module

    calls = []
    original = json_module.dumps

    def counting(*args, **kwargs):
        calls.append(kwargs.get("allow_nan"))
        return original(*args, **kwargs)

    import pandas as pd

    db = infino.connect("memory://")
    docs = db.create_table("docs", _title_schema(), infino.IndexSpec())
    json_module.dumps = counting
    try:
        # A frame, since that is where a NaN is a missing value. One row
        # carries a real number so the column exists to read back: a column
        # whose every value is null is never created.
        docs.append(pd.DataFrame({"title": ["a", "b"], "score": [1.5, float("nan")]}))
    finally:
        json_module.dumps = original

    assert len(calls) == 2, f"one to detect, one to serialize; got {len(calls)}"
    rows = db.query_sql("SELECT score FROM docs ORDER BY _id").to_pylist()
    assert [r["score"] for r in rows] == [1.5, None]


def test_a_frame_without_a_nan_is_serialized_once():
    """The common path pays nothing for the NaN handling: no sanitizing walk,
    and no second dump."""
    import json as json_module

    calls = []
    original = json_module.dumps

    def counting(*args, **kwargs):
        calls.append(kwargs.get("allow_nan"))
        return original(*args, **kwargs)

    import pandas as pd

    db = infino.connect("memory://")
    docs = db.create_table("docs", _title_schema(), infino.IndexSpec())
    json_module.dumps = counting
    try:
        docs.append(pd.DataFrame({"title": ["a", "b"], "score": [1.5, 2.5]}))
    finally:
        json_module.dumps = original

    assert len(calls) == 1, f"nothing but the one dump; got {len(calls)}"
