# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Infino Authors
#
# `bm25_search_fields` through the Python binding: each clause scores a row by
# its best column, a row's score is the sum over the clauses it matches, and
# the method, the SQL table function and per-column `bm25_search` agree.

import infino
import pyarrow as pa
import pytest

ROWS = [
    ("quick fox", "a lazy dog sleeps"),
    ("lazy dog", "the quick brown fox jumps over the lazy dog"),
    ("hound", "the fox and the hound"),
    ("nothing here", "nothing here either"),
]


def _docs():
    schema = pa.schema(
        [pa.field("title", pa.large_utf8(), nullable=False), pa.field("body", pa.large_utf8(), nullable=False)]
    )
    db = infino.connect("memory://")
    t = db.create_table("docs", schema, infino.IndexSpec().fts("title").fts("body"))
    titles, bodies = zip(*ROWS)
    t.append(
        pa.record_batch(
            [pa.array(list(titles), type=pa.large_utf8()), pa.array(list(bodies), type=pa.large_utf8())],
            schema=schema,
        )
    )
    return db, t


def _scores(table: pa.Table) -> dict:
    return dict(zip(table.column("_id").to_pylist(), table.column("score").to_pylist()))


def _expected(t, clauses, weights=None):
    weights = weights or {}
    total: dict = {}
    for query, fields in clauses:
        best: dict = {}
        for f in fields:
            for rid, s in _scores(t.bm25_search(f, query, 100)).items():
                best[rid] = max(best.get(rid, 0.0), s * weights.get(f, 1.0))
        for rid, s in best.items():
            total[rid] = total.get(rid, 0.0) + s
    return total


def test_scores_are_best_column_per_clause_summed_over_clauses():
    _, t = _docs()
    clauses = [("fox", ["title", "body"]), ("hound", ["title", "body"])]
    got = _scores(t.bm25_search_fields(clauses, 10))
    want = _expected(t, clauses)
    assert got.keys() == want.keys()
    for rid, s in want.items():
        assert got[rid] == pytest.approx(s, rel=1e-5)


def test_weights_multiply_a_columns_scores():
    _, t = _docs()
    clauses = [{"query": "fox", "fields": ["title", "body"]}]
    weights = {"title": 3.0}
    got = _scores(t.bm25_search_fields(clauses, 10, weights=weights))
    want = _expected(t, [("fox", ["title", "body"])], weights)
    assert got.keys() == want.keys()
    for rid, s in want.items():
        assert got[rid] == pytest.approx(s, rel=1e-5)


def test_and_mode_keeps_rows_matching_every_clause():
    _, t = _docs()
    clauses = [("fox", ["title", "body"]), ("hound", ["title", "body"])]
    either = t.bm25_search_fields(clauses, 10)
    both = t.bm25_search_fields(clauses, 10, mode="and")
    assert either.num_rows == 3
    assert both.num_rows == 1


def test_k_truncates_best_first():
    _, t = _docs()
    clauses = [("fox", ["title", "body"])]
    full = t.bm25_search_fields(clauses, 10)
    top = t.bm25_search_fields(clauses, 1)
    assert top.num_rows == 1
    assert top.column("_id")[0] == full.column("_id")[0]
    assert full.column("score").to_pylist() == sorted(full.column("score").to_pylist(), reverse=True)


def test_projection_returns_the_named_columns():
    _, t = _docs()
    out = t.bm25_search_fields([("hound", ["title"])], 10, projection=["title", "score"])
    assert out.column_names == ["title", "score"]
    assert out.column("title").to_pylist() == ["hound"]


def test_sql_table_function_matches_the_method():
    db, t = _docs()
    clauses = [("fox", ["title", "body"]), ("hound", ["body"])]
    method = _scores(t.bm25_search_fields(clauses, 10, weights={"body": 2.0}))
    sql = db.query_sql(
        "SELECT _id, score FROM bm25_search_fields('docs', "
        '\'[{"query": "fox", "fields": ["title", "body"]}, {"query": "hound", "fields": ["body"]}]\', '
        "10, 'or', '{\"body\": 2.0}')"
    )
    got = _scores(sql)
    assert got.keys() == method.keys()
    for rid, s in method.items():
        assert got[rid] == pytest.approx(s, rel=1e-6)


@pytest.mark.parametrize(
    "clauses, kwargs",
    [
        ([], {}),
        ([("fox", [])], {}),
        ([("fox", ["missing"])], {}),
        ([("fox", ["title"])], {"weights": {"title": 0.0}}),
        ([("fox", ["title"])], {"mode": "xor"}),
    ],
)
def test_bad_arguments_are_rejected(clauses, kwargs):
    _, t = _docs()
    with pytest.raises(Exception):
        t.bm25_search_fields(clauses, 10, **kwargs)
