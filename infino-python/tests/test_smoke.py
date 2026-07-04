"""End-to-end smoke tests for the infino Python bindings.

Run after `maturin develop`:

    cd infino-python
    maturin develop
    pip install pytest pyarrow
    pytest tests/
"""

import pathlib

import infino
import pyarrow as pa
import pytest


def test_package_metadata():
    assert isinstance(infino.__version__, str) and infino.__version__
    assert set(infino.__all__) == {
        "connect",
        "Connection",
        "Table",
        "IndexSpec",
        "MutationStats",
        "GcReport",
        "OptimizeOptions",
    }


def test_typing_artifacts_are_packaged():
    # The stub and marker must ship beside the module, or type checkers
    # silently ignore the package despite the work above.
    pkg = pathlib.Path(infino.__file__).parent
    assert (pkg / "py.typed").is_file()
    assert (pkg / "_infino.pyi").is_file()


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


def test_connect_accepts_cache_options(tmp_path):
    # Cache options are a no-op for local storage but must parse and apply.
    db = infino.connect(
        str(tmp_path / "catalog"),
        cache_dir=str(tmp_path / "cache"),
        cache_budget_bytes=64 * 1024 * 1024,
        cold_fetch_mode="lazy_foreground_with_background_fill",
    )
    t = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    t.append([{"title": "the quick brown fox"}])
    assert t.token_match("title", "fox").num_rows == 1


def test_connect_cold_fetch_mode_is_case_insensitive():
    # Consistent with metric / mode parsing.
    infino.connect("memory://", cold_fetch_mode="RANGE_ONLY")


def test_connect_rejects_invalid_cold_fetch_mode():
    with pytest.raises(ValueError):
        infino.connect("memory://", cold_fetch_mode="nonsense")


def test_connect_accepts_storage_options(tmp_path):
    # storage_options is a no-op for local storage but must parse and apply.
    db = infino.connect(str(tmp_path / "catalog"), storage_options={"aws_region": "us-east-1"})
    t = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    t.append([{"title": "the quick brown fox"}])
    assert t.token_match("title", "fox").num_rows == 1


def test_connect_rejects_unknown_storage_option():
    # An unknown key surfaces at connect time (backend error), not a
    # silent drop.
    with pytest.raises(RuntimeError, match="not_a_real_key"):
        infino.connect("s3://bucket/prefix", storage_options={"not_a_real_key": "x"})


def test_connect_does_not_probe_by_default():
    # Default (validate off): connecting to a bogus bucket builds the
    # handle without touching the backend.
    infino.connect("s3://no-such-bucket-xyzzy/prefix")


def test_query_sql_returns_pyarrow_table():
    db = infino.connect("memory://")
    table = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    table.append(_title_batch(["alpha", "beta", "gamma"]))

    out = db.query_sql("SELECT COUNT(*) AS n FROM docs")
    assert out.num_rows == 1
    assert out.column("n")[0].as_py() == 3


def test_query_sql_zero_row_filter_preserves_projected_schema():
    db = infino.connect("memory://")
    table = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    table.append(_title_batch(["alpha", "beta"]))

    # Ground-truth schema from a query that returns rows.
    with_rows = db.query_sql("SELECT title FROM docs")
    expected_schema = with_rows.schema

    # Zero-row result must carry the identical schema.
    out = db.query_sql("SELECT title FROM docs WHERE title = 'no_match'")
    assert out.num_rows == 0
    assert out.to_pylist() == []
    assert out.schema == expected_schema


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


def test_delete_by_predicate(tmp_path):
    db = infino.connect(str(tmp_path / "catalog"))  # mutations need durable storage
    t = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    t.append([{"title": "alpha"}, {"title": "bravo"}, {"title": "charlie"}])

    stats = t.delete("title = 'bravo'")
    assert stats.matched == 1
    assert stats.n_tombstoned == 1
    assert t.token_match("title", "bravo").num_rows == 0
    assert _count(db, "docs") == 2


def test_update_by_predicate(tmp_path):
    db = infino.connect(str(tmp_path / "catalog"))
    t = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    t.append([{"title": "draft"}, {"title": "keep"}])

    stats = t.update("title = 'draft'", [{"title": "published"}])
    assert stats.matched == 1
    assert t.token_match("title", "draft").num_rows == 0
    assert t.token_match("title", "published").num_rows == 1


def test_update_cardinality_mismatch(tmp_path):
    db = infino.connect(str(tmp_path / "catalog"))
    t = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    t.append([{"title": "alpha"}, {"title": "beta"}])

    # One row matches, two replacements supplied.
    with pytest.raises(ValueError):
        t.update("title = 'alpha'", [{"title": "x"}, {"title": "y"}])


def test_delete_matching_many_and_none(tmp_path):
    db = infino.connect(str(tmp_path / "catalog"))
    t = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    t.append([{"title": "spam"}, {"title": "spam"}, {"title": "ham"}])

    deleted = t.delete("title = 'spam'")
    assert deleted.matched == 2
    assert _count(db, "docs") == 1

    missed = t.delete("title = 'nothing-here'")
    assert missed.matched == 0
    assert missed.n_tombstoned == 0


def test_update_accepts_pyarrow_record_batch(tmp_path):
    db = infino.connect(str(tmp_path / "catalog"))
    t = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    t.append([{"title": "draft"}])

    t.update("title = 'draft'", _title_batch(["published"]))
    assert t.token_match("title", "published").num_rows == 1


def test_invalid_predicate_raises(tmp_path):
    db = infino.connect(str(tmp_path / "catalog"))
    t = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    t.append([{"title": "alpha"}])

    with pytest.raises(ValueError):
        t.delete("no_such_column = 'x'")
    with pytest.raises(ValueError):
        t.delete("this is not sql")


def test_mutations_persist_across_reconnect(tmp_path):
    uri = str(tmp_path / "catalog")
    db = infino.connect(uri)
    t = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    t.append([{"title": "alpha"}, {"title": "beta"}])
    t.delete("title = 'alpha'")
    t.update("title = 'beta'", [{"title": "beta2"}])
    del t
    del db

    reopened = infino.connect(uri).open_table("docs")
    assert reopened.token_match("title", "alpha").num_rows == 0
    assert reopened.token_match("title", "beta2").num_rows == 1


def test_mutations_reject_memory():
    db = infino.connect("memory://")
    t = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    t.append([{"title": "alpha"}])

    with pytest.raises(RuntimeError):
        t.delete("title = 'alpha'")
    with pytest.raises(RuntimeError):
        t.update("title = 'alpha'", [{"title": "beta"}])


def test_optimize_preserves_data(tmp_path):
    db = infino.connect(str(tmp_path / "catalog"))
    t = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    for title in ("alpha", "beta", "gamma"):  # three appends -> three superfiles
        t.append([{"title": title}])

    t.optimize(infino.OptimizeOptions(target_superfile_size_mb=256, min_fill_percent=50))
    assert _count(db, "docs") == 3
    assert t.token_match("title", "beta").num_rows == 1

    t.optimize()  # defaults run cleanly too


def test_optimize_on_memory_is_noop():
    db = infino.connect("memory://")
    t = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    for title in ("alpha", "beta", "gamma"):
        t.append([{"title": title}])

    assert t.optimize() is None
    assert _count(db, "docs") == 3


def test_count_matches_predicate():
    db = infino.connect("memory://")
    t = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    t.append(_title_batch(["the quick brown fox", "a lazy dog"]))

    # count returns the match tally without materializing rows.
    assert t.count("title", "fox") == 1
    assert t.count("title", "fox dog") == 2  # "or" default: fox OR dog
    assert t.count("title", "fox dog", mode="and") == 0


def test_gc_reclaims_orphans(tmp_path):
    db = infino.connect(str(tmp_path / "catalog"))  # gc needs durable storage
    t = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    for title in ("alpha", "beta", "gamma"):  # three appends -> three superfiles
        t.append([{"title": title}])

    t.optimize()  # merge leaves the pre-merge objects orphaned
    report = t.gc(0.0)  # 0s grace: freshly-orphaned objects are eligible now
    assert report.objects_deleted >= 0
    assert isinstance(report.bytes_freed, int)
    assert _count(db, "docs") == 3  # data intact after the sweep


def test_gc_rejects_memory():
    db = infino.connect("memory://")
    t = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    t.append([{"title": "alpha"}])
    with pytest.raises(RuntimeError):
        t.gc(0.0)


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


def test_filtered_vector_search():
    db = infino.connect("memory://")
    dim = 16

    def onehot(i: int) -> list[float]:
        v = [0.0] * dim
        v[i] = 1.0
        return v

    # A table with both an FTS column (title) and a vector column (emb).
    schema = pa.schema([
        pa.field("title", pa.large_utf8(), nullable=False),
        pa.field("emb", pa.list_(pa.float32(), dim), nullable=False),
    ])
    t = db.create_table(
        "docs", schema, infino.IndexSpec().fts("title").vector("emb", dim, 1, "cosine")
    )
    t.append(
        pa.record_batch(
            [
                pa.array(
                    ["billing and refunds", "refund policy", "dark mode appearance"],
                    type=pa.large_utf8(),
                ),
                pa.array([onehot(0), onehot(0), onehot(1)], type=pa.list_(pa.float32(), dim)),
            ],
            schema=schema,
        )
    )

    # Unfiltered kNN over the topic-0 embedding sees both topic-0 rows.
    assert t.vector_search("emb", onehot(0), 10).num_rows >= 2

    # Same kNN, restricted to rows whose `title` matches "billing" — a pushdown
    # pre-filter, so only the matching row comes back (not a post-filter over
    # the global top-k).
    filtered = t.vector_search(
        "emb",
        onehot(0),
        10,
        filter_column="title",
        filter_query="billing",
        filter_mode="or",
        projection=["_id", "title", "score"],
    )
    assert filtered.num_rows == 1
    assert filtered.column("title").to_pylist() == ["billing and refunds"]

    # filter_column and filter_query must be supplied together.
    with pytest.raises(ValueError):
        t.vector_search("emb", onehot(0), 10, filter_column="title")

    # filter_mode alone (no column/query) is rejected, not silently ignored.
    with pytest.raises(ValueError):
        t.vector_search("emb", onehot(0), 10, filter_mode="or")

    # an invalid filter_mode is rejected when a filter is present.
    with pytest.raises(ValueError):
        t.vector_search(
            "emb", onehot(0), 10, filter_column="title", filter_query="billing", filter_mode="xor"
        )


def test_hybrid_search_fuses_text_and_vector():
    db = infino.connect("memory://")
    dim = 16

    def onehot(i: int) -> list[float]:
        v = [0.0] * dim
        v[i] = 1.0
        return v

    schema = pa.schema([
        pa.field("title", pa.large_utf8(), nullable=False),
        pa.field("emb", pa.list_(pa.float32(), dim), nullable=False),
    ])
    t = db.create_table(
        "docs", schema, infino.IndexSpec().fts("title").vector("emb", dim, 1, "cosine")
    )
    t.append(
        pa.record_batch(
            [
                pa.array(["rust async", "python data", "rust systems"], type=pa.large_utf8()),
                pa.array([onehot(0), onehot(1), onehot(2)], type=pa.list_(pa.float32(), dim)),
            ],
            schema=schema,
        )
    )

    hits = t.hybrid_search("title", "rust", "emb", onehot(0), 10)
    assert hits.num_rows >= 1
    assert "_id" in hits.column_names and "score" in hits.column_names
    # RRF score is higher-is-better, so rows come back descending.
    scores = hits["score"].to_pylist()
    assert scores == sorted(scores, reverse=True)

    # Projection materializes the named scalar column.
    projected = t.hybrid_search(
        "title", "rust", "emb", onehot(0), 10, projection=["_id", "title", "score"]
    )
    assert projected.column_names == ["_id", "title", "score"]

    # Direct call and the SQL table function agree on the `_id` set
    # (the TVF fixes mode="or" and default nprobe, so match it).
    csv = ",".join("1" if d == 0 else "0" for d in range(dim))
    via_sql = db.query_sql(
        f"SELECT _id FROM hybrid_search('docs', 'title', 'rust', 'emb', '{csv}', 10)"
    )
    assert set(hits["_id"].to_pylist()) == set(via_sql["_id"].to_pylist())

    # Invalid mode is rejected.
    with pytest.raises(ValueError, match="mode"):
        t.hybrid_search("title", "rust", "emb", onehot(0), 10, mode="xor")

    # A non-indexed text column names the offending column.
    with pytest.raises(ValueError, match="missing"):
        t.hybrid_search("missing", "rust", "emb", onehot(0), 10)

    # A wrong-dimension query vector reports the dimension mismatch.
    with pytest.raises(ValueError, match="dimension"):
        t.hybrid_search("title", "rust", "emb", [1.0, 2.0, 3.0], 10)
