"""Small SQL helpers shared by the examples that query Infino via `query_sql`.

Both are one-liners, but they appear in several notebooks, so they live here to
stay consistent — especially `query`, which guards the empty-result case.
"""


def sql_lit(value: str) -> str:
    """Quote a string as a SQL literal (single quotes doubled)."""
    return "'" + value.replace("'", "''") + "'"


def query(db, sql: str) -> dict:
    """Run a SQL query and return columns as a dict; `{}` if no rows matched."""
    res = db.query_sql(sql)
    return res.to_pydict() if res.num_rows else {}
