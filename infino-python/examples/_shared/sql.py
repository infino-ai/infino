"""Small SQL helpers shared by the example notebooks."""


def sql_lit(value: str) -> str:
    """Quote a string as a SQL literal (single quotes doubled)."""
    return "'" + value.replace("'", "''") + "'"


def query(db, sql: str) -> dict:
    """Run a SQL query and return columns as a dict; `{}` if no rows matched."""
    res = db.query_sql(sql)
    return res.to_pydict() if res.num_rows else {}
