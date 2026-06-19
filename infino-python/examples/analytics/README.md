# Analytics

Aggregate **and** search the same data over a **single**
[Infino](https://pypi.org/project/infino/) table — one engine instead of a
separate analytics database and search cluster:

- **`query_sql`** — `GROUP BY` time-series, top-N, and leaderboards with
  `COUNT` / `AVG` / `MAX`.
- **`bm25_search`** — ranked full-text search over titles, no `LIKE` scan.
- **Both at once** — `bm25_search` is a SQL table function, so search results
  feed straight into `JOIN` + `GROUP BY`.

The dataset is Hacker News stories (title, author, points, comments, timestamp)
from the HuggingFace Hub, indexed on local disk. This example uses no embeddings
— it's the SQL and full-text path.

> Setup and the shared helpers are covered in the
> [examples README](../README.md). Run that setup once, then open the notebook.

## Search + analytics in one query

The payoff: full-text search composes with SQL aggregation. Counting how often a
topic hit the front page, by year, is one query —
`bm25_search('stories', 'title', 'google', 500)` joined back to the table and
grouped by year:

```
Stories mentioning 'google' in their title, by year:
  2016:  7 stories, avg 39 points
  2019: 11 stories, avg 17 points
  2021:  7 stories, avg 17 points
```

## Example

| # | Example | What it shows | Dataset |
| - | ------- | ------------- | ------- |
| 1 | [`01_hackernews_sql_search.ipynb`](01_hackernews_sql_search.ipynb) | SQL analytics (`GROUP BY` time-series, top-N, leaderboard) + `bm25_search` full-text, and the two fused in one query | Hacker News |
