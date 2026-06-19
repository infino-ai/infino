# Mutations

Infino tables are **mutable**. You `append`, `update`, and `delete` rows over a
**single** [Infino](https://pypi.org/project/infino/) table and the full-text
index and SQL views stay correct — durably, with no rebuild:

- **`update`** — replace matching rows in place; SQL and search reflect the
  change immediately.
- **`delete`** — remove matching rows; they leave SQL and full-text search at
  once.
- **`optimize`** — compact the storage into fewer, fuller files, without
  changing query results.

`MutationStats` reports how many rows matched and changed. Every mutation is
committed to disk (object storage in production), so the changes survive a
reconnect — one engine, no separate store to update and keep in sync.

> Setup and the shared helpers are covered in the
> [examples README](../README.md). Run that setup once, then open the notebook.

## The life cycle in one run

```
updated 1 row(s):  price $15.99 -> $8.00      # update, reflected in SQL at once
clearance: delete price < $6.99
  matched 114, removed 114                     # rows leave search and SQL
  count 1200 -> 1086   cleared item searchable? False
after optimize: 1086 products (unchanged)      # compaction keeps results identical
reopened catalog: 1086 products, item still $8.00   # every change persisted
```

## Example

| # | Example | What it shows | Dataset |
| - | ------- | ------------- | ------- |
| 1 | [`01_mutable_catalog.ipynb`](01_mutable_catalog.ipynb) | `append` / `update` / `delete` / `optimize` with `MutationStats`, then reopen to confirm durability | Amazon product catalog |
