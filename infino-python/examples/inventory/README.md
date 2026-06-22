# Inventory

Keep a live product inventory current over a **single**
[Infino](https://pypi.org/project/infino/) table. Infino tables are
**mutable** — `append`, `update`, and `delete` rows and the full-text index and
SQL views stay correct, durably, with no rebuild:

- **`update`** — change a row in place (a price markdown); SQL and search reflect
  it immediately.
- **`delete`** — remove rows (discontinued items); they leave SQL and full-text
  search at once.
- **`optimize`** — compact the storage into fewer, fuller files, without
  changing query results.

`MutationStats` reports how many rows matched and changed. Every change is
committed to disk (object storage in production), so it survives a reconnect —
one engine, no separate store to update and keep in sync.

> Setup and the shared helpers are covered in the
> [examples README](../README.md). Run that setup once, then open the notebook.

## The life cycle in one run

```
updated 1 row(s):  price $15.99 -> $8.00      # update, reflected in SQL at once
clearance: delete price < $6.99
  matched 114, removed 114                     # rows leave search and SQL
  count 1200 -> 1086   cleared item searchable? False
after optimize: 1086 products (unchanged)      # compaction keeps results identical
reopened inventory: 1086 products, item still $8.00   # every change persisted
```

## Example

| # | Example | What it shows | Dataset |
| - | ------- | ------------- | ------- |
| 1 | [`01_live_inventory.ipynb`](01_live_inventory.ipynb) | `append` / `update` / `delete` / `optimize` with `MutationStats`, then reopen to confirm durability | Amazon product catalog |
