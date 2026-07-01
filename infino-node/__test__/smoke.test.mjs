// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//
// End-to-end smoke tests for the infino Node.js bindings (friendly API).
// Run after `npm run build`:
//
//     cd infino-node
//     npm install && npm run build && npm test
//
// Mirrors infino-python/tests/test_smoke.py. The wrapper takes arrays of
// objects in and returns plain records out — Arrow only appears when
// building the create-table schema.

import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { connect, IndexSpec, BUILDER_ID } from "../infino/index.js";
import { Schema, Field, LargeUtf8, Float32, FixedSizeList, Table, vectorFromArray } from "apache-arrow";

// FTS columns must be LargeUtf8; non-null to match the appended data.
const titleSchema = () => new Schema([new Field("title", new LargeUtf8(), false)]);

const embSchema = (dim) =>
  new Schema([new Field("emb", new FixedSizeList(dim, new Field("item", new Float32(), true)), false)]);

const onehot = (i, dim) => {
  const v = new Array(dim).fill(0);
  v[i] = 1.0;
  return v;
};

test("memory roundtrip: create, append, search, drop", () => {
  const db = connect("memory://");
  const docs = db.createTable("docs", titleSchema(), new IndexSpec().fts("title"));

  // append takes plain objects (also an apache-arrow Table / RecordBatch).
  docs.append([{ title: "the quick brown fox" }, { title: "a lazy dog" }]);

  assert.deepEqual(db.listTables(), ["docs"]);

  const reopened = db.openTable("docs");

  // bm25Search returns matching rows as plain records; `_id` is a bigint.
  const ranked = reopened.bm25Search("title", "fox", 10);
  assert.equal(ranked.length, 1);
  assert.equal(typeof ranked[0]._id, "bigint");

  // tokenMatch returns matching rows (unranked, score 0); `_id` is a bigint.
  const hits = reopened.tokenMatch("title", "fox");
  assert.equal(hits.length, 1);
  assert.equal(typeof hits[0]._id, "bigint");
  assert.equal(hits[0].score, 0);

  db.dropTable("docs");
  assert.deepEqual(db.listTables(), []);
});

test("append accepts an apache-arrow RecordBatch", () => {
  const db = connect("memory://");
  const docs = db.createTable("docs", titleSchema(), new IndexSpec().fts("title"));
  // A RecordBatch (or a Table) can be appended directly, not just objects.
  const batch = new Table({ title: vectorFromArray(["the quick brown fox"], new LargeUtf8()) }).batches[0];
  docs.append(batch);
  assert.equal(docs.tokenMatch("title", "fox").length, 1);
});

test("createTable accepts a plain { column: type } descriptor", () => {
  const db = connect("memory://");
  // No apache-arrow needed to define the schema.
  const docs = db.createTable("docs", { title: "large_utf8" }, new IndexSpec().fts("title"));
  docs.append([{ title: "the quick brown fox" }]);
  assert.equal(docs.tokenMatch("title", "fox").length, 1);
});

test("querySql returns records", () => {
  const db = connect("memory://");
  const docs = db.createTable("docs", titleSchema(), new IndexSpec().fts("title"));
  docs.append([{ title: "alpha" }, { title: "beta" }, { title: "gamma" }]);

  const rows = db.querySql("SELECT COUNT(*) AS n FROM docs");
  assert.equal(Number(rows[0].n), 3);
});

test("querySql can return an Arrow table with { arrow: true }", () => {
  const db = connect("memory://");
  const docs = db.createTable("docs", titleSchema(), new IndexSpec().fts("title"));
  docs.append([{ title: "the quick brown fox" }, { title: "a lazy dog" }]);

  const tbl = db.querySql("SELECT _id, score FROM bm25_search('docs', 'title', 'fox', 10)", { arrow: true });
  assert.equal(tbl.numRows, 1);
});

test("query_sql with a hybrid_search TVF returns rows from a sync host (#170)", () => {
  const db = connect("memory://");
  const dim = 16;
  const schema = new Schema([
    new Field("title", new LargeUtf8(), false),
    new Field("emb", new FixedSizeList(dim, new Field("item", new Float32(), true)), false),
  ]);
  const docs = db.createTable("docs", schema, new IndexSpec().fts("title").vector("emb", dim, 1, "cosine"));
  docs.append([
    { title: "billing and refunds", emb: onehot(0, dim) },
    { title: "dark mode appearance", emb: onehot(1, dim) },
  ]);
  // A search TVF through query_sql used to abort the process from a sync host
  // (no ambient multi-thread runtime); it must now return rows, not abort.
  const qvec = onehot(0, dim).join(",");
  const rows = db.querySql(
    `SELECT _id, score FROM hybrid_search('docs', 'title', 'billing', 'emb', '${qvec}', 5)`,
  );
  assert.ok(rows.length >= 1);
});

test("hybridSearch fuses BM25 and vector retrieval", () => {
  const db = connect("memory://");
  const dim = 16;
  const schema = new Schema([
    new Field("title", new LargeUtf8(), false),
    new Field("emb", new FixedSizeList(dim, new Field("item", new Float32(), true)), false),
  ]);
  const docs = db.createTable("docs", schema, new IndexSpec().fts("title").vector("emb", dim, 1, "cosine"));
  docs.append([
    { title: "rust async", emb: onehot(0, dim) },
    { title: "python data", emb: onehot(1, dim) },
    { title: "rust systems", emb: onehot(2, dim) },
  ]);

  const hits = docs.hybridSearch("title", "rust", "emb", onehot(0, dim), 10, { projection: ["_id", "score"] });
  assert.ok(hits.length >= 1);
  assert.equal(typeof hits[0]._id, "bigint");
  // RRF score is higher-is-better, so rows come back descending.
  const scores = hits.map((r) => r.score);
  assert.deepEqual(scores, [...scores].sort((a, b) => b - a));

  // The SQL TVF fixes mode="or" and default nprobe, so the direct call matches.
  const qvec = onehot(0, dim).join(",");
  const viaSql = db.querySql(`SELECT _id FROM hybrid_search('docs', 'title', 'rust', 'emb', '${qvec}', 10)`);
  assert.equal(viaSql.length, hits.length);
});

test("tokenMatch and exactMatch return unranked rows", () => {
  const db = connect("memory://");
  const docs = db.createTable("docs", titleSchema(), new IndexSpec().fts("title"));
  docs.append([{ title: "the quick brown fox" }, { title: "a lazy dog" }]);

  // Project to just _id + score (score is 0 for unranked matches).
  const tok = docs.tokenMatch("title", "fox", { projection: ["_id", "score"] });
  assert.equal(tok.length, 1);
  assert.equal(typeof tok[0]._id, "bigint");
  assert.equal(tok[0].score, 0);

  const ex = docs.exactMatch("title", "a lazy dog");
  assert.equal(ex.length, 1);

  // count matches the same predicate without materializing rows.
  assert.equal(docs.count("title", "fox"), 1);
  assert.equal(docs.count("title", "fox dog"), 2); // "or" default: fox OR dog
  assert.equal(docs.count("title", "fox dog", { mode: "and" }), 0);
});

test("BUILDER_ID is a non-empty string", () => {
  assert.equal(typeof BUILDER_ID, "string");
  assert.ok(BUILDER_ID.length > 0);
});

test("unknown table throws", () => {
  const db = connect("memory://");
  assert.throws(() => db.openTable("nope"));
});

test("localfs persists across reconnect", () => {
  const dir = mkdtempSync(join(tmpdir(), "infino-node-smoke-"));
  const db = connect(dir);
  const docs = db.createTable("docs", titleSchema(), new IndexSpec().fts("title"));
  docs.append([{ title: "a lazy sleeping fox" }]);

  const db2 = connect(dir);
  assert.deepEqual(db2.listTables(), ["docs"]);
  assert.equal(db2.openTable("docs").tokenMatch("title", "fox").length, 1);
});

test("connect accepts storageOptions", () => {
  // storageOptions is a no-op for local storage but must parse and apply.
  const dir = mkdtempSync(join(tmpdir(), "infino-node-opts-"));
  const db = connect(dir, { storageOptions: { aws_region: "us-east-1" } });
  const docs = db.createTable("docs", titleSchema(), new IndexSpec().fts("title"));
  docs.append([{ title: "the quick brown fox" }]);
  assert.equal(docs.tokenMatch("title", "fox").length, 1);
});

test("connect rejects an unknown storageOptions key", () => {
  // An unknown key surfaces at connect time, not as a silent drop.
  assert.throws(() => connect("s3://bucket/prefix", { storageOptions: { not_a_real_key: "x" } }));
});

test("connect does not probe by default", () => {
  // Default (validate off): a bogus bucket builds the handle without
  // touching the backend.
  connect("s3://no-such-bucket-xyzzy/prefix");
});

test("vector search end-to-end", () => {
  const db = connect("memory://");
  const dim = 16; // infino requires vector dim in [16, 4096]
  const docs = db.createTable("vecs", embSchema(dim), new IndexSpec().vector("emb", dim, 1, "cosine"));

  // append objects with the vector as a plain number[] — the wrapper
  // builds the FixedSizeList<float32> column from the declared schema.
  docs.append([{ emb: onehot(0, dim) }, { emb: onehot(1, dim) }, { emb: onehot(2, dim) }]);

  // query vector as a plain array (wrapper coerces to Float32Array).
  const rows = docs.vectorSearch("emb", onehot(0, dim), 10);
  assert.ok(rows.length >= 1);

  // nprobe + rerankMult tuning knobs are accepted.
  const tuned = docs.vectorSearch("emb", onehot(0, dim), 5, { nprobe: 1, rerankMult: 4 });
  assert.ok(tuned.length >= 1);
});

test("filtered vector search (pushdown text predicate)", () => {
  const db = connect("memory://");
  const dim = 16;
  // A table with both an FTS column (title) and a vector column (emb).
  const schema = new Schema([
    new Field("title", new LargeUtf8(), false),
    new Field("emb", new FixedSizeList(dim, new Field("item", new Float32(), true)), false),
  ]);
  const docs = db.createTable(
    "docs",
    schema,
    new IndexSpec().fts("title").vector("emb", dim, 1, "cosine"),
  );
  docs.append([
    { title: "billing and refunds", emb: onehot(0, dim) },
    { title: "refund policy", emb: onehot(0, dim) },
    { title: "dark mode appearance", emb: onehot(1, dim) },
  ]);

  // Unfiltered kNN over the topic-0 embedding sees both topic-0 rows.
  const all = docs.vectorSearch("emb", onehot(0, dim), 10);
  assert.ok(all.length >= 2);

  // Same kNN, restricted to rows whose `title` matches "billing" — a
  // pushdown pre-filter, so only the matching row comes back (not a
  // post-filter over the global top-k).
  const filtered = docs.vectorSearch("emb", onehot(0, dim), 10, {
    filter: { column: "title", query: "billing", mode: "or" },
    projection: ["_id", "title", "score"],
  });
  assert.equal(filtered.length, 1);
  assert.equal(filtered[0].title, "billing and refunds");
});

test("update and delete by SQL predicate (localfs)", () => {
  const dir = mkdtempSync(join(tmpdir(), "infino-node-mut-"));
  const db = connect(dir); // mutations require durable storage
  const docs = db.createTable("docs", { title: "large_utf8" }, new IndexSpec().fts("title"));
  docs.append([{ title: "alpha" }, { title: "bravo" }, { title: "charlie" }]);

  const del = docs.delete("title = 'bravo'");
  assert.equal(del.matched, 1);
  assert.equal(typeof del.nTombstoned, "number");
  assert.equal(docs.tokenMatch("title", "bravo").length, 0);

  const upd = docs.update("title = 'alpha'", [{ title: "alpha2" }]); // 1:1 replacement
  assert.equal(upd.matched, 1);
  assert.equal(docs.tokenMatch("title", "alpha").length, 0);
  assert.equal(docs.tokenMatch("title", "alpha2").length, 1);
});

test("optimize merges superfiles, data intact (localfs)", () => {
  const dir = mkdtempSync(join(tmpdir(), "infino-node-compact-"));
  const db = connect(dir);
  const docs = db.createTable("docs", { title: "large_utf8" }, new IndexSpec().fts("title"));
  // three appends => three superfiles
  docs.append([{ title: "alpha" }]);
  docs.append([{ title: "beta" }]);
  docs.append([{ title: "gamma" }]);

  docs.optimize({ targetSuperfileSizeMb: 256, minFillPercent: 50, maxMemoryMb: 512 });
  // data is intact after compaction
  assert.equal(docs.tokenMatch("title", "beta").length, 1);
  assert.equal(docs.tokenMatch("title", "alpha").length, 1);
  assert.equal(Number(db.querySql("SELECT COUNT(*) AS n FROM docs")[0].n), 3);

  docs.optimize(); // default settings also run cleanly

  // gc after compaction reclaims the now-orphaned pre-merge objects. Use a
  // 0s grace so freshly-orphaned objects are eligible immediately.
  const report = docs.gc(0);
  assert.equal(typeof report.bytesFreed, "number");
  assert.equal(typeof report.objectsDeleted, "number");
  assert.ok(report.objectsDeleted >= 0);
  // data still intact after the sweep
  assert.equal(Number(db.querySql("SELECT COUNT(*) AS n FROM docs")[0].n), 3);
});

test("gc requires durable storage (reject memory://)", () => {
  const db = connect("memory://");
  const docs = db.createTable("docs", { title: "large_utf8" }, new IndexSpec().fts("title"));
  docs.append([{ title: "alpha" }]);
  assert.throws(() => docs.gc(0));
});

test("update and delete require durable storage (reject memory://)", () => {
  const db = connect("memory://");
  const docs = db.createTable("docs", { title: "large_utf8" }, new IndexSpec().fts("title"));
  docs.append([{ title: "alpha" }]);
  assert.throws(() => docs.delete("title = 'alpha'"));
  assert.throws(() => docs.update("title = 'alpha'", [{ title: "beta" }]));
});

test("update enforces 1:1 cardinality", () => {
  const dir = mkdtempSync(join(tmpdir(), "infino-node-card-"));
  const db = connect(dir);
  const docs = db.createTable("docs", { title: "large_utf8" }, new IndexSpec().fts("title"));
  docs.append([{ title: "alpha" }, { title: "beta" }]);
  // predicate matches one row but two replacement rows are supplied
  assert.throws(() => docs.update("title = 'alpha'", [{ title: "x" }, { title: "y" }]));
});

test("connect parses cache + cold-fetch options", () => {
  const dir = mkdtempSync(join(tmpdir(), "infino-node-cache-"));
  const cacheDir = mkdtempSync(join(tmpdir(), "infino-node-cachedir-"));
  // options parse and apply without error; they're a no-op for local storage.
  const db = connect(dir, {
    cacheDir,
    cacheBudgetBytes: 64 * 1024 * 1024,
    coldFetchMode: "lazy_foreground_with_background_fill",
  });
  const docs = db.createTable("docs", { title: "large_utf8" }, new IndexSpec().fts("title"));
  docs.append([{ title: "the quick brown fox" }]);
  assert.equal(docs.tokenMatch("title", "fox").length, 1);
});

test("connect rejects an invalid coldFetchMode", () => {
  assert.throws(() => connect("memory://", { coldFetchMode: "nonsense" }));
});
