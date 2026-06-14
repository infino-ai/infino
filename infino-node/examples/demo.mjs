// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//
// A walkthrough of how someone uses the infino Node binding, end to end.
// Run after building the addon:  node examples/demo.mjs
//
// (TypeScript usage is identical — same imports, fully typed.
// This file is .mjs so it runs with plain `node`, no build step.)

import { connect, IndexSpec } from "../infino/index.js";
import { Schema, Field, LargeUtf8, Float32, FixedSizeList } from "apache-arrow";

const dim = 16;
const onehot = (i) => { const v = Array(dim).fill(0); v[i] = 1; return v; };

// --- 1. open a catalog (in-memory here; use "./data" or "s3://…" for real) ---
const db = connect("memory://");
console.log("connected:", db.listTables(), "tables");

// --- 2. full-text table — append plain objects (the wrapper builds Arrow) ---
const docs = db.createTable(
  "docs",
  new Schema([new Field("title", new LargeUtf8(), false)]),
  new IndexSpec().fts("title"),
);
docs.append([
  { title: "the quick brown fox" },
  { title: "a lazy dog" },
  { title: "the fox and the hound" },
]);
console.log("\nappended 3 docs; tables now:", db.listTables());

// --- 3. ranked BM25 search — matching rows come back as plain records ---
console.log("\nbm25Search('fox') rows:");
for (const row of docs.bm25Search("title", "fox", 10)) console.log("  ", row);

// --- 3b. unranked token match — lightweight _id (bigint) + score ---
console.log("\ntokenMatch('fox'):");
for (const r of docs.tokenMatch("title", "fox", { projection: ["_id", "score"] }))
  console.log(`  _id=${r._id}  score=${r.score}`);

// --- 4. SQL across the catalog — records by default ---
console.log("\nquerySql rows:");
for (const row of db.querySql("SELECT _id, title FROM docs ORDER BY _id")) console.log("  ", row);

// --- 5. vector kNN — append objects, query with a plain number[] ---
const vecs = db.createTable(
  "vecs",
  new Schema([new Field("emb", new FixedSizeList(dim, new Field("item", new Float32(), true)), false)]),
  new IndexSpec().vector("emb", dim, 1, "cosine"),
);
vecs.append([{ emb: onehot(0) }, { emb: onehot(1) }, { emb: onehot(2) }]);
console.log("\nvectorSearch(onehot(0)) rows:");
for (const row of vecs.vectorSearch("emb", onehot(0), 3)) console.log("  ", row);

console.log("\n✓ demo complete");
