// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//
// Agent memory with infino — give an agent durable long-term memory it can
// store to and recall from. One engine does keyword (BM25) + vector recall and
// SQL over the same data, so there is no separate vector store, keyword index,
// or metadata database to stitch together.
//
//   npm install      # in this folder — pulls a small local embedding model
//   node index.mjs   # after building the infino addon (see the repo README)
//
// TypeScript usage is identical — same imports, fully typed.

import { connect, IndexSpec } from "../../infino/index.js";
import { pipeline } from "@huggingface/transformers";

// --- a local embedding model (no API key, runs on CPU) --------------------
// infino is bring-your-own-vectors: it stores and searches whatever vectors you
// hand it. Swap this for any model or embeddings API — just keep DIM in sync.
const DIM = 384;
const extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
const embed = async (text) =>
  Array.from((await extractor(text, { pooling: "mean", normalize: true })).data);

// --- open a catalog (in-memory here; use "./data" or "s3://…" to persist) -
const db = connect("memory://");

// One table is the whole memory store: text (full-text indexed) + embedding
// (vector indexed) + structured fields, all queryable together.
const mem = db.createTable(
  "memories",
  { id: "large_utf8", text: "large_utf8", vector: { vector: DIM }, importance: "float64", created_at: "float64" },
  new IndexSpec().fts("text").vector("vector", DIM, 1, "cosine"),
);

// --- the agent remembers a few things about the user ----------------------
const DAY = 86400000;
const t0 = Date.parse("2026-06-01T00:00:00Z");
const facts = [
  { text: "The user's name is Ada; she prefers to be called Ade.", importance: 0.9, day: 0 },
  { text: "Ada is severely allergic to peanuts.", importance: 1.0, day: 2 },
  { text: "Ada is planning a holiday to Japan this October.", importance: 0.7, day: 5 },
  { text: "Ada writes Postgres at work and is learning Rust on the side.", importance: 0.6, day: 9 },
  { text: "Ada has a cat named Biscuit.", importance: 0.5, day: 12 },
];
const rows = [];
for (let i = 0; i < facts.length; i++) {
  rows.push({
    id: `m${i}`,
    text: facts[i].text,
    vector: await embed(facts[i].text),
    importance: facts[i].importance,
    created_at: t0 + facts[i].day * DAY,
  });
}
mem.append(rows); // one append == one commit; durable on return
console.log(`stored ${rows.length} memories\n`);

// --- 1. semantic recall — meaning, not exact words ------------------------
const q1 = "what foods are dangerous for the user?";
console.log(`semantic  "${q1}"`);
for (const r of mem.vectorSearch("vector", await embed(q1), 2, { projection: ["text"] }))
  console.log("   ·", r.text); // → the peanut allergy, with no shared keywords

// --- 2. keyword recall — exact terms (names, IDs, rare words) -------------
console.log(`\nkeyword   "Biscuit"`);
for (const r of mem.bm25Search("text", "Biscuit", 2, { projection: ["text"] }))
  console.log("   ·", r.text);

// --- 3. hybrid recall — BM25 + vector fused with RRF, in one engine -------
// Catches lexical anchors (names, dates) AND paraphrased meaning at once.
async function recallHybrid(query, k = 3) {
  const n = k * 4;
  const qv = await embed(query);
  const kw = mem.bm25Search("text", query, n, { projection: ["id", "text"] });
  const vec = mem.vectorSearch("vector", qv, n, { projection: ["id", "text"] });
  const score = new Map();
  const byId = new Map();
  for (const list of [kw, vec]) {
    list.forEach((r, rank) => {
      score.set(r.id, (score.get(r.id) ?? 0) + 1 / (60 + rank)); // reciprocal rank fusion
      byId.set(r.id, r);
    });
  }
  return [...score.entries()].sort((a, b) => b[1] - a[1]).slice(0, k).map(([id]) => byId.get(id));
}
const q3 = "any travel coming up for Ada?";
console.log(`\nhybrid    "${q3}"`);
for (const r of await recallHybrid(q3, 2)) console.log("   ·", r.text);

// --- 4. SQL over the same memories — structured / temporal recall ---------
// The view a plain vector store can't give you: query memory by its fields.
console.log("\nSQL  most important memories first:");
for (const r of db.querySql("SELECT text, importance FROM memories ORDER BY importance DESC LIMIT 3"))
  console.log(`   · [${r.importance}] ${r.text}`);

console.log("\n✓ agent-memory demo complete");
