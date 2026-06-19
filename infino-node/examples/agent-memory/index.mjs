// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//
// Agent long-term memory with infino — a runnable walkthrough.
//
// An agent that talks with a user over weeks needs to remember the conversation
// and recall the right bits later. This example loads a real months-long, multi-
// session conversation (a public long-term-conversational-memory dataset), turns
// each message into a memory, and shows the three things infino does over the
// SAME table:
//   1. hybrid recall   — meaning + keywords, fused in one query
//   2. SQL over memory  — "who said how much?", filter by session (a vector store can't)
//   3. mutation         — forget part of the history
//
// Build the addon first, then:
//   npm install        # infino + a small local embedder
//   node index.mjs
//
// (TypeScript usage is identical — same imports, fully typed.)

import assert from "node:assert/strict";
import { rmSync } from "node:fs";
import { connect, IndexSpec } from "infino";
import { pipeline } from "@huggingface/transformers";

// --- a tiny local embedder: no API key, downloads once then cached -----------
const DIM = 384;
const extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
const embed = async (text) => {
  const out = await extractor(text, { pooling: "mean", normalize: true });
  return Array.from(out.data, Number);
};

// --- load one real conversation (key-free, from the public dataset) ----------
// LOCOMO is a public dataset of long, multi-session conversations — a natural
// stand-in for an agent's chat history with a user.
const SOURCE = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json";

async function fetchJson(url, attempt = 0) {
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(30_000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (e) {
    if (attempt < 3) {
      await new Promise((r) => setTimeout(r, 1000 * 2 ** attempt));
      return fetchJson(url, attempt + 1);
    }
    throw new Error(`failed to load the conversation: ${e.message}`);
  }
}

process.stderr.write("loading a conversation …\n");
const payload = await fetchJson(SOURCE);
if (!Array.isArray(payload) || !payload[0]?.conversation) {
  throw new Error("unexpected dataset shape — expected a non-empty array of conversations");
}
const conv = payload[0].conversation; // one multi-session conversation

// Each message becomes one memory, prefixed with its session timestamp (so
// "when did X happen?" is answerable) and tagged with the session + speaker.
const memories = [];
let id = 0;
for (let s = 1; Array.isArray(conv[`session_${s}`]); s++) {
  const date = String(conv[`session_${s}_date_time`] ?? "");
  for (const turn of conv[`session_${s}`]) {
    const text = String(turn.text ?? "").trim();
    const caption = String(turn.blip_caption ?? "").trim(); // a shared photo's content is text here
    const body = [text, caption && `[shared image: ${caption}]`].filter(Boolean).join(" ");
    if (!body) continue;
    memories.push({ id: `m${id++}`, text: `(${date}) ${turn.speaker}: ${body}`, speaker: String(turn.speaker), session: s, date });
  }
}

// --- the memory store: one table, full-text AND vector indexes ---------------
const DIR = "./agent-memory-store";
rmSync(DIR, { recursive: true, force: true });
const db = connect(DIR);
const mem = db.createTable(
  "memory",
  { id: "large_utf8", text: "large_utf8", speaker: "large_utf8", session: "float64", date: "large_utf8", vector: { vector: DIM } },
  new IndexSpec().fts("text").vector("vector", DIM, 1, "cosine"),
);

process.stderr.write(`indexing ${memories.length} memories …\n`);
const rows = [];
for (const m of memories) rows.push({ ...m, vector: await embed(m.text) });
mem.append(rows);

// --- recall(): native single-pass hybrid search (BM25 + vector, one query) ---
const sqlStr = (s) => s.replace(/'/g, "''");
async function recall(query, k = 3) {
  const qvec = (await embed(query)).join(",");
  return db.querySql(
    `SELECT text, speaker, score FROM hybrid_search('memory', 'text', '${sqlStr(query)}', 'vector', '${qvec}', ${k})`,
  );
}

console.log('recall: "what did they say about a support group?"');
const hits = await recall("what did they say about a support group?");
for (const r of hits) console.log("  •", r.text.slice(0, 100));

// --- SQL over memory: the structured view a plain vector store can't give ----
console.log("\nSQL: messages per speaker");
const bySpeaker = db.querySql("SELECT speaker, COUNT(*) AS n FROM memory GROUP BY speaker ORDER BY n DESC");
for (const r of bySpeaker) console.log(`  • ${r.speaker}: ${Number(r.n)}`);

// --- mutation: forget part of the history (e.g. the opening session) ---------
const total = memories.length;
const forgotten = mem.delete("session = 1");
console.log(`\nforgot session 1 — ${forgotten.nTombstoned} memories`);
const remaining = Number(db.querySql("SELECT COUNT(*) AS n FROM memory")[0].n);
console.log("remaining memories:", remaining);

// --- self-check: a regression (bad hybrid ranking, broken SQL/mutation) fails
// the process, so `node index.mjs` doubles as an end-to-end smoke test in CI. --
assert.ok(hits.some((r) => /support group/i.test(r.text)), "recall should surface the support-group message");
assert.ok(bySpeaker.length >= 2, "GROUP BY should return both speakers");
assert.equal(bySpeaker.reduce((a, r) => a + Number(r.n), 0), total, "speaker counts should sum to all memories");
assert.ok(forgotten.nTombstoned > 0 && remaining < total, "forgetting session 1 should remove memories");

console.log("\n✓ agent-memory example complete");
