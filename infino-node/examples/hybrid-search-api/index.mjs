// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//
// Embedded hybrid search API — infino running *inside* your app process, with no
// separate search server to deploy. Indexes a real product catalog (a sample of
// Amazon product metadata, pulled key-free from the HuggingFace Hub) with BM25 +
// vector, and serves production-style search over HTTP:
//
//   GET /search?q=<query>&k=<n>   ->  { query, results: [{ title, category, price, rating, score }] }
//
// Ranking is infino's native single-pass `hybrid_search` (keywords + meaning,
// fused in the engine). Build the addon first, then:
//
//   npm install
//   node index.mjs          # downloads a live catalog sample + model, serves on :3000
//   SMOKE=1 node index.mjs  # self-check against a bundled catalog, then exit (the CI gate)
//
// Interactive runs pull a live catalog from the HuggingFace Hub and fall back to a
// small bundled sample if the Hub is unreachable; the SMOKE gate always indexes the
// bundled sample so it stays deterministic and never depends on a third-party
// service being up.
//
// (TypeScript usage is identical — same imports, fully typed.)

import { createServer } from "node:http";
import assert from "node:assert/strict";
import { rmSync } from "node:fs";
import { connect, IndexSpec } from "infino";
import { pipeline } from "@huggingface/transformers";
import { SAMPLE_CATALOG } from "./sample-catalog.mjs";

// --- a tiny local embedder: no API key, downloads once then cached -----------
const DIM = 384;
const extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
const embed = async (text) => {
  const out = await extractor(text, { pooling: "mean", normalize: true });
  return Array.from(out.data, Number);
};

// --- load a real catalog sample from the HuggingFace Hub (key-free) ----------
// The dataset is grouped by category, so we spread reads across it for variety.
// Same source the Python examples use: smartcat/Amazon_Sample_Metadata_2023.
const DATASET = "smartcat/Amazon_Sample_Metadata_2023";
const ROWS_URL = `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(DATASET)}&config=default&split=train`;
const CATALOG_N = Number(process.env.CATALOG_N ?? 200);

async function fetchJson(url, attempt = 0) {
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(20_000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (e) {
    if (attempt < 3) {
      await new Promise((r) => setTimeout(r, 1000 * 2 ** attempt));
      return fetchJson(url, attempt + 1);
    }
    throw new Error(`failed to load catalog from the Hub: ${e.message}`);
  }
}

function toProduct(row) {
  const price = Number(row.price);
  const title = String(row.title ?? "").trim();
  if (!title || !Number.isFinite(price) || price <= 0) return null;
  const desc = Array.isArray(row.description) ? row.description.join(" ") : String(row.description ?? "");
  return {
    title,
    text: `${title}. ${desc.slice(0, 400)}`, // indexed for search (title + description)
    price,
    rating: Number(row.average_rating) || 0,
    category: String(row.main_category ?? "Unknown"),
  };
}

async function loadCatalog(n) {
  const first = await fetchJson(`${ROWS_URL}&offset=0&length=100`);
  const total = first.num_rows_total ?? 100_000;
  const pages = 8; // spread reads across the (category-grouped) dataset for variety
  const out = [];
  const take = (rows) => {
    for (const r of rows ?? []) {
      const p = toProduct(r.row);
      if (p) out.push(p);
      if (out.length >= n) return;
    }
  };
  take(first.rows);
  for (let i = 1; i < pages && out.length < n; i++) {
    const offset = Math.min(total - 100, Math.floor((total / pages) * i));
    take((await fetchJson(`${ROWS_URL}&offset=${offset}&length=100`)).rows);
  }
  return out;
}

// The offline catalog: shipped with the example so a run never hard-depends on
// the Hub being reachable (see sample-catalog.mjs).
function loadBundledSample(n) {
  return SAMPLE_CATALOG.slice(0, n);
}

// --- index the catalog once at startup: one table, FTS + vector --------------
const DIR = "./catalog-store";
rmSync(DIR, { recursive: true, force: true });
const db = connect(DIR);
const catalog = db.createTable(
  "catalog",
  { id: "large_utf8", title: "large_utf8", text: "large_utf8", category: "large_utf8", price: "float64", rating: "float64", vector: { vector: DIM } },
  new IndexSpec().fts("text").vector("vector", DIM, 1, "cosine"),
);

// Interactive runs index a live sample from the Hub, falling back to the bundled
// catalog if it's unreachable. The SMOKE self-check always indexes the bundled
// catalog so the end-to-end gate is deterministic and independent of Hub uptime.
let products;
if (process.env.SMOKE) {
  products = loadBundledSample(CATALOG_N);
  process.stderr.write(`indexing ${products.length} bundled products (SMOKE self-check)\n`);
} else {
  process.stderr.write(`loading ${CATALOG_N} products from ${DATASET} …\n`);
  try {
    products = await loadCatalog(CATALOG_N);
  } catch (e) {
    products = loadBundledSample(CATALOG_N);
    process.stderr.write(`${e.message}; falling back to ${products.length} bundled products\n`);
  }
}
const rows = [];
let i = 0;
for (const p of products) rows.push({ id: `p${i++}`, title: p.title, text: p.text, category: p.category, price: p.price, rating: p.rating, vector: await embed(p.text) });
catalog.append(rows);
process.stderr.write(`indexed ${rows.length} products\n`);

// --- search: native single-pass hybrid (BM25 + vector, fused in the engine) --
const sqlStr = (s) => s.replace(/'/g, "''");
async function search(q, k = 5) {
  const qvec = (await embed(q)).join(",");
  const hits = db.querySql(
    `SELECT title, category, price, rating, score FROM hybrid_search(` +
      `'catalog', 'text', '${sqlStr(q)}', 'vector', '${qvec}', ${k})`,
  );
  return hits.map((r) => ({ title: r.title, category: r.category, price: Number(r.price), rating: Number(r.rating), score: r.score }));
}

// --- the HTTP endpoint (node:http — no web framework needed) -----------------
const server = createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  if (url.pathname !== "/search") {
    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "GET /search?q=…&k=…" }));
    return;
  }
  const q = url.searchParams.get("q") ?? "";
  // Clamp the user-controlled k to a sane positive integer before it reaches SQL
  // (rejects NaN / Infinity / negatives / absurd values).
  const k = Math.min(50, Math.max(1, Math.trunc(Number(url.searchParams.get("k"))) || 5));
  try {
    const results = await search(q, k);
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ query: q, results }, null, 2));
  } catch (e) {
    res.writeHead(500, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: e.message }));
  }
});

// Interactive runs use PORT (default 3000); the SMOKE self-check binds an
// ephemeral port (0) so CI never collides with whatever's already on 3000.
const PORT = Number(process.env.PORT ?? (process.env.SMOKE ? 0 : 3000));
server.listen(PORT, async () => {
  try {
    const address = server.address();
    if (!address || typeof address === "string") throw new Error("server did not bind to a TCP port");
    const port = address.port;
    // Self-check on startup, so `node index.mjs` doubles as an end-to-end smoke
    // test. Results depend on a live dataset sample, so we assert SHAPE (the
    // endpoint works and hybrid ranking returns well-formed rows), not an exact hit.
    const res = await fetch(`http://localhost:${port}/search?q=${encodeURIComponent("a thoughtful birthday gift")}&k=5`);
    const body = await res.json();
    assert.equal(res.status, 200, "search endpoint should return 200");
    assert.ok(body.results.length > 0, "search should return results");
    assert.equal(typeof body.results[0].title, "string");
    assert.ok(body.results[0].title.length > 0, "result should have a title");
    assert.ok(Number.isFinite(body.results[0].price), "result should have a numeric price");

    console.log(`✓ hybrid search API listening on http://localhost:${port} (${rows.length} products)`);
    console.log(`  try: curl 'http://localhost:${port}/search?q=gift+for+someone+who+loves+cooking'`);
    console.log(`       curl 'http://localhost:${port}/search?q=something+to+keep+skin+moisturized'`);

    // CI runs with SMOKE=1: the self-check passed, so exit cleanly.
    if (process.env.SMOKE) {
      console.log("✓ self-check passed");
      server.close();
    }
  } catch (e) {
    // Close the listener so a failed self-check terminates instead of hanging.
    server.close();
    throw e;
  }
});
