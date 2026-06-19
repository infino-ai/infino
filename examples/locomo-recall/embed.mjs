// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//
// Fixture generator for the LOCOMO recall repro (examples/locomo-recall).
//
// This is the ONE step that needs an embedder + network. It fetches the public
// LOCOMO dataset, builds one memory per conversation turn, embeds the memories
// AND the questions with an OpenAI-compatible embeddings API
// (text-embedding-3-small, 1536d), and writes a single self-contained
// `fixture.json`:
//
//   { dim, embedder, conversation, corpus: [{id,text,vector}], cases: [{question,category,queryVector,expected}] }
//
// Once that fixture is committed, NOBODY needs this script (or a key, or the
// network) to run the repro — `main.rs` reads the frozen vectors and exercises
// only the infino engine. Regenerate the fixture only when you want a different
// slice or a different embedder.
//
//   EMBED_BASE_URL=https://<endpoint>/v1 EMBED_API_KEY=<key> \
//     node examples/locomo-recall/embed.mjs --conversations=1 \  # all of conv-26 (default)
//       --out=examples/locomo-recall/fixture.json
//
// Requires Node 22+ (global fetch). No npm dependencies.
import { writeFileSync, readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).map((a) => {
  const [k, v] = a.replace(/^--/, "").split("=");
  return [k, v ?? true];
}));
const LIMIT_CONV = args.conversations ? Number(args.conversations) : 1;
// Default to the whole conversation; a number caps it for a quick test fixture.
const LIMIT_Q = args.questions && args.questions !== "all" ? Number(args.questions) : Infinity;
const OUT = String(args.out || join(HERE, "fixture.json"));
const SOURCE = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json";

// --- embedder: OpenAI-compatible, text-embedding-3-small (1536d) -------------
// Sends BOTH `Authorization: Bearer` and `api-key` so the same script works
// against OpenAI and Azure OpenAI (Azure authenticates with `api-key`).
const BASE = (process.env.EMBED_BASE_URL || process.env.OPENAI_BASE_URL || "https://api.openai.com/v1").replace(/\/+$/, "");
const KEY = process.env.EMBED_API_KEY || process.env.OPENAI_API_KEY || "";
const EMBED_MODEL = process.env.EMBED_MODEL || "text-embedding-3-small";
const EMBED_DIM = Number(process.env.EMBED_DIM || 1536);
const BATCH = 96;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function embedBatch(texts, attempt = 0) {
  const res = await fetch(`${BASE}/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(KEY ? { Authorization: `Bearer ${KEY}`, "api-key": KEY } : {}) },
    body: JSON.stringify({ model: EMBED_MODEL, input: texts }),
  });
  if (!res.ok) {
    const t = await res.text();
    if ((res.status === 429 || res.status >= 500) && attempt < 4) {
      await sleep(600 * 2 ** attempt);
      return embedBatch(texts, attempt + 1);
    }
    throw new Error(`embeddings ${res.status}: ${t.slice(0, 160)}`);
  }
  const json = await res.json();
  return json.data.slice().sort((a, b) => a.index - b.index).map((d) => d.embedding);
}

async function embedMany(texts) {
  const out = [];
  for (let i = 0; i < texts.length; i += BATCH) out.push(...(await embedBatch(texts.slice(i, i + BATCH))));
  return out;
}

// --- dataset shaping (kept identical to how the memories are built elsewhere
// so the local corpus matches) ----------------------------------------------
const CAT = { 1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial" };

function sessionsOf(conv) {
  const out = [];
  for (let i = 1; ; i++) {
    const turns = conv[`session_${i}`];
    if (!Array.isArray(turns)) break;
    out.push({ date: conv[`session_${i}_date_time`] || "", turns });
  }
  return out;
}

// One memory per turn. The id IS LOCOMO's own dia_id ("D6:3" = session 6, turn 3),
// so a question's `evidence` dia_ids ARE the memory ids — no separate mapping.
function memoriesOf(conv) {
  const mems = [];
  let idx = 0;
  const a = String(conv.speaker_a ?? "").trim();
  const b = String(conv.speaker_b ?? "").trim();
  const participants = [a, b].filter(Boolean).join(" & ");
  sessionsOf(conv).forEach((s, si) => {
    for (const turn of s.turns) {
      const text = String(turn.text ?? turn.clean_text ?? "").trim();
      const caption = String(turn.blip_caption ?? "").trim();
      const body = [text, caption && `[shared image: ${caption}]`].filter(Boolean).join(" ");
      if (!body) continue;
      const diaId = turn.dia_id ? String(turn.dia_id).trim() : "";
      const id = diaId || `t${idx}`;
      idx++;
      const meta = `session ${si + 1}${participants ? `, ${participants}` : ""}`;
      mems.push({ id, text: `(${s.date}) [${meta}] ${turn.speaker}${diaId ? ` (${diaId})` : ""}: ${body}` });
    }
  });
  return mems;
}

function qaOf(item) {
  return (item.qa ?? [])
    .filter((q) => String(q.answer ?? q.adversarial_answer ?? "").trim())
    .map((q) => ({
      question: q.question,
      category: CAT[q.category] ?? `cat-${q.category ?? "?"}`,
      // the gold answer — carried so the repro can show query → answer → evidence
      gold: String(q.answer ?? q.adversarial_answer ?? ""),
      // LOCOMO sometimes packs several dia_ids in one evidence string ("D8:6; D9:17").
      evidence: (Array.isArray(q.evidence) ? q.evidence : []).flatMap((e) => String(e).match(/D\d+:\d+/g) || []),
    }));
}

async function fetchJson(url, attempt = 0) {
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(30_000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (e) {
    if (attempt < 3) {
      await sleep(1000 * 2 ** attempt);
      return fetchJson(url, attempt + 1);
    }
    throw new Error(`failed to load LOCOMO: ${e.message}`);
  }
}

(async () => {
  if (!KEY) {
    console.error("no embedder key — set EMBED_API_KEY (or OPENAI_API_KEY). This step needs an embedder; the repro itself does not.");
    process.exit(1);
  }
  const data = args.input && existsSync(String(args.input))
    ? JSON.parse(readFileSync(String(args.input), "utf8"))
    : await fetchJson(SOURCE);
  if (!Array.isArray(data) || !data[0]?.conversation) throw new Error("unexpected dataset shape");

  // The fixture format and the repro assume ONE corpus, and dia_ids ("D1:3") are
  // only unique within a conversation — combining several would collide ids. So
  // we embed a single conversation; reject >1 rather than silently mis-build.
  if (LIMIT_CONV !== 1) throw new Error(`--conversations must be 1 (the fixture is single-corpus); got ${LIMIT_CONV}`);
  const item = data[0];
  const conv = item.conversation;
  const mems = memoriesOf(conv);
  const memIds = new Set(mems.map((m) => m.id));
  const qas = qaOf(item).slice(0, LIMIT_Q);

  process.stderr.write(`embedding ${mems.length} memories + ${qas.length} questions (${EMBED_MODEL}, ${EMBED_DIM}d) …\n`);
  const memVecs = await embedMany(mems.map((m) => m.text));
  const qVecs = await embedMany(qas.map((q) => q.question));

  const fixture = {
    dim: EMBED_DIM,
    embedder: `${EMBED_MODEL} (${EMBED_DIM}d)`,
    conversation: item.sample_id ?? "conv",
    corpus: mems.map((m, i) => ({ id: m.id, text: m.text, vector: memVecs[i] })),
    cases: qas.map((q, i) => ({
      question: q.question,
      category: q.category,
      gold: q.gold,
      queryVector: qVecs[i],
      // keep only evidence ids that point at a real memory (some point at skipped/empty turns)
      expected: q.evidence.filter((d) => memIds.has(d)),
    })),
  };
  writeFileSync(OUT, JSON.stringify(fixture));
  process.stderr.write(`wrote ${OUT} — ${fixture.corpus.length} memories, ${fixture.cases.length} cases\n`);
})();
