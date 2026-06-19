# locomo-recall — reproduce & debug infino retrieval on LOCOMO

A self-contained way to see *why* infino did or didn't surface the right memory
for a question on the public [LOCOMO](https://github.com/snap-research/locomo)
long-term-memory benchmark — and to check, after an engine change, whether a
specific miss is fixed.

The trick: **freeze the embedder out.** A one-time step embeds the corpus and the
questions and writes the vectors to `fixture.json`. From then on the Rust program
ingests those *frozen* vectors and runs only the infino engine. So between two
runs the only thing that changes is the engine code — any movement in recall is
attributable to your change, with no embedding model, API key, or network in the
loop.

> **Determinism caveat.** Vector and keyword modes are bit-for-bit stable across
> runs on the same fixture. Hybrid is *not* quite: RRF fusion produces many tied
> scores and the tie-break ordering isn't stable, so hybrid recall@10 jitters by
> up to ~1 point run-to-run on identical input. Treat sub-1pt hybrid moves as
> noise, not signal, and size any CI floor as a tolerance band below the baseline
> (see [The debug loop](#the-debug-loop)).

## The two parts

| File | Role | Needs |
|------|------|-------|
| `embed.mjs` | one-time fixture generator: fetch LOCOMO, embed corpus + questions, write `fixture.json` | Node 22+, an embeddings API key, network |
| `main.rs` | the repro/reporter: ingest the frozen vectors, run vector/keyword/hybrid, score recall@k, drill into misses | nothing — pure infino core |

`fixture.json` is committed, so day to day you only ever run `main.rs`.

## Run the repro

```sh
cargo run --example locomo-recall                  # full report: recall@k per mode + every hybrid miss
cargo run --example locomo-recall -- --id=D6:3     # focus the case(s) whose evidence is D6:3
cargo run --example locomo-recall -- --case=42     # focus one question by index
cargo run --example locomo-recall -- --fail-under=0.68   # exit non-zero if hybrid recall@10 drops below a floor
```

Flags: `--k` (top-k, default 10) · `--fixture=<path>` · `--id=<dia_id>` ·
`--case=<n>` · `--fail-under=<recall@10>`.

### Reading a miss

A LOCOMO id like `D6:3` is the dataset's own `dia_id` — **session 6, turn 3**. We
use it directly as the memory id, and the fixture carries each id's text, so the
report resolves every id to its source turn:

```
[#7] (single-hop) When did Caroline start her new job?
  expected evidence:
    D6:3  [NOT in hybrid top-k · vector #14 · keyword #31]
        (2023-05-12) [session 6, Caroline & Melanie] Caroline (D6:3): Just started at the clinic…
  hybrid top-10:
    #1  D6:9   (2023-05-12) … Caroline (D6:9): the commute is 40 min …
    #2  D3:2   …
    …
```

`vector #14 · keyword #31` is the actual debugging signal: the right turn was
*near-ish* in vector space but buried — a ranking/fusion problem — not far away,
which would point at the embedding instead.

## The debug loop

1. Run the repro; pick a miss.
2. Change the engine — BM25 scoring (`src/superfile/fts/`), the vector codec
   (`src/superfile/vector/`), or the `hybrid_search` fusion
   (`src/supertable/query/exec/hybrid_exec.rs`).
3. `cargo run --example locomo-recall -- --id=<that id>` — same frozen vectors,
   new engine code. Did it move into the top-k? Did recall@k go up? `--id` exits
   non-zero while the evidence is still absent, so it doubles as a one-line
   regression check once a real bug is fixed.

### Baseline & the CI floor

Current baseline on the committed fixture (conv-26, all 197 scored questions,
`text-embedding-3-small` 1536d): hybrid **recall@10 ≈ 0.72**, vector **0.71**,
keyword **0.51**.

Because hybrid jitters ~0.4pt run-to-run (see the determinism caveat), the CI
tripwire (the `LOCOMO recall test` job in `.github/workflows/ci.yml`) sizes its floor as a **tolerance
band below baseline**, not an exact value — currently `--fail-under=0.68`, a few
points under 0.72 and well above the jitter. That catches a real ranking/fusion
regression while absorbing the tie-break noise. Tighten it once hybrid
tie-breaking is made deterministic (a secondary sort on `_id` in the fusion step).

## Regenerate the fixture (rare)

Only when you want a different slice or embedder:

```sh
EMBED_BASE_URL=https://<endpoint>/v1 EMBED_API_KEY=<key> \
  node examples/locomo-recall/embed.mjs --out=examples/locomo-recall/fixture.json
```

By default it embeds the whole conversation (all ~199 conv-26 questions); pass
`--questions=<n>` to cap it for a quick test fixture.

Defaults to `text-embedding-3-small` (1536d); override with `EMBED_MODEL` /
`EMBED_DIM`. The fixture is plain JSON and runs ~15 MB at 1536d — readable in a
`git diff` but chunky; compressing it with the `zstd` dep the crate already
carries is a sensible follow-up before committing.
