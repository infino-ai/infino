#!/usr/bin/env python3
"""Summarize benchmark deltas for a PR comment.

Diffs this run against the latest `main` baseline, keeps only changes past the
noise threshold, and writes a lean markdown summary. Foundry narrates; the
percentages are computed here, never by the model. On any model failure or
missing creds the deterministic tables are the summary on their own. Stdlib
only — the runner needs no pip install.

Inputs (env):
  REPORTS                  space-separated report names (basenames, no .json)
  BASELINE_DIR             dir holding <report>.json from the main baseline
  CURRENT_DIR              dir holding <report>.json from this run
  BENCH_NOISE_THRESHOLD_PCT  threshold in percent (default 10)
  OUT_FILE                 markdown destination (default /tmp/ai-summary.md)
  BENCH_LABEL              human label for the run (the `bench` input)
  RUN_URL                  link to the full Actions run
  ERRORS                   newline-separated panic/error lines (may be empty)
  AZURE_AI_ENDPOINT        Foundry OpenAI-compatible base (…/openai/v1)
  AZURE_AI_API_KEY         Foundry key (Bearer)
  AZURE_AI_MODEL           model deployment (default gpt-5.4)
"""

import json
import os
import sys
import urllib.error
import urllib.request

# Report keys are "anchor|subtitle|label|header"; split on this into 4 fields.
KEY_PARTS = 4

# Mirrors the bench renderer's Better enum: these headers are higher-is-better,
# everything else comparable is lower-is-better. Text-only columns are skipped.
HIGHER_BETTER = ("Throughput", "Bandwidth")
TEXT_ONLY = ("Corpus", "Superfiles")
# Cost cells are USD/queries-per-$ figures, not nanoseconds, and their keys
# embed volatile text — they don't diff cleanly. Skip them (the comment says
# so) rather than mis-unit them as latency.
COST_TOKENS = ("$", "cost", "measured", "per-unit")

# Primary metrics — controllable CPU / footprint, flagged at `threshold`.
PRIMARY_HEADERS = ("warm min", "Time", "Stored")
# Secondary metrics — cold (object-store network variance) and peak RSS
# (run-order biased) are too noisy to gate tightly; surfaced only on a large
# move, as directional side-notes.
SECONDARY_HEADERS = ("cold search", "Peak RSS")
SECONDARY_THRESHOLD_PCT = 30.0

# Map a report basename to (subsystem label, source area).
SUBSYSTEM = {
    "supertable": ("Ingest", "src/supertable/writer.rs"),
    "supertable_fts": ("FTS", "src/superfile/fts/"),
    "supertable_vector": ("Vector", "src/superfile/vector/"),
    "supertable_sql": ("SQL", "src/supertable/query/"),
    "superfile_fts": ("FTS", "src/superfile/fts/"),
    "superfile_vector": ("Vector", "src/superfile/vector/"),
    "sql": ("SQL", "src/supertable/query/"),
}

# Latency at/under this (ns) rounds to ~0.00 ms — a big percentage of nearly
# nothing. Don't flag it; real sub-ms queries above it still count. 0.1 ms.
MIN_LATENCY_NS = 100_000.0

DEFAULT_MODEL = "gpt-5.4"
DEFAULT_OUT = "/tmp/ai-summary.md"
DEFAULT_THRESHOLD = 15.0
# Cap each table so a broad swing can't blow past GitHub's comment limit.
MAX_ROWS = 15
# Foundry latency ceiling; the fallback tables cover us if we trip it.
HTTP_TIMEOUT_S = 60


def is_text_only(header):
    return any(t in header for t in TEXT_ONLY)


def higher_is_better(header):
    return any(t in header for t in HIGHER_BETTER)


def is_cost(header):
    h = header.lower()
    return any(t in h for t in COST_TOKENS)


def tier(header):
    """`primary`, `secondary`, or None (context — not surfaced)."""
    if any(t in header for t in PRIMARY_HEADERS):
        return "primary"
    if any(t in header for t in SECONDARY_HEADERS):
        return "secondary"
    return None


def is_latency(header):
    """Lower-is-better and measured in nanoseconds (Time, p50, cold, count())."""
    h = header.lower()
    return not higher_is_better(header) and "rss" not in h and "stored" not in h


def human(header, value):
    """Format a raw f64 into a unit appropriate to its header token."""
    h = header.lower()
    if "throughput" in h:
        return f"{value:,.0f} docs/s"
    if "bandwidth" in h:
        return f"{value / 1048576:,.1f} MiB/s"
    if "rss" in h or "stored" in h:
        if value >= 1073741824:
            return f"{value / 1073741824:.2f} GiB"
        return f"{value / 1048576:.1f} MiB"
    if value >= 1e9:
        return f"{value / 1e9:.2f} s"
    return f"{value / 1e6:.2f} ms"


def load(path):
    try:
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
        return {k: float(v) for k, v in obj.items() if isinstance(v, (int, float))}
    except (OSError, ValueError):
        return {}


def diff(reports, baseline_dir, current_dir, threshold):
    """Classify changes per report.

    Returns (regressions, improvements, had_baseline, cost_present).
    `had_baseline` distinguishes 'nothing moved' from 'nothing to compare'.
    """
    regressions, improvements = [], []
    had_baseline = False
    cost_present = False
    for report in reports:
        base = load(os.path.join(baseline_dir, f"{report}.json"))
        cur = load(os.path.join(current_dir, f"{report}.json"))
        if not cur:
            continue
        subsystem, area = SUBSYSTEM.get(report, (report, ""))
        for key, new in cur.items():
            parts = key.split("|")
            if len(parts) != KEY_PARTS:
                continue
            _anchor, _subtitle, label, header = parts
            if is_text_only(header):
                continue
            if is_cost(header):
                cost_present = True
                continue
            t = tier(header)
            if t is None:
                continue
            old = base.get(key)
            if old is None or old == 0.0:
                continue
            had_baseline = True
            if is_latency(header) and max(abs(old), abs(new)) < MIN_LATENCY_NS:
                continue
            limit = threshold if t == "primary" else max(threshold, SECONDARY_THRESHOLD_PCT)
            pct = (new - old) / old * 100.0
            if abs(pct) < limit:
                continue
            improved = pct > 0 if higher_is_better(header) else pct < 0
            entry = {
                "subsystem": subsystem,
                "area": area,
                "metric": f"{label} / {header}".strip(" /"),
                "change": f"{human(header, old)} → {human(header, new)}",
                "pct": round(pct, 1),
                "tier": t,
            }
            (improvements if improved else regressions).append(entry)
    regressions.sort(key=lambda e: -abs(e["pct"]))
    improvements.sort(key=lambda e: -abs(e["pct"]))
    return regressions, improvements, had_baseline, cost_present


def table(rows):
    out = ["| Subsystem | Metric | main → run | Δ |",
           "|---|---|---|---|"]
    for e in rows[:MAX_ROWS]:
        out.append(f"| {e['subsystem']} | {e['metric']} | {e['change']} "
                   f"| {e['pct']:+.0f}% |")
    extra = len(rows) - MAX_ROWS
    if extra > 0:
        out.append(f"| _+{extra} more_ | | | |")
    return "\n".join(out)


def narrate(payload, endpoint, key, model):
    """Ask Foundry for prose. Return None on any failure (fallback handles it)."""
    if not endpoint or not key:
        return None
    system = (
        "You are a performance engineer giving a PR reviewer the headline on a "
        "benchmark run. You are given JSON of metric changes ALREADY filtered "
        "past the noise threshold, plus any run failures. Each change has a "
        "`tier`: `primary` (latency / build / size — the real signal) or "
        "`secondary` (cold + peak RSS — noisy and run-order biased). Write 2-5 "
        "lines of GitHub markdown:\n"
        "- Line 1: a one-line verdict from PRIMARY changes only "
        "(net slower / net faster / mixed / no change).\n"
        "- Call out the biggest PRIMARY regressions and any failures. "
        "Mention SECONDARY changes only if large, in one trailing note framed as "
        "directional / likely noise — never let them drive the verdict.\n"
        "- Do NOT restate every row; cite ONLY numbers in the payload.\n"
        "- If there are no changes and no failures, say "
        f"'No significant changes (within ±{payload['threshold']}%).'\n"
        "No preamble, no sign-off, no tables."
    )
    body = json.dumps({
        "model": model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload)},
        ],
    }).encode("utf-8")
    url = endpoint.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "api-key": key,  # Azure accepts either on the /openai/v1 surface.
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            data = json.load(resp)
        text_out = data["choices"][0]["message"]["content"].strip()
        return text_out or None
    except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as exc:
        print(f"::warning::Foundry summary unavailable ({exc}); using tables", file=sys.stderr)
        return None


def main():
    reports = os.environ.get("REPORTS", "").split()
    baseline_dir = os.environ.get("BASELINE_DIR", "baseline")
    current_dir = os.environ.get("CURRENT_DIR", "current")
    out_file = os.environ.get("OUT_FILE", DEFAULT_OUT)
    label = os.environ.get("BENCH_LABEL", "benchmark")
    run_url = os.environ.get("RUN_URL", "")
    try:
        threshold = float(os.environ.get("BENCH_NOISE_THRESHOLD_PCT", DEFAULT_THRESHOLD))
    except ValueError:
        threshold = DEFAULT_THRESHOLD

    failures = [ln.strip() for ln in os.environ.get("ERRORS", "").splitlines() if ln.strip()]
    regressions, improvements, had_baseline, cost_present = diff(
        reports, baseline_dir, current_dir, threshold)

    prose = narrate(
        {
            "threshold": threshold,
            "regressions": regressions[:MAX_ROWS],
            "improvements": improvements[:MAX_ROWS],
            "failures": failures[:20],
        },
        os.environ.get("AZURE_AI_ENDPOINT", ""),
        os.environ.get("AZURE_AI_API_KEY", ""),
        os.environ.get("AZURE_AI_MODEL", DEFAULT_MODEL),
    )

    # Badge, verdict, and counts come from the primary signal only; secondary
    # (cold, peak RSS) is directional and never gates.
    prim_regr = [e for e in regressions if e["tier"] == "primary"]
    prim_impr = [e for e in improvements if e["tier"] == "primary"]
    secondary = [e for e in regressions + improvements if e["tier"] == "secondary"]

    if failures or (prim_regr and not prim_impr):
        badge = "🔴"
    elif prim_regr:
        badge = "🟡"
    elif prim_impr:
        badge = "🟢"
    else:
        badge = "⚪"
    counts = f"{len(prim_regr)} worse · {len(prim_impr)} better"
    parts = [f"## {badge} Benchmark `{label}` — {counts} (±{threshold:g}% vs main)", ""]

    if prose:
        parts += [prose, ""]
    elif failures:
        parts += ["### 🛑 Failures", "```", "\n".join(failures[:20]), "```", ""]
    elif not had_baseline:
        parts += ["_No `main` baseline to diff against (first run or new config) — see the full report._", ""]
    elif not prim_regr and not prim_impr and not secondary:
        parts += [f"No significant changes (within ±{threshold:g}%).", ""]

    if prim_regr:
        parts += [f"### 🔴 Worse ({len(prim_regr)})", table(prim_regr), ""]
    if prim_impr:
        parts += [f"<details><summary>🟢 Better ({len(prim_impr)})</summary>",
                  "", table(prim_impr), "", "</details>", ""]
    if secondary:
        parts += [f"<details><summary>Directional — cold / peak RSS, noisy ({len(secondary)})</summary>",
                  "", table(secondary), "", "</details>", ""]

    if prim_regr or prim_impr:
        touched = {e["subsystem"]: e["area"] for e in prim_regr + prim_impr if e.get("area")}
        if touched:
            parts.append("_Where to look: " + " · ".join(
                f"{s} → `{a}`" for s, a in sorted(touched.items())) + "_")
    if cost_present:
        parts.append("_Cost metrics (bytes / requests / queries-per-$) are not delta-tracked — see the full report._")
    if prim_regr or prim_impr or secondary:
        parts.append("")

    if run_url:
        parts.append(f"[Full report & logs ↗]({run_url})")

    body = "\n".join(parts).rstrip() + "\n"
    with open(out_file, "w", encoding="utf-8") as fh:
        fh.write(body)
    print(f"wrote {out_file}: {len(regressions)} worse, {len(improvements)} better, "
          f"{len(failures)} failure line(s), "
          f"baseline={'yes' if had_baseline else 'no'}")


if __name__ == "__main__":
    main()
