#!/usr/bin/env python3
"""Summarize benchmark deltas for a PR comment.

Diffs this run against the `main` baseline, keeps only changes past the noise
threshold, annotates each with main's own trend, and writes a lean markdown
summary. Foundry narrates; the percentages are computed here, never by the
model. On any model failure or missing creds the deterministic delta table is
the summary on its own. Stdlib only — the runner needs no pip install.

Inputs (env):
  REPORTS                  space-separated report names (basenames, no .json)
  BASELINE_DIR             dir holding <report>.json from the main baseline
  CURRENT_DIR              dir holding <report>.json from this run
  HISTORY_DIR              dir with <report>/<idx>-<sha>.json main-history points
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

# Map a report basename to (subsystem label, source area) shown in the summary.
SUBSYSTEM = {
    "supertable": ("Ingest", "src/supertable/writer.rs"),
    "supertable_fts": ("FTS", "src/superfile/fts/"),
    "supertable_vector": ("Vector", "src/superfile/vector/"),
    "supertable_sql": ("SQL", "src/supertable/query/"),
    "superfile_fts": ("FTS", "src/superfile/fts/"),
    "superfile_vector": ("Vector", "src/superfile/vector/"),
    "sql": ("SQL", "src/supertable/query/"),
}

# Recall is a pass/fail merge gate, not a noise-banded latency metric: any
# drop below this bar is flagged regardless of percentage delta.
RECALL_GATE = 0.99

DEFAULT_MODEL = "gpt-5.4"
DEFAULT_OUT = "/tmp/ai-summary.md"
DEFAULT_THRESHOLD = 10.0
# Most recent main commits to read when describing main's own trend.
HISTORY_WINDOW = 10
# Cap each list so a broad regression can't blow past GitHub's comment limit.
MAX_ROWS = 12
# Foundry latency ceiling; the fallback table covers us if we trip it.
HTTP_TIMEOUT_S = 60


def is_text_only(header):
    return any(t in header for t in TEXT_ONLY)


def higher_is_better(header):
    return any(t in header for t in HIGHER_BETTER)


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
    # Latency-like (Time, p50/p90, cold, latency) is stored in nanoseconds.
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


def load_history(history_dir, report):
    """Main-history metric maps for `report`, oldest→newest (≤ window)."""
    path = os.path.join(history_dir, report)
    try:
        names = sorted(n for n in os.listdir(path) if n.endswith(".json"))
    except OSError:
        return []
    maps = [load(os.path.join(path, n)) for n in names[-HISTORY_WINDOW:]]
    return [m for m in maps if m]


def main_trend(history_maps, key, header, threshold):
    """How `main` drifted across the window — a PR regression on a metric
    main has already been pushing the wrong way is the case worth flagging.
    Returns (note, short), or (None, None) without enough points."""
    series = [m[key] for m in history_maps if key in m and m[key] != 0.0]
    if len(series) < 2 or series[0] == 0.0:
        return None, None
    drift = (series[-1] - series[0]) / series[0] * 100.0
    n = len(series)
    if abs(drift) < threshold:
        return f"main stable ({drift:+.0f}% over {n}c)", "stable"
    improved = drift > 0 if higher_is_better(header) else drift < 0
    word = "improving" if improved else "worsening"
    return f"main {word} {drift:+.0f}% over {n} commits", f"{drift:+.0f}%/{n}c"


def diff(reports, baseline_dir, current_dir, history_dir, threshold):
    """Classify changes per report.

    Returns (regressions, improvements, gates, had_baseline, cost_present).
    `had_baseline` distinguishes 'nothing moved' from 'nothing to compare'.
    """
    regressions, improvements, gates = [], [], []
    had_baseline = False
    cost_present = False
    for report in reports:
        base = load(os.path.join(baseline_dir, f"{report}.json"))
        cur = load(os.path.join(current_dir, f"{report}.json"))
        if not cur:
            continue
        subsystem, area = SUBSYSTEM.get(report, (report, ""))
        history = load_history(history_dir, report)
        for key, new in cur.items():
            parts = key.split("|")
            if len(parts) != KEY_PARTS:
                continue
            _anchor, _subtitle, label, header = parts
            if "$" in key:
                cost_present = True
            # Recall is a hard gate, evaluated on its absolute value.
            if "recall" in f"{label} {header}".lower():
                breach = new < RECALL_GATE if new <= 1.0 else new < RECALL_GATE * 100
                gates.append({
                    "subsystem": subsystem,
                    "metric": f"{label} / {header}".strip(" /"),
                    "value": f"{new:.4f}" if new <= 1.0 else f"{new:.2f}",
                    "breach": breach,
                })
                continue
            if is_text_only(header):
                continue
            old = base.get(key)
            if old is None or old == 0.0:
                continue
            had_baseline = True
            pct = (new - old) / old * 100.0
            if abs(pct) < threshold:
                continue
            improved = pct > 0 if higher_is_better(header) else pct < 0
            note, short = main_trend(history, key, header, threshold)
            entry = {
                "subsystem": subsystem,
                "area": area,
                "metric": f"{label} / {header}".strip(" /"),
                "old": human(header, old),
                "new": human(header, new),
                "pct": round(pct, 1),
                "verdict": "improvement" if improved else "regression",
                "main_trend": note,
                "main_trend_short": short,
            }
            (improvements if improved else regressions).append(entry)
    regressions.sort(key=lambda e: -abs(e["pct"]))
    improvements.sort(key=lambda e: -abs(e["pct"]))
    return regressions, improvements, gates, had_baseline, cost_present


def verdict_badge(regressions, gates, failures, improvements):
    if failures or any(g["breach"] for g in gates) or regressions:
        return "🔴"
    if improvements:
        return "🟢"
    return "⚪"


def gate_table(gates):
    breached = [g for g in gates if g["breach"]]
    if not breached:
        return ""
    rows = ["| Gate | Subsystem | Value | Bar |", "|---|---|---|---|"]
    for g in breached:
        rows.append(f"| 🛑 {g['metric']} | {g['subsystem']} | {g['value']} | ≥ {RECALL_GATE} |")
    return "\n".join(rows)


def delta_table(regressions, improvements):
    """Deterministic markdown table — the always-correct fallback body."""
    rows = regressions[:MAX_ROWS] + improvements[:MAX_ROWS]
    if not rows:
        return "", 0
    out = ["| Verdict | Subsystem | Metric | main | this run | Δ% | Main trend |",
           "|---|---|---|---|---|---|---|"]
    for e in rows:
        mark = "🔴" if e["verdict"] == "regression" else "🟢"
        trend = e.get("main_trend_short") or "—"
        out.append(
            f"| {mark} {e['verdict']} | {e['subsystem']} | {e['metric']} "
            f"| {e['old']} | {e['new']} | {e['pct']:+.1f}% | {trend} |"
        )
    dropped = max(0, len(regressions) - MAX_ROWS) + max(0, len(improvements) - MAX_ROWS)
    return "\n".join(out), dropped


def narrate(payload, endpoint, key, model):
    """Ask Foundry for prose. Return None on any failure (fallback handles it)."""
    if not endpoint or not key:
        return None
    system = (
        "You are a performance engineer summarizing benchmark results for a "
        "pull-request reviewer. You are given a JSON payload of metric changes "
        "ALREADY filtered to exceed the noise threshold, plus recall gate "
        "checks and any run failures. Write a terse summary in GitHub markdown:\n"
        "- Open with a one-line verdict (net regression / net improvement / mixed / clean).\n"
        "- Call out failures and any breached gate FIRST and plainly.\n"
        "- Then bullet the notable changes grouped by subsystem (Ingest, FTS, "
        "Vector, SQL). Regressions before improvements.\n"
        "- When a change has a 'main_trend', weave it in: a regression on a "
        "metric main is already worsening is more serious than a one-off.\n"
        "- Cite ONLY numbers in the payload. Never invent or recompute a value.\n"
        "- If there are no changes, no breached gates, and no failures, say "
        f"'No significant changes (within ±{payload['threshold']}%).'\n"
        "Keep it to what a reviewer needs at a glance. No preamble, no sign-off, "
        "no restating the raw table."
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
        text = data["choices"][0]["message"]["content"].strip()
        return text or None
    except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as exc:
        print(f"::warning::Foundry summary unavailable ({exc}); using delta table", file=sys.stderr)
        return None


def main():
    reports = os.environ.get("REPORTS", "").split()
    baseline_dir = os.environ.get("BASELINE_DIR", "baseline")
    current_dir = os.environ.get("CURRENT_DIR", "current")
    history_dir = os.environ.get("HISTORY_DIR", "history")
    out_file = os.environ.get("OUT_FILE", DEFAULT_OUT)
    label = os.environ.get("BENCH_LABEL", "benchmark")
    run_url = os.environ.get("RUN_URL", "")
    try:
        threshold = float(os.environ.get("BENCH_NOISE_THRESHOLD_PCT", DEFAULT_THRESHOLD))
    except ValueError:
        threshold = DEFAULT_THRESHOLD

    failures = [ln.strip() for ln in os.environ.get("ERRORS", "").splitlines() if ln.strip()]
    regressions, improvements, gates, had_baseline, cost_present = diff(
        reports, baseline_dir, current_dir, history_dir, threshold)

    payload = {
        "threshold": threshold,
        "regressions": regressions[:MAX_ROWS],
        "improvements": improvements[:MAX_ROWS],
        "gates": gates,
        "failures": failures[:20],
    }

    prose = narrate(
        payload,
        os.environ.get("AZURE_AI_ENDPOINT", ""),
        os.environ.get("AZURE_AI_API_KEY", ""),
        os.environ.get("AZURE_AI_MODEL", DEFAULT_MODEL),
    )

    badge = verdict_badge(regressions, gates, failures, improvements)
    parts = [f"## {badge} Benchmark `{label}` — significant changes (±{threshold:g}% threshold)", ""]

    if prose:
        parts += [prose, ""]
    else:
        # Deterministic fallback body: gates + failures + the delta table.
        if not had_baseline:
            parts += ["_No `main` baseline to diff against (first run or new config) — see the full report._", ""]
        elif not regressions and not improvements and not gates:
            parts += [f"No significant changes (within ±{threshold:g}%).", ""]
        gt = gate_table(gates)
        if gt:
            parts += ["### 🛑 Gate breach", gt, ""]
        if failures:
            parts += ["### 🛑 Failures", "```", "\n".join(failures[:20]), "```", ""]

    table, dropped = delta_table(regressions, improvements)
    if table:
        parts += ["<details><summary>Significant deltas vs <code>main</code></summary>", "", table]
        if dropped:
            parts += ["", f"_…and {dropped} more change(s) past threshold — see the full report._"]
        parts += ["", "</details>", ""]

    # Source-area hints for the subsystems that actually moved.
    touched = {}
    for e in regressions + improvements:
        if e.get("area"):
            touched[e["subsystem"]] = e["area"]
    if touched:
        hints = " · ".join(f"{s} → `{a}`" for s, a in sorted(touched.items()))
        parts += [f"_Where to look: {hints}_", ""]

    if cost_present:
        parts += ["_Cost metrics (bytes / requests / queries-per-$) are not delta-tracked "
                  "(volatile keys) — check the full report._", ""]

    if run_url:
        parts.append(f"[Full report & logs ↗]({run_url})")

    body = "\n".join(parts).rstrip() + "\n"
    with open(out_file, "w", encoding="utf-8") as fh:
        fh.write(body)
    print(f"wrote {out_file}: {len(regressions)} regression(s), {len(improvements)} "
          f"improvement(s), {len(gates)} gate(s), {len(failures)} failure line(s), "
          f"baseline={'yes' if had_baseline else 'no'}")


if __name__ == "__main__":
    main()
