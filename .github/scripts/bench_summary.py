#!/usr/bin/env python3
"""Summarize benchmark deltas for a PR comment.

Diffs this run against the latest base-ref baseline, keeps only changes past
the noise threshold, and writes a deterministic SRE-style markdown summary.

Inputs (env):
  REPORTS                    space-separated report names (basenames, no .json)
  BASELINE_DIR               dir holding <report>.json from the base-ref baseline
  CURRENT_DIR                dir holding <report>.json from this run
  BENCH_NOISE_THRESHOLD_PCT  threshold in percent (default 5)
  OUT_FILE                   markdown destination (default /tmp/ai-summary.md)
  BENCH_LABEL                human label for the run (the `bench` input)
  BENCH_VM_SIZE              VM size label for context
  BENCH_LOCATION             region label for context
  BENCH_CPUSET               cpuset input label for context
  RUN_URL                    link to the full Actions run
  ERRORS                     newline-separated panic/error lines (may be empty)
"""

import json
import os

# Report keys are "anchor|subtitle|label|header"; split on this into 4 fields.
KEY_PARTS = 4

# Mirrors the bench renderer's Better enum: these headers are higher-is-better,
# everything else comparable is lower-is-better. Text-only columns are skipped.
HIGHER_BETTER = ("Throughput", "Bandwidth")
TEXT_ONLY = ("Corpus", "Superfiles")
# Cost cells are USD/queries-per-$ figures, not nanoseconds, and their keys
# embed volatile text - they do not diff cleanly.
COST_TOKENS = ("$", "cost", "measured", "per-unit")

# Primary metrics - controllable CPU / footprint, flagged at `threshold`.
PRIMARY_HEADERS = ("warm p90", "Time", "Stored")
# Secondary metrics - cold (object-store network variance) and peak RSS
# (run-order biased) are noisy and non-gating for PR decisions.
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

# Latency at/under this (ns) rounds to ~0.00 ms - a big percentage of nearly
# nothing. Do not flag it. 0.1 ms.
MIN_LATENCY_NS = 100_000.0

DEFAULT_OUT = "/tmp/ai-summary.md"
DEFAULT_THRESHOLD = 5.0


def is_text_only(header):
    return any(t in header for t in TEXT_ONLY)


def higher_is_better(header):
    return any(t in header for t in HIGHER_BETTER)


def is_cost(header):
    h = header.lower()
    return any(t in h for t in COST_TOKENS)


def tier(header):
    """`primary`, `secondary`, or None (context - not surfaced)."""
    if any(t in header for t in PRIMARY_HEADERS):
        return "primary"
    if any(t in header for t in SECONDARY_HEADERS):
        return "secondary"
    return None


def primary_latency_header_from_gate_metric(metric):
    m = (metric or "p90").strip().lower()
    if m == "min":
        return "warm min"
    if m == "p50":
        return "warm p50"
    return "warm p90"


def is_latency(header):
    """Lower-is-better and measured in nanoseconds (Time, p50, cold, count())."""
    h = header.lower()
    return not higher_is_better(header) and "rss" not in h and "stored" not in h


def human(header, value):
    """Format raw f64 into unit appropriate to header token."""
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
                "change": f"{human(header, old)} -> {human(header, new)}",
                "pct": round(pct, 1),
                "tier": t,
            }
            (improvements if improved else regressions).append(entry)
    regressions.sort(key=lambda e: -abs(e["pct"]))
    improvements.sort(key=lambda e: -abs(e["pct"]))
    return regressions, improvements, had_baseline, cost_present


def finding(entry):
    return f"- `{entry['metric']}`: {entry['change']} ({entry['pct']:+.0f}%)"


def main():
    global PRIMARY_HEADERS
    reports = os.environ.get("REPORTS", "").split()
    baseline_dir = os.environ.get("BASELINE_DIR", "baseline")
    current_dir = os.environ.get("CURRENT_DIR", "current")
    out_file = os.environ.get("OUT_FILE", DEFAULT_OUT)
    label = os.environ.get("BENCH_LABEL", "benchmark")
    base_ref = os.environ.get("BASE_REF_LABEL", "main")
    vm_size = os.environ.get("BENCH_VM_SIZE", "n/a")
    location = os.environ.get("BENCH_LOCATION", "n/a")
    cpuset = os.environ.get("BENCH_CPUSET", "auto")
    bench_gate_metric = os.environ.get("BENCH_GATE_METRIC", "p90")
    run_url = os.environ.get("RUN_URL", "")
    PRIMARY_HEADERS = (
        primary_latency_header_from_gate_metric(bench_gate_metric),
        "Time",
        "Stored",
    )
    try:
        threshold = float(os.environ.get("BENCH_NOISE_THRESHOLD_PCT", DEFAULT_THRESHOLD))
    except ValueError:
        threshold = DEFAULT_THRESHOLD

    failures = [ln.strip() for ln in os.environ.get("ERRORS", "").splitlines() if ln.strip()]
    regressions, improvements, had_baseline, cost_present = diff(
        reports, baseline_dir, current_dir, threshold
    )

    prim_regr = [e for e in regressions if e["tier"] == "primary"]
    prim_impr = [e for e in improvements if e["tier"] == "primary"]
    secondary_present = any(e["tier"] == "secondary" for e in regressions + improvements)

    if failures or prim_regr:
        status = "FAIL"
    else:
        status = "PASS"

    counts = f"{len(prim_regr)} regressions · {len(prim_impr)} improvements"
    parts = [f"## Benchmark Summary (A/B vs {base_ref})", ""]
    parts.append(f"Status: {status}")
    parts.append(f"Primary Gate: {counts}, threshold ±{threshold:g}%")
    parts.append(
        f"Run Context: bench={label} vm={vm_size} region={location} cpuset={cpuset or 'auto'}"
    )
    parts.append("")

    if failures:
        parts += ["### Failures", "```", "\n".join(failures[:20]), "```", ""]

    if not failures and not had_baseline:
        parts += [f"_No {base_ref} baseline to diff against (first run or new config)._", ""]
    elif not failures:
        parts += ["### Primary Findings", ""]
        if prim_regr:
            # Never truncate gate-failing signals.
            parts.append("**Regressions:**")
            parts.extend(finding(e) for e in prim_regr)
            parts.append("")
        if prim_impr:
            parts.append("**Improvements:**")
            parts.extend(finding(e) for e in prim_impr)
            parts.append("")
        if not prim_regr and not prim_impr:
            parts.append(f"- No primary regressions detected vs {base_ref}.")
            parts.append("")

    parts.append("### Decision")
    if failures or prim_regr:
        parts.append("- Merge Gate: FAIL")
        if prim_regr:
            parts.append("- Reason: Primary regressions above threshold.")
        else:
            parts.append("- Reason: Benchmark run reported failures.")
    else:
        parts.append("- Merge Gate: PASS")
        if prim_impr:
            parts.append("- Reason: No primary regressions; primary improvements observed.")
        else:
            parts.append("- Reason: No primary regressions above threshold.")
    parts.append("")

    parts.append("### Actions")
    if prim_regr:
        touched = {e["subsystem"]: e["area"] for e in prim_regr if e.get("area")}
        if touched:
            focus = " · ".join(f"`{a}`" for _, a in sorted(touched.items()))
            parts.append(f"- Owner: PR author")
            parts.append(f"- Follow-up: investigate {focus}.")
        else:
            parts.append("- Owner: PR author")
            parts.append("- Follow-up: investigate primary regressions.")
    else:
        parts.append("- Owner: PR author")
        parts.append("- Follow-up: none")
    parts.append("")

    parts.append("### Notes")
    if secondary_present or cost_present:
        parts.append(
            "- Cold-search and cost metrics measured, non-gating."
        )
    else:
        parts.append("- No secondary/cost deltas above reporting threshold.")

    if run_url:
        parts.append(f"- Full report & logs: {run_url}")

    body = "\n".join(parts).rstrip() + "\n"
    with open(out_file, "w", encoding="utf-8") as fh:
        fh.write(body)

    print(
        f"wrote {out_file}: {len(regressions)} regressions, {len(improvements)} improvements, "
        f"{len(failures)} failure line(s), baseline={'yes' if had_baseline else 'no'}"
    )


if __name__ == "__main__":
    main()
