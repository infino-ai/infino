#!/usr/bin/env python3
"""Summarize benchmark deltas for a PR comment.

Diffs this run against the latest base-ref baseline, keeps only changes past
the noise threshold, and writes a deterministic SRE-style markdown summary.

Inputs (env):
  REPORTS                    space-separated report names (basenames, no .json)
  BASELINE_DIR               dir holding <report>.json from the base-ref baseline
  CURRENT_DIR                dir holding <report>.json from this run
  BENCH_NOISE_THRESHOLD_PCT  threshold in percent (default 5)
  GATE_FILE                  merge-gate verdict destination (default /tmp/bench-gate.status)
  INFO_FILE                  non-gating findings (default /tmp/bench-gate.info)
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
from dataclasses import dataclass, field

# Report keys are "anchor|subtitle|label|header"; split on this into 4 fields.
KEY_PARTS = 4

# Mirrors the bench renderer's Better enum: these headers are higher-is-better,
# everything else comparable is lower-is-better. Text-only columns are skipped.
# Tokens are lowercase; matchers lowercase the header so capitalization in the
# report ("Time", "warm p90") never silently drops a metric from gating.
HIGHER_BETTER = ("throughput", "bandwidth")
TEXT_ONLY = ("corpus", "superfiles")

# Byte-valued headers: memory footprints, on-disk sizes, and object-store
# transfer volumes. These must be classified explicitly rather than inferred,
# because the merge gate treats every metric it believes is time-valued as
# nanoseconds: a byte metric reaching it is both formatted as a duration and
# compared against the millisecond threshold, so a 494 MiB -> 788 MiB
# file-backed RSS move blocked a merge as "294 ms over a 5 ms gate".
BYTE_HEADERS = (
    "rss",
    "stored",
    "peak anon",
    "peak file",
    "payload",
    "bytes",
    "downloaded",
    "uploaded",
)

# Count-valued headers: request and row counts, thread counts, and ratios
# (recall, share). Also not nanoseconds. Substring match on the lowercased
# header, so "Cold open GET/bytes" and "Warm GET/query" are both covered.
COUNT_HEADERS = (
    "get",
    "put",
    "head",
    "requests",
    "rows",
    "hits",
    "threads",
    "recall",
    "share of node",
)
# Cost cells are USD/queries-per-$ figures, not nanoseconds, and their keys
# embed volatile text - they do not diff cleanly.
COST_TOKENS = ("$", "cost", "measured", "per-unit")

# Secondary metrics - cold (object-store network variance) and peak RSS
# (run-order biased) are noisy and non-gating for PR decisions. Substring
# match, so every cold spelling is covered, not just one.
SECONDARY_HEADERS = ("cold", "peak rss")
SECONDARY_THRESHOLD_PCT = 30.0

# Advisory (report-only) primary tier beyond the gate metric: build time,
# stored size, and transition walls. Flagged at `threshold`; none block.
PRIMARY_EXTRA_HEADERS = ("time", "stored", "wall")

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

# Below this, latency rounds to ~0.00 ms - a big percent of nearly nothing. 0.1 ms.
MIN_LATENCY_NS = 100_000.0

# Smallest absolute latency move worth flagging - smaller deltas are noise. 0.1 ms.
MIN_LATENCY_DELTA_NS = 100_000.0

DEFAULT_OUT = "/tmp/ai-summary.md"
DEFAULT_THRESHOLD = 5.0
DEFAULT_GATE_FILE = "/tmp/bench-gate.status"
DEFAULT_INFO_FILE = "/tmp/bench-gate.info"

# Merge-blocking gate: the run's designated gate metric (BENCH_GATE_METRIC,
# default warm p90) must be worse than the main baseline by BOTH of these to
# block the merge. The AND is the noise guard: a big percent of a
# sub-millisecond metric never blocks, and neither does a small-percent
# wiggle on a long wall. The 5% advisory tiers above stay as REPORTING
# thresholds only; this pair is the sole blocking criterion (enforced by
# the workflow's "Enforce benchmark merge gate" step reading GATE_FILE).
#
# Exactly one statistic gates. Every other time-valued column is reported as
# an informational mover, because each of the three families below failed the
# gate on runs where BOTH A/B arms built the SAME commit — a push to main
# benchmarks main against main, so those verdicts were measurement noise:
#   - Tail percentiles. `warm p99` is not a percentile at these sample counts:
#     the benches take 30 (supertable) or 50 (superfile) warm iterations and
#     nearest-rank p99 lands on the last one, so it IS the max sample and one
#     slow iteration on a shared CI VM trips the gate.
#   - Cold latency. A fresh-cache network round trip tracks the object store,
#     not the diff.
#   - Transition walls. `delta commit`, drain, and optimize are each ONE
#     un-repeated mutation against a real object store, and the A/B always
#     runs the ref arm second on the same VM and storage account — so they
#     carry a systematic order bias, not just variance, and no threshold
#     separates that from a real regression.
GATE_REL_PCT = 50.0
GATE_ABS_NS = 5_000_000.0


@dataclass
class Diff:
    """The classified outcome of one A/B comparison."""

    regressions: list = field(default_factory=list)
    improvements: list = field(default_factory=list)
    blocking: list = field(default_factory=list)
    informational: list = field(default_factory=list)
    had_baseline: bool = False
    cost_present: bool = False
    # False with a baseline present means no report emitted the gate metric,
    # so nothing could have blocked — a dead gate, not a clean run.
    gate_metric_seen: bool = False


def is_text_only(header):
    return any(t in header.lower() for t in TEXT_ONLY)


def higher_is_better(header):
    return any(t in header.lower() for t in HIGHER_BETTER)


def is_cost(header):
    h = header.lower()
    return any(t in h for t in COST_TOKENS)


def is_gating(header, gate_header):
    """Whether `header` is the one statistic allowed to block a merge.

    An allowlist, not a denylist: every other time-valued column is reported
    and never blocks. Exact match on the run's designated gate metric, with
    both sides normalized so neither report capitalization nor a caller's
    spelling can silently kill the gate — see GATE_REL_PCT for the exclusions.
    """
    return header.strip().lower() == gate_header.strip().lower()


def tier(header, primary_headers):
    """`primary`, `secondary`, or None (context - not surfaced).

    Primary metrics - controllable CPU / footprint, flagged at `threshold`.
    The latency header among them tracks the run's gate metric, so it is
    resolved at call time rather than fixed as a module constant.
    """
    h = header.lower()
    if any(t in h for t in primary_headers):
        return "primary"
    if any(t in h for t in SECONDARY_HEADERS):
        return "secondary"
    return None


def primary_latency_header_from_gate_metric(metric):
    m = (metric or "p90").strip().lower()
    if m == "min":
        return "warm min"
    if m == "p50":
        return "warm p50"
    return "warm p90"


def is_bytes(header):
    return any(t in header.lower() for t in BYTE_HEADERS)


def is_count(header):
    return any(t in header.lower() for t in COUNT_HEADERS)


def is_latency(header):
    """Lower-is-better AND measured in nanoseconds (Time, p50, cold, wall).

    Only these are eligible for the merge gate, which compares raw deltas
    against a nanosecond threshold. Byte- and count-valued headers are
    excluded by name: judging them by the absence of a couple of substrings
    silently swept every new non-time column into the latency gate.
    `throughput` / `bandwidth` are excluded as higher-is-better.
    """
    if higher_is_better(header) or is_bytes(header) or is_count(header):
        return False
    return True


def human(header, value):
    """Format raw f64 into unit appropriate to header token."""
    h = header.lower()
    if "throughput" in h:
        return f"{value:,.0f} docs/s"
    if "bandwidth" in h:
        return f"{value / 1048576:,.1f} MiB/s"
    if is_bytes(header):
        if value >= 1073741824:
            return f"{value / 1073741824:.2f} GiB"
        if value >= 1048576:
            return f"{value / 1048576:.1f} MiB"
        if value >= 1024:
            return f"{value / 1024:.1f} KiB"
        return f"{value:,.0f} B"
    if is_count(header):
        return f"{value:,.0f}"
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


def entry(subsystem, area, label, header, old, new, pct, kind):
    """One classified A/B finding, as `finding()` expects to render it.

    `kind` is the bucket the finding landed in: the gate pass supplies
    blocking / informational, the advisory pass a `tier()` result.
    """
    return {
        "subsystem": subsystem,
        "area": area,
        "metric": f"{label} / {header}".strip(" /"),
        "change": f"{human(header, old)} -> {human(header, new)}",
        "pct": round(pct, 1),
        "tier": kind,
    }


def diff(reports, baseline_dir, current_dir, threshold, gate_header):
    """Classify changes per report into a `Diff`.

    `regressions` / `improvements` are the advisory findings (the 5%/30%
    tier thresholds — report-only). `blocking` is the merge gate: the
    designated gate metric worse than baseline by BOTH >GATE_REL_PCT and
    >GATE_ABS_NS. `informational` holds the same-sized moves on every other
    time-valued header.
    """
    primary_headers = (gate_header, *PRIMARY_EXTRA_HEADERS)
    regressions, improvements, blocking, informational = [], [], [], []
    had_baseline = False
    cost_present = False
    gate_metric_seen = False
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
            old = base.get(key)
            if old is None or old == 0.0:
                continue
            had_baseline = True
            latency = is_latency(header)
            delta = new - old
            pct = delta / old * 100.0

            # Gate pass. Every time-valued metric is measured against the gate
            # thresholds; only the designated gate metric may block on it.
            if latency:
                gating = is_gating(header, gate_header)
                gate_metric_seen |= gating
                if delta > GATE_ABS_NS and pct > GATE_REL_PCT:
                    verdict = "blocking" if gating else "informational"
                    bucket = blocking if gating else informational
                    bucket.append(
                        entry(subsystem, area, label, header, old, new, pct, verdict)
                    )

            # Advisory pass. The 5%/30% tier thresholds — report-only, and
            # noise-floored so a big percent of nearly nothing never shows up.
            t = tier(header, primary_headers)
            if t is None:
                continue
            if latency and (
                max(abs(old), abs(new)) < MIN_LATENCY_NS or abs(delta) < MIN_LATENCY_DELTA_NS
            ):
                continue
            limit = threshold if t == "primary" else max(threshold, SECONDARY_THRESHOLD_PCT)
            if abs(pct) < limit:
                continue
            improved = pct > 0 if higher_is_better(header) else pct < 0
            bucket = improvements if improved else regressions
            bucket.append(entry(subsystem, area, label, header, old, new, pct, t))
    for bucket in (regressions, improvements, blocking, informational):
        bucket.sort(key=lambda e: -abs(e["pct"]))
    return Diff(
        regressions=regressions,
        improvements=improvements,
        blocking=blocking,
        informational=informational,
        had_baseline=had_baseline,
        cost_present=cost_present,
        gate_metric_seen=gate_metric_seen,
    )


def finding(e):
    return f"- `{e['metric']}`: {e['change']} ({e['pct']:+.0f}%)"


def main():
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
    gate_header = primary_latency_header_from_gate_metric(bench_gate_metric)
    try:
        threshold = float(os.environ.get("BENCH_NOISE_THRESHOLD_PCT", DEFAULT_THRESHOLD))
    except ValueError:
        threshold = DEFAULT_THRESHOLD

    gate_file = os.environ.get("GATE_FILE", DEFAULT_GATE_FILE)
    info_file = os.environ.get("INFO_FILE", DEFAULT_INFO_FILE)
    failures = [ln.strip() for ln in os.environ.get("ERRORS", "").splitlines() if ln.strip()]
    ab = diff(reports, baseline_dir, current_dir, threshold, gate_header)
    blocking, informational = ab.blocking, ab.informational
    regressions, improvements = ab.regressions, ab.improvements

    prim_regr = [e for e in regressions if e["tier"] == "primary"]
    prim_impr = [e for e in improvements if e["tier"] == "primary"]
    secondary_present = any(e["tier"] == "secondary" for e in regressions + improvements)
    # A gate that cannot fire reads as a pass; say so rather than stay silent.
    dead_gate = ab.had_baseline and not ab.gate_metric_seen
    if dead_gate:
        print(f"::warning::no report emitted `{gate_header}` — the merge gate had nothing to check")

    if failures or blocking:
        status = "FAIL"
    else:
        status = "PASS"

    counts = f"{len(prim_regr)} regressions · {len(prim_impr)} improvements"
    parts = [f"## Benchmark Summary (A/B vs {base_ref})", ""]
    parts.append(f"Status: {status}")
    parts.append(
        f"Merge Gate (blocking): {len(blocking)} `{gate_header}` regressions past "
        f">{GATE_REL_PCT:g}% AND >{GATE_ABS_NS / 1e6:g} ms vs {base_ref}"
    )
    parts.append(f"Advisory findings: {counts}, threshold ±{threshold:g}% (report-only)")
    parts.append(f"Non-gating movers (informational): {len(informational)}")
    if dead_gate:
        parts.append(f"Warning: no report emitted `{gate_header}` — the gate checked nothing")
    parts.append(
        f"Run Context: bench={label} vm={vm_size} region={location} cpuset={cpuset or 'auto'}"
    )
    parts.append("")

    if failures:
        parts += ["### Failures", "```", "\n".join(failures[:20]), "```", ""]

    if blocking:
        # Never truncate merge-blocking signals.
        parts += ["### Blocking Regressions (merge gate)", ""]
        parts.extend(finding(e) for e in blocking)
        parts.append("")

    if informational:
        parts += [
            "### Non-Gating Movers (informational)",
            "",
            "_Tail percentiles, cold reads, and one-shot transition walls. "
            "Measured and reported; they never block a merge._",
            "",
        ]
        parts.extend(finding(e) for e in informational)
        parts.append("")

    if not failures and not ab.had_baseline:
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
    if failures or blocking:
        parts.append("- Merge Gate: FAIL")
        if blocking:
            parts.append(
                f"- Reason: `{gate_header}` worse than {base_ref} by both "
                f">{GATE_REL_PCT:g}% and >{GATE_ABS_NS / 1e6:g} ms."
            )
        else:
            parts.append("- Reason: Benchmark run reported failures.")
    else:
        parts.append("- Merge Gate: PASS")
        if prim_regr:
            parts.append(
                "- Reason: advisory regressions only — none past the blocking pair."
            )
        else:
            parts.append("- Reason: No blocking regressions.")
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
    if secondary_present or ab.cost_present:
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

    # The enforcement step reads this verdict; FAIL lines carry the reasons.
    with open(gate_file, "w", encoding="utf-8") as fh:
        if failures or blocking:
            fh.write("FAIL\n")
            for ln in failures[:20]:
                fh.write(f"failure: {ln}\n")
            for e in blocking:
                fh.write(f"{e['metric']}: {e['change']} ({e['pct']:+.0f}%)\n")
        else:
            fh.write("PASS\n")

    # Surfaced as workflow warnings; deliberately kept out of the verdict file.
    with open(info_file, "w", encoding="utf-8") as fh:
        for e in informational:
            fh.write(f"{e['metric']}: {e['change']} ({e['pct']:+.0f}%)\n")

    print(
        f"wrote {out_file}: gate={gate_header}, {len(blocking)} blocking, "
        f"{len(informational)} informational, {len(regressions)} advisory regressions, "
        f"{len(improvements)} improvements, {len(failures)} failure line(s), "
        f"baseline={'yes' if ab.had_baseline else 'no'}"
    )


if __name__ == "__main__":
    main()
