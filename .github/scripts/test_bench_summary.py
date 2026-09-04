#!/usr/bin/env python3
"""Unit tests for `bench_summary.py`'s metric classification.

Run: `make bench-gate` (or this file directly). Two rules decide whether a
benchmark metric can block a merge, and both are pinned here:

  - Units. The gate compares raw deltas against a NANOSECOND threshold, so a
    byte- or count-valued header reaching it is both formatted as a duration
    and judged against a millisecond floor — a 494 MiB -> 788 MiB file-backed
    RSS move once blocked a PR as "294 ms past a 5 ms gate".
  - The gate allowlist. Of the genuinely time-valued headers, exactly one
    statistic blocks; the rest are reported only.
"""

import json
import os
import tempfile
import unittest

import bench_summary as bs

# A move well past both gate thresholds (>50% and >5 ms), in nanoseconds.
SMALL_NS = 10_000_000.0
BIG_NS = 30_000_000.0

# A move past the percent threshold but under the 5 ms absolute floor.
TINY_OLD_NS = 1_000_000.0
TINY_NEW_NS = 2_000_000.0

# A move past the absolute floor but under the 50% percent threshold.
WIDE_OLD_NS = 100_000_000.0
WIDE_NEW_NS = 130_000_000.0

BYTES_PER_MIB = 1048576

# The file-backed RSS move, in bytes, that blocked a merge when read as
# nanoseconds. Kept as the concrete regression case for the units rule.
RSS_OLD_BYTES = 494.23 * BYTES_PER_MIB
RSS_NEW_BYTES = 787.78 * BYTES_PER_MIB

REPORT = "supertable_sql"

# Every header the bench emits that reaches the gate, with whether it is
# genuinely nanosecond-valued (and therefore gate-eligible).
HEADER_UNITS = [
    # Byte-valued — footprints, on-disk sizes, transfer volumes.
    ("Peak file", False),
    ("Peak anon", False),
    ("Peak RSS", False),
    ("Median RSS", False),
    ("P90 RSS", False),
    ("Stored", False),
    ("Payload", False),
    ("Downloaded", False),
    ("Uploaded", False),
    ("Cold open GET/bytes", False),
    ("Hidden data GET/bytes", False),
    ("Manifest GET/bytes", False),
    ("User data GET/bytes", False),
    # Count- and ratio-valued.
    ("Warm GET/query", False),
    ("Requests", False),
    ("Rows", False),
    ("Hits (u/h)", False),
    ("OS threads", False),
    ("Recall", False),
    # Higher-is-better rates.
    ("Throughput", False),
    ("Bandwidth", False),
    # Genuine nanosecond metrics — these MUST stay gate-eligible.
    ("Time", True),
    ("Wall", True),
    ("warm p50", True),
    ("warm p90", True),
    ("warm p99", True),
    ("+fetch p50", True),
    ("cold 1st query (median)", True),
    ("cold open (median)", True),
    ("Delta commit", True),
    ("Drain", True),
    ("Optimize", True),
    ("Ingest", True),
]


class HeaderUnits(unittest.TestCase):
    """Only nanosecond-valued headers may reach the gate's ns thresholds."""

    def test_gate_eligibility_matches_real_units(self):
        for header, want in HEADER_UNITS:
            with self.subTest(header=header):
                self.assertEqual(bs.is_latency(header), want)

    def test_byte_header_cannot_reach_the_gate(self):
        """The RSS move that blocked a merge clears both ns thresholds."""
        delta = RSS_NEW_BYTES - RSS_OLD_BYTES
        pct = delta / RSS_OLD_BYTES * 100.0
        self.assertGreater(delta, bs.GATE_ABS_NS)
        self.assertGreater(pct, bs.GATE_REL_PCT)
        # So its exclusion has to come from the units rule, not the thresholds.
        self.assertFalse(bs.is_latency("Peak file"))

    def test_bytes_render_as_bytes(self):
        rendered = bs.human("Peak file", RSS_NEW_BYTES)
        self.assertIn("MiB", rendered)
        self.assertNotIn("ms", rendered)

    def test_latency_rendering_is_untouched(self):
        self.assertEqual(bs.human("warm p50", 195_150_000.0), "195.15 ms")
        self.assertEqual(bs.human("Wall", 24_400_000_000.0), "24.40 s")


def key(label, header):
    """A report key: anchor|subtitle|label|header."""
    return f"anchor|sub|{label}|{header}"


class GateClassification(unittest.TestCase):
    """Of the time-valued headers, exactly one statistic blocks a merge."""

    def diff(self, metrics, gate_header="warm p90"):
        """Run `diff` over one report of {(label, header): (old, new)}."""
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = os.path.join(tmp, "baseline")
            cur_dir = os.path.join(tmp, "current")
            os.mkdir(base_dir)
            os.mkdir(cur_dir)
            base = {key(l, h): old for (l, h), (old, _) in metrics.items()}
            cur = {key(l, h): new for (l, h), (_, new) in metrics.items()}
            for d, m in ((base_dir, base), (cur_dir, cur)):
                with open(os.path.join(d, f"{REPORT}.json"), "w", encoding="utf-8") as fh:
                    json.dump(m, fh)
            return bs.diff([REPORT], base_dir, cur_dir, bs.DEFAULT_THRESHOLD, gate_header)

    def metrics(self, result, bucket):
        return {e["metric"] for e in getattr(result, bucket)}

    def test_gate_metric_blocks(self):
        """The designated gate metric past both thresholds blocks the merge."""
        r = self.diff({("point lookup", "warm p90"): (SMALL_NS, BIG_NS)})
        self.assertEqual(self.metrics(r, "blocking"), {"point lookup / warm p90"})
        self.assertEqual(self.metrics(r, "informational"), set())

    def test_tail_percentile_never_blocks(self):
        """`warm p99` is the max sample at 30-50 iters — it reports, never blocks."""
        r = self.diff({("point lookup", "warm p99"): (SMALL_NS, BIG_NS)})
        self.assertEqual(self.metrics(r, "blocking"), set())
        self.assertEqual(self.metrics(r, "informational"), {"point lookup / warm p99"})

    def test_median_is_not_the_gate_metric(self):
        """Only the designated statistic gates; p50 reports alongside p99."""
        r = self.diff({("point lookup", "warm p50"): (SMALL_NS, BIG_NS)})
        self.assertEqual(self.metrics(r, "blocking"), set())
        self.assertEqual(self.metrics(r, "informational"), {"point lookup / warm p50"})

    def test_transition_wall_never_blocks(self):
        """A one-shot mutation wall against a real object store reports only."""
        r = self.diff({("delta commit", "Wall"): (SMALL_NS, BIG_NS)})
        self.assertEqual(self.metrics(r, "blocking"), set())
        self.assertEqual(self.metrics(r, "informational"), {"delta commit / Wall"})

    def test_cold_never_blocks(self):
        """Cold latency tracks the object store, not the diff."""
        r = self.diff({("five_term_or", "cold 1st query (median)"): (SMALL_NS, BIG_NS)})
        self.assertEqual(self.metrics(r, "blocking"), set())
        self.assertEqual(
            self.metrics(r, "informational"), {"five_term_or / cold 1st query (median)"}
        )

    def test_gate_metric_tracks_bench_gate_metric(self):
        """`BENCH_GATE_METRIC=p50` moves the gate to p50 and demotes p90."""
        r = self.diff(
            {
                ("point lookup", "warm p50"): (SMALL_NS, BIG_NS),
                ("point lookup", "warm p90"): (SMALL_NS, BIG_NS),
            },
            gate_header="warm p50",
        )
        self.assertEqual(self.metrics(r, "blocking"), {"point lookup / warm p50"})
        self.assertEqual(self.metrics(r, "informational"), {"point lookup / warm p90"})

    def test_header_match_is_case_insensitive(self):
        """Report capitalization must never silently drop the gate metric."""
        r = self.diff({("point lookup", "Warm P90"): (SMALL_NS, BIG_NS)})
        self.assertEqual(self.metrics(r, "blocking"), {"point lookup / Warm P90"})

    def test_percent_alone_does_not_block(self):
        """+100% of 1 ms is under the 5 ms absolute floor — noise guard."""
        r = self.diff({("point lookup", "warm p90"): (TINY_OLD_NS, TINY_NEW_NS)})
        self.assertEqual(self.metrics(r, "blocking"), set())

    def test_absolute_alone_does_not_block(self):
        """+30 ms on a 100 ms metric is under the 50% floor — noise guard."""
        r = self.diff({("point lookup", "warm p90"): (WIDE_OLD_NS, WIDE_NEW_NS)})
        self.assertEqual(self.metrics(r, "blocking"), set())

    def test_improvement_does_not_block(self):
        r = self.diff({("point lookup", "warm p90"): (BIG_NS, SMALL_NS)})
        self.assertEqual(self.metrics(r, "blocking"), set())
        self.assertEqual(self.metrics(r, "informational"), set())

    def test_higher_is_better_never_blocks(self):
        """Throughput rises are good and its raw value is not nanoseconds."""
        r = self.diff({("ingest", "Throughput"): (SMALL_NS, BIG_NS)})
        self.assertEqual(self.metrics(r, "blocking"), set())

    def test_byte_metric_never_blocks(self):
        """A byte column past both ns thresholds reaches neither bucket."""
        r = self.diff({("point lookup", "Peak file"): (RSS_OLD_BYTES, RSS_NEW_BYTES)})
        self.assertEqual(self.metrics(r, "blocking"), set())
        self.assertEqual(self.metrics(r, "informational"), set())

    def test_gate_metric_absent_is_reported(self):
        """A gate header no report emits would leave the gate silently dead."""
        r = self.diff({("point lookup", "warm p99"): (SMALL_NS, BIG_NS)})
        self.assertTrue(r.had_baseline)
        self.assertFalse(r.gate_metric_seen)

    def test_gate_metric_present_without_regression_is_seen(self):
        r = self.diff({("point lookup", "warm p90"): (SMALL_NS, SMALL_NS)})
        self.assertTrue(r.gate_metric_seen)


if __name__ == "__main__":
    unittest.main()
