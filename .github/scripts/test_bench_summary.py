#!/usr/bin/env python3
"""Regression tests for the benchmark merge gate (bench_summary.py).

Run: python3 -m unittest .github/scripts/test_bench_summary.py

The gate must fire only on controllable, low-variance time metrics - warm
latency, count(), ingest Time, and the drain/optimize walls. Cold object-store
timings and memory-footprint columns swing by hundreds of percent run-to-run on
identical code, so gating on them fails PRs that changed no engine code (and one
footprint column, "Peak file", was even being misread as a nanosecond latency).
These tests pin that boundary.
"""

import importlib.util
import json
import os
import tempfile
import unittest

_SPEC = importlib.util.spec_from_file_location(
    "bench_summary", os.path.join(os.path.dirname(__file__), "bench_summary.py")
)
bs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bs)

MS = 1e6  # nanoseconds per millisecond


def _key(label, header):
    # Report keys are "anchor|subtitle|label|header".
    return f"anchor|subtitle|{label}|{header}"


class PredicateTests(unittest.TestCase):
    def test_footprint_columns_are_not_latencies(self):
        for header in ("Peak file", "Peak anon", "Peak RSS", "Median RSS", "Stored"):
            self.assertFalse(bs.is_latency(header), header)

    def test_time_metrics_are_latencies(self):
        for header in ("warm p90", "warm p50", "Time", "count()", "optimize wall"):
            self.assertTrue(bs.is_latency(header), header)

    def test_gate_excludes_cold_and_footprint(self):
        for header in (
            "cold 1st query (median)",
            "cold open (median)",
            "+fetch cold",
            "Peak file",
            "Peak RSS",
        ):
            self.assertFalse(bs.gates(header), header)

    def test_gate_includes_warm_and_walls(self):
        for header in ("warm p90", "warm p50", "Time", "count()", "drain wall"):
            self.assertTrue(bs.gates(header), header)

    def test_footprint_renders_as_bytes_not_ms(self):
        self.assertTrue(bs.human("Peak file", 322.59e6).endswith("MiB"))
        self.assertTrue(bs.human("Peak RSS", 2 * 1073741824).endswith("GiB"))


class BlockingGateTests(unittest.TestCase):
    def _diff(self, base, cur):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "base"))
        os.makedirs(os.path.join(d, "cur"))
        with open(os.path.join(d, "base", "r.json"), "w") as fh:
            json.dump(base, fh)
        with open(os.path.join(d, "cur", "r.json"), "w") as fh:
            json.dump(cur, fh)
        primary = (bs.primary_latency_header_from_gate_metric("p90"), "time", "stored", "wall")
        _, _, blocking, _, _ = bs.diff(
            ["r"], os.path.join(d, "base"), os.path.join(d, "cur"), 5.0, primary
        )
        return {e["metric"] for e in blocking}

    def test_cold_regression_does_not_block(self):
        # The #585 / #586 signature: a huge cold-cache swing on identical code.
        blocked = self._diff(
            {_key("two_term_and", "cold 1st query (median)"): 18.72 * MS},
            {_key("two_term_and", "cold 1st query (median)"): 1260.0 * MS},  # +6640%
        )
        self.assertEqual(blocked, set())

    def test_fetch_cold_regression_does_not_block(self):
        blocked = self._diff(
            {_key("forty_term_or", "+fetch cold"): 189.78 * MS},
            {_key("forty_term_or", "+fetch cold"): 1550.0 * MS},  # +718%
        )
        self.assertEqual(blocked, set())

    def test_peak_footprint_does_not_block(self):
        # 322 MiB -> 524 MiB was blocking as a mislabeled "322 ms -> 523 ms".
        blocked = self._diff(
            {_key("vector-only", "Peak file"): 322.59e6},
            {_key("vector-only", "Peak file"): 523.80e6},
        )
        self.assertEqual(blocked, set())

    def test_real_warm_regression_blocks(self):
        blocked = self._diff(
            {_key("bm25_search", "warm p90"): 10.0 * MS},
            {_key("bm25_search", "warm p90"): 20.0 * MS},  # +100%, +10ms
        )
        self.assertIn("bm25_search / warm p90", blocked)

    def test_ingest_wall_regression_blocks(self):
        blocked = self._diff(
            {_key("ingest", "Time"): 1000.0 * MS},
            {_key("ingest", "Time"): 2000.0 * MS},  # +100%
        )
        self.assertIn("ingest / Time", blocked)

    def test_sub_5ms_warm_move_does_not_block(self):
        # The AND noise guard: +60% but under the 5 ms absolute floor.
        blocked = self._diff(
            {_key("single_common", "warm p90"): 4.0 * MS},
            {_key("single_common", "warm p90"): 6.4 * MS},  # +60%, +2.4ms
        )
        self.assertEqual(blocked, set())


if __name__ == "__main__":
    unittest.main()
