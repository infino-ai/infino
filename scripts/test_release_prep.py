"""Unit tests for release_prep.py. Run from the repo root:

  python3 -m unittest discover -s scripts -q
"""

import json
import tempfile
import unittest
from pathlib import Path

from check_version_sync import check, locked_version, manifest_version
from release_prep import ReleasePrepError, prepare
from test_check_version_sync import PLATFORM_PACKAGES, write_fixture


class Prepare(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        # crate 0.3.4, node 0.3.2, python 0.3.5 — the worked example of
        # legitimate patch divergence from docs/versioning.md.
        write_fixture(self.root)

    def node_manifest(self):
        return json.loads((self.root / "infino-node" / "package.json").read_text())

    def test_crate_patch_stamps_only_the_root_manifest_and_lock(self):
        changed = prepare(self.root, "crate", "0.3.5")
        self.assertEqual(sorted(changed), ["Cargo.lock", "Cargo.toml"])
        self.assertEqual(manifest_version(self.root / "Cargo.toml"), "0.3.5")
        self.assertEqual(locked_version(self.root / "Cargo.lock", "infino"), "0.3.5")
        self.assertEqual(self.node_manifest()["version"], "0.3.2")
        self.assertEqual(
            manifest_version(self.root / "infino-python" / "Cargo.toml"), "0.3.5"
        )
        self.assertEqual(check(self.root), [])

    def test_node_patch_stamps_package_json_pins_and_node_crate(self):
        changed = prepare(self.root, "node", "0.3.3")
        self.assertEqual(
            sorted(changed),
            ["infino-node/Cargo.lock", "infino-node/Cargo.toml",
             "infino-node/package.json"],
        )
        manifest = self.node_manifest()
        self.assertEqual(manifest["version"], "0.3.3")
        for pin in PLATFORM_PACKAGES:
            self.assertEqual(manifest["optionalDependencies"][pin], "0.3.3")
        self.assertEqual(
            manifest_version(self.root / "infino-node" / "Cargo.toml"), "0.3.3"
        )
        self.assertEqual(
            locked_version(self.root / "infino-node" / "Cargo.lock", "infino-node"),
            "0.3.3",
        )
        self.assertEqual(manifest_version(self.root / "Cargo.toml"), "0.3.4")
        self.assertEqual(check(self.root), [])

    def test_python_patch_stamps_only_the_python_manifest_and_lock(self):
        changed = prepare(self.root, "python", "0.3.6")
        self.assertEqual(
            sorted(changed),
            ["infino-python/Cargo.lock", "infino-python/Cargo.toml"],
        )
        self.assertEqual(
            manifest_version(self.root / "infino-python" / "Cargo.toml"), "0.3.6"
        )
        self.assertEqual(
            locked_version(
                self.root / "infino-python" / "Cargo.lock", "infino-python"
            ),
            "0.3.6",
        )
        self.assertEqual(check(self.root), [])

    def test_coordinated_release_realigns_all_three_packages(self):
        changed = prepare(self.root, "all", "0.4.0")
        self.assertEqual(len(changed), 7)
        self.assertEqual(manifest_version(self.root / "Cargo.toml"), "0.4.0")
        self.assertEqual(self.node_manifest()["version"], "0.4.0")
        self.assertEqual(
            manifest_version(self.root / "infino-python" / "Cargo.toml"), "0.4.0"
        )
        self.assertEqual(check(self.root), [])

    def test_rejects_coordinated_release_with_nonzero_patch(self):
        with self.assertRaisesRegex(ReleasePrepError, "patch"):
            prepare(self.root, "all", "0.4.1")

    def test_rejects_single_package_bump_off_the_crate_line(self):
        with self.assertRaisesRegex(ReleasePrepError, "0.3"):
            prepare(self.root, "node", "0.4.0")

    def test_rejects_version_not_above_the_current_one(self):
        with self.assertRaisesRegex(ReleasePrepError, "0.3.2"):
            prepare(self.root, "node", "0.3.2")
        with self.assertRaisesRegex(ReleasePrepError, "0.3.4"):
            prepare(self.root, "crate", "0.3.3")

    def test_rejects_coordinated_version_that_moves_a_package_backwards(self):
        # Every package sits above 0.3.0 (python at 0.3.5); a "coordinated"
        # 0.3.0 would move them backwards (and registries are immutable).
        with self.assertRaisesRegex(ReleasePrepError, "does not move"):
            prepare(self.root, "all", "0.3.0")

    def test_each_bumps_every_package_to_its_own_next_patch(self):
        # crate 0.3.4 / node 0.3.2 / python 0.3.5 move independently —
        # divergent patch counters stay divergent, everyone steps forward.
        changed = prepare(self.root, "each")
        self.assertEqual(len(changed), 7)
        self.assertEqual(manifest_version(self.root / "Cargo.toml"), "0.3.5")
        manifest = self.node_manifest()
        self.assertEqual(manifest["version"], "0.3.3")
        for pin in PLATFORM_PACKAGES:
            self.assertEqual(manifest["optionalDependencies"][pin], "0.3.3")
        self.assertEqual(
            manifest_version(self.root / "infino-python" / "Cargo.toml"), "0.3.6"
        )
        self.assertEqual(check(self.root), [])

    def test_each_rejects_an_explicit_version(self):
        with self.assertRaisesRegex(ReleasePrepError, "next patch"):
            prepare(self.root, "each", "0.4.0")

    def test_single_package_scope_requires_a_version(self):
        with self.assertRaisesRegex(ReleasePrepError, "required"):
            prepare(self.root, "crate", None)

    def test_rejects_malformed_version(self):
        with self.assertRaisesRegex(ReleasePrepError, "X.Y.Z"):
            prepare(self.root, "crate", "1.2")

    def test_rejected_prepare_leaves_the_tree_untouched(self):
        before = (self.root / "Cargo.toml").read_text()
        with self.assertRaises(ReleasePrepError):
            prepare(self.root, "all", "0.4.1")
        self.assertEqual((self.root / "Cargo.toml").read_text(), before)


if __name__ == "__main__":
    unittest.main()
