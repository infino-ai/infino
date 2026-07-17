"""Unit tests for check_version_sync.py. Run from the repo root:

  python3 -m unittest discover -s scripts -q
"""

import json
import tempfile
import unittest
from pathlib import Path

from check_version_sync import check

PLATFORM_PACKAGES = (
    "infx-darwin-x64",
    "infx-darwin-arm64",
    "infx-linux-x64-gnu",
    "infx-linux-arm64-gnu",
    "infx-linux-x64-musl",
    "infx-linux-arm64-musl",
)


def write_fixture(root, crate="0.3.4", crate_lock=None, node="0.3.2",
                  pins=None, node_crate=None, node_crate_lock=None,
                  python="0.3.5", python_lock=None):
    """Materialize the minimal manifest/lockfile tree the guard reads."""
    root = Path(root)
    crate_lock = crate_lock if crate_lock is not None else crate
    node_crate = node_crate if node_crate is not None else node
    node_crate_lock = node_crate_lock if node_crate_lock is not None else node_crate
    python_lock = python_lock if python_lock is not None else python
    pins = pins if pins is not None else {p: node for p in PLATFORM_PACKAGES}

    (root / "Cargo.toml").write_text(
        "[package]\n"
        'name = "infino"\n'
        f'version = "{crate}"\n'
        'edition = "2024"\n'
        "\n"
        "[dependencies]\n"
        'arrow = { version = "58" }\n'
    )
    (root / "Cargo.lock").write_text(
        "[[package]]\n"
        'name = "arrow"\n'
        'version = "58.0.0"\n'
        "\n"
        "[[package]]\n"
        'name = "infino"\n'
        f'version = "{crate_lock}"\n'
    )

    node_dir = root / "infino-node"
    node_dir.mkdir()
    (node_dir / "package.json").write_text(json.dumps({
        "name": "@infino-ai/infino",
        "version": node,
        "optionalDependencies": pins,
    }, indent=2) + "\n")
    (node_dir / "Cargo.toml").write_text(
        "[package]\n"
        'name = "infino-node"\n'
        f'version = "{node_crate}"\n'
    )
    (node_dir / "Cargo.lock").write_text(
        "[[package]]\n"
        'name = "infino"\n'
        f'version = "{crate}"\n'
        "\n"
        "[[package]]\n"
        'name = "infino-node"\n'
        f'version = "{node_crate_lock}"\n'
    )

    py_dir = root / "infino-python"
    py_dir.mkdir()
    (py_dir / "Cargo.toml").write_text(
        "[package]\n"
        'name = "infino-python"\n'
        f'version = "{python}"\n'
    )
    (py_dir / "Cargo.lock").write_text(
        "[[package]]\n"
        'name = "infino"\n'
        f'version = "{crate}"\n'
        "\n"
        "[[package]]\n"
        'name = "infino-python"\n'
        f'version = "{python_lock}"\n'
    )
    return root


class CheckVersionSync(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_in_sync_tree_with_divergent_patches_passes(self):
        # Patch numbers legitimately diverge; only major.minor must agree.
        write_fixture(self.root, crate="0.3.4", node="0.3.2", python="0.3.5")
        self.assertEqual(check(self.root), [])

    def test_node_on_a_different_minor_fails(self):
        write_fixture(self.root, crate="0.3.4", node="0.2.9",
                      pins={p: "0.2.9" for p in PLATFORM_PACKAGES})
        errors = check(self.root)
        self.assertEqual(len(errors), 1)
        self.assertIn("infino-node/package.json", errors[0])
        self.assertIn("0.2", errors[0])
        self.assertIn("0.3", errors[0])

    def test_python_on_a_different_minor_fails(self):
        write_fixture(self.root, crate="0.3.4", python="0.4.0",
                      python_lock="0.4.0")
        errors = check(self.root)
        self.assertEqual(len(errors), 1)
        self.assertIn("infino-python/Cargo.toml", errors[0])

    def test_platform_pin_out_of_step_with_package_version_fails(self):
        pins = {p: "0.3.2" for p in PLATFORM_PACKAGES}
        pins["infx-linux-arm64-musl"] = "0.3.1"
        write_fixture(self.root, node="0.3.2", pins=pins)
        errors = check(self.root)
        self.assertEqual(len(errors), 1)
        self.assertIn("infx-linux-arm64-musl", errors[0])
        self.assertIn("0.3.1", errors[0])

    def test_node_crate_manifest_out_of_step_with_package_json_fails(self):
        # infino-node/Cargo.toml is unpublished (`publish = false`) but its
        # version must track package.json so no version file lies.
        write_fixture(self.root, node="0.3.2", node_crate="0.3.1",
                      node_crate_lock="0.3.1")
        errors = check(self.root)
        self.assertEqual(len(errors), 1)
        self.assertIn("infino-node/Cargo.toml", errors[0])
        self.assertIn("0.3.1", errors[0])

    def test_node_crate_lockfile_behind_manifest_fails(self):
        write_fixture(self.root, node="0.3.2", node_crate="0.3.2",
                      node_crate_lock="0.3.1")
        errors = check(self.root)
        self.assertEqual(len(errors), 1)
        self.assertIn("infino-node/Cargo.lock", errors[0])

    def test_crate_lockfile_behind_manifest_fails(self):
        # A version bump that skips Cargo.lock breaks `cargo publish --locked`
        # at release time; catch it on the PR instead.
        write_fixture(self.root, crate="0.3.5", crate_lock="0.3.4")
        errors = check(self.root)
        self.assertEqual(len(errors), 1)
        self.assertIn("Cargo.lock", errors[0])

    def test_python_lockfile_behind_manifest_fails(self):
        write_fixture(self.root, python="0.3.6", python_lock="0.3.5")
        errors = check(self.root)
        self.assertEqual(len(errors), 1)
        self.assertIn("infino-python/Cargo.lock", errors[0])

    def test_all_drift_reported_not_just_first(self):
        write_fixture(self.root, crate="0.4.0", node="0.3.2", python="0.3.5")
        errors = check(self.root)
        self.assertEqual(len(errors), 2)

    def test_release_version_on_the_crate_line_passes(self):
        # A binding patch release (any patch) is fine as long as it sits on
        # the crate's major.minor line.
        write_fixture(self.root, crate="0.3.4")
        self.assertEqual(check(self.root, release_version="0.3.9"), [])

    def test_release_version_off_the_crate_line_fails(self):
        write_fixture(self.root, crate="0.3.4")
        errors = check(self.root, release_version="0.4.0")
        self.assertEqual(len(errors), 1)
        self.assertIn("release version 0.4.0", errors[0])
        self.assertIn("0.3", errors[0])

    def test_missing_version_field_fails_loudly(self):
        write_fixture(self.root)
        (self.root / "infino-python" / "Cargo.toml").write_text(
            "[package]\n"
            'name = "infino-python"\n'
        )
        errors = check(self.root)
        self.assertTrue(any("infino-python/Cargo.toml" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
