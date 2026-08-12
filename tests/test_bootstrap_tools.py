#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "h3cspeed_bootstrap", ROOT / "scripts/bootstrap.py")
assert SPEC and SPEC.loader
BOOTSTRAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOOTSTRAP)


class BootstrapToolTest(unittest.TestCase):
    def test_git_blob_hash_matches_git_empty_blob(self) -> None:
        self.assertEqual(
            BOOTSTRAP.git_blob_sha(b""),
            "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
        )

    def test_cli_name_patch_is_narrow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "main.c").write_text(
                'const char *a="Usage: h3 -d MODEL";\n'
                'const char *b="./h3 --info";\n', encoding="utf-8")
            (root / "h3_cli.c").write_text(
                'const char *a="usage: h3 [flags]";\n', encoding="utf-8")
            BOOTSTRAP.patch_cli_name(root)
            self.assertIn("Usage: h3cspeed", (root / "main.c").read_text())
            self.assertIn("./h3cspeed", (root / "main.c").read_text())
            self.assertIn("usage: h3cspeed", (root / "h3_cli.c").read_text())


if __name__ == "__main__":
    unittest.main()
