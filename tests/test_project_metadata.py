#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import importlib.util
import re
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def load_package_module():
    spec = importlib.util.spec_from_file_location(
        "h3cspeed_package_modes", ROOT / "scripts/package.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProjectMetadataTest(unittest.TestCase):
    def test_script_entrypoints_are_pinned_to_lf(self) -> None:
        attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("*.sh text eol=lf", attributes)
        self.assertIn("*.py text eol=lf", attributes)

    def test_native_build_rejects_shell_metacharacters_in_architectures(self) -> None:
        script = (ROOT / "scripts/build-native.ps1").read_text(encoding="utf-8")
        self.assertIn("^[0-9]+(;[0-9]+)*$", script)
        self.assertIn("CudaArchitectures must be", script)

    def test_github_actions_are_pinned_to_immutable_commits(self) -> None:
        workflow = (ROOT / ".github/workflows/overlay.yml").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("actions/checkout@v4", workflow)
        self.assertRegex(workflow, r"actions/checkout@[0-9a-f]{40}\s+# v4")
        self.assertIn("permissions:\n  contents: read", workflow)

    def test_versions_and_upstream_pin_are_consistent(self) -> None:
        version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        chinese = (ROOT / "README.zh-TW.md").read_text(encoding="utf-8")
        bootstrap = (ROOT / "scripts/bootstrap.py").read_text(encoding="utf-8")
        self.assertEqual(version, "0.2.0")
        self.assertIn(f"project(h3cspeed VERSION {version}", cmake)
        self.assertIn(f"h3cspeed-v{version}.zip", readme)
        self.assertIn(version, chinese)
        pin = "8974cc055ea9c02fcd14cc27dfda3e1027c05153"
        self.assertIn(pin, readme)
        self.assertIn(f'UPSTREAM_COMMIT = "{pin}"', bootstrap)

        yyjson_pin = "9ddba001a4ea88e93b46932e5c5b87b222e19a5f"
        self.assertIn(f"GIT_TAG {yyjson_pin}", cmake)
        self.assertIn("GIT_SHALLOW FALSE", cmake)
        self.assertNotIn("GIT_TAG 0.10.0", cmake)

        model_pin = "939557dc319dd91227e30195a763f272ba7f8765"
        downloader = (ROOT / "scripts/download_h3_fl2va.py").read_text(
            encoding="utf-8")
        provenance = (ROOT / "PROVENANCE.md").read_text(encoding="utf-8")
        self.assertIn(f'REVISION = "{model_pin}"', downloader)
        self.assertIn(model_pin, provenance)
        self.assertIn("does not assert a model license", provenance)

    def test_source_package_excludes_generated_prefixes_and_binaries(self) -> None:
        package_source = (ROOT / "scripts/package.py").read_text(
            encoding="utf-8")
        self.assertIn("EXCLUDED_DIR_PREFIXES", package_source)
        self.assertIn(".pdb", package_source)
        self.assertIn(".safetensors", package_source)
        self.assertIn(".netrc", package_source)
        self.assertIn("credentials.json", package_source)
        self.assertIn(".p12", package_source)

    def test_source_archive_preserves_script_execute_bits_on_windows(self) -> None:
        package = load_package_module()
        with tempfile.TemporaryDirectory(prefix="h3cspeed-zip-mode-") as temporary:
            root = Path(temporary) / "h3cspeed"
            script = root / "scripts" / "probe.sh"
            script.parent.mkdir(parents=True)
            script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            archive = Path(temporary) / "source.zip"
            package.write_zip(root, archive, 1786476000)
            with zipfile.ZipFile(archive) as bundle:
                info = bundle.getinfo("h3cspeed/scripts/probe.sh")
                mode = info.external_attr >> 16
            self.assertEqual(info.create_system, 3)
            self.assertEqual(mode & 0o111, 0o111)

    def test_model_downloader_has_a_platform_safe_default(self) -> None:
        downloader = (ROOT / "scripts/download_h3_fl2va.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('if os.name == "nt"', downloader)
        self.assertIn('"h3cspeed" / "models"', downloader)
        self.assertIn("XDG_CACHE_HOME", downloader)

    def test_backend_symbol_snapshot_is_complete_and_unique(self) -> None:
        symbols = [line.strip() for line in
                   (ROOT / "tests/backend_api_symbols.txt")
                   .read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(symbols), 103)
        self.assertEqual(len(set(symbols)), 103)
        self.assertTrue(all(re.fullmatch(r"h3_gpu_[A-Za-z0-9_]+", item)
                            for item in symbols))

    def test_build_does_not_compile_apple_backend(self) -> None:
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        for forbidden in ("h3_gpu.m", "h3_metal.m", "h3_tokenizer.m",
                          "-framework Metal", "MetalPerformanceShaders"):
            self.assertNotIn(forbidden, cmake)
        self.assertIn("src/h3_gpu_cuda.cu", cmake)
        self.assertIn("src/h3_offload_policy.c", cmake)
        self.assertIn("H3CSPEED_FAST_MATH", cmake)
        self.assertIn("OFF)", cmake)

    def test_documentation_matches_attention_and_offload_policy(self) -> None:
        architecture = (ROOT / "docs/ARCHITECTURE.md").read_text(encoding="utf-8")
        coverage = (ROOT / "docs/BACKEND_COVERAGE.md").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("one-pass", architecture)
        self.assertIn("one-pass", coverage)
        self.assertNotIn("two-pass", architecture)
        self.assertNotIn("two-pass", coverage)
        self.assertIn("last-use", architecture)
        self.assertIn("three-tier", architecture)
        self.assertIn("12-tap polyphase FIR", coverage)
        self.assertIn("system-RAM", readme)
        self.assertIn("generated INT8", readme)
        self.assertNotIn("FIR filters not yet applied", coverage)


if __name__ == "__main__":
    unittest.main()
