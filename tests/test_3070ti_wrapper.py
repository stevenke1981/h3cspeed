#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/run-3070ti-8gb.sh"


@unittest.skipUnless(os.name != "nt", "the launcher is a POSIX shell script")
class Rtx3070TiWrapperTest(unittest.TestCase):
    def run_wrapper(self, arguments: list[str], *, check: bool = True) -> dict[str, object]:
        with tempfile.TemporaryDirectory(prefix="h3cspeed-wrapper-") as temporary:
            directory = Path(temporary)
            wrapper = directory / "h3cspeed-3070ti-8gb"
            binary = directory / "h3cspeed"
            capture = directory / "capture.json"
            shutil.copy2(WRAPPER, wrapper)
            wrapper.chmod(0o755)
            binary.write_text(
                """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
Path(os.environ['H3CSPEED_TEST_CAPTURE']).write_text(json.dumps({
    'args': sys.argv[1:],
    'env': {name: os.environ.get(name) for name in (
        'H3_CUDA_LOW_VRAM', 'H3_CUDA_OFFLOAD',
        'H3_CUDA_VRAM_BUDGET_MIB', 'H3_CUDA_WEIGHT_CACHE_MIB',
        'H3_CUDA_PINNED_HOST_MIB', 'H3_CUDA_STAGING_MIB',
        'H3_CUDA_RELEASE_SCRATCH', 'H3_PROFILE')},
}), encoding='utf-8')
""",
                encoding="utf-8",
            )
            binary.chmod(0o755)
            environment = dict(os.environ)
            environment["H3CSPEED_TEST_CAPTURE"] = str(capture)
            completed = subprocess.run(
                [str(wrapper), *arguments],
                check=check,
                env=environment,
                capture_output=True,
                text=True,
            )
            if not capture.exists():
                return {
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            return json.loads(capture.read_text(encoding="utf-8"))

    def test_low_vram_defaults_preserve_upstream_quality_settings(self) -> None:
        result = self.run_wrapper(["-d", "model", "-p", "prompt"])
        arguments = result["args"]
        self.assertIsInstance(arguments, list)
        joined = " ".join(arguments)
        for expected in (
            "--ssd-streaming", "--frames 22",
            "--width 256", "--height 256",
        ):
            self.assertIn(expected, joined)
        self.assertNotIn("--reuse", arguments)
        self.assertNotIn("--core-reuse", arguments)
        self.assertNotIn("--token-reduction", arguments)
        self.assertNotIn("--layers", arguments)
        self.assertNotIn("--render-width", arguments)
        self.assertNotIn("--render-height", arguments)
        environment = result["env"]
        self.assertEqual(environment["H3_CUDA_OFFLOAD"], "ram+file")
        self.assertEqual(environment["H3_CUDA_VRAM_BUDGET_MIB"], "5888")
        self.assertEqual(environment["H3_CUDA_WEIGHT_CACHE_MIB"], "1536")
        self.assertEqual(environment["H3_CUDA_PINNED_HOST_MIB"], "128")

    def test_explicit_capacity_values_are_not_duplicated(self) -> None:
        supplied = [
            "-d", "model", "-p", "prompt",
            "--width", "576", "--height", "320",
            "--render-width", "288", "--render-height", "160",
            "--frames", "39", "--layers", "35",
            "--core-reuse", "6",
            "--ssd-streaming", "--token-reduction",
        ]
        result = self.run_wrapper(supplied)
        arguments = result["args"]
        self.assertEqual(arguments.count("--frames"), 1)
        self.assertEqual(arguments.count("--render-width"), 1)
        self.assertEqual(arguments.count("--render-height"), 1)
        self.assertEqual(arguments.count("--layers"), 1)
        self.assertEqual(arguments.count("--reuse"), 0)
        self.assertEqual(arguments.count("--core-reuse"), 1)
        self.assertEqual(arguments.count("--ssd-streaming"), 1)
        self.assertEqual(arguments.count("--token-reduction"), 1)

    def test_explicit_denoiser_reuse_does_not_add_core_reuse(self) -> None:
        result = self.run_wrapper(["-d", "model", "-p", "prompt", "--reuse", "2"])
        arguments = result["args"]
        self.assertEqual(arguments.count("--reuse"), 1)
        self.assertEqual(arguments.count("--core-reuse"), 0)

    def test_conflicting_explicit_reuse_modes_fail_early(self) -> None:
        result = self.run_wrapper(
            ["-d", "model", "-p", "prompt", "--reuse", "2", "--core-reuse", "4"],
            check=False,
        )
        self.assertEqual(result["returncode"], 2)
        self.assertIn("cannot be combined", result["stderr"])

    def test_explicit_one_each_is_allowed(self) -> None:
        result = self.run_wrapper(
            ["-d", "model", "-p", "prompt", "--reuse", "1", "--core-reuse", "1"]
        )
        arguments = result["args"]
        self.assertEqual(arguments.count("--reuse"), 1)
        self.assertEqual(arguments.count("--core-reuse"), 1)


class FastQualityProfileStaticTest(unittest.TestCase):
    def test_preset_locks_quality_shape_and_offload(self) -> None:
        script = (ROOT / "scripts/fast-quality-3070ti-8gb.sh").read_text(
            encoding="utf-8"
        )
        profile = (ROOT / "profiles/rtx3070ti-fast-quality.env").read_text(
            encoding="utf-8"
        )
        for value in (
            "--width 864 --height 480",
            "--render-width 288 --render-height 160",
            "--seconds 5 --steps 20",
            "--seed 42",
            "--layers 50 --reuse 1 --core-reuse 4",
            "--ssd-streaming",
            "H3_CUDA_OFFLOAD=ram+file",
            "H3_CUDA_VRAM_BUDGET_MIB=5888",
            "H3_CUDA_WEIGHT_CACHE_MIB=1536",
            "H3_FAST_QUALITY_MODEL_DIR",
            "H3_FAST_QUALITY_PROMPT",
            "H3_FAST_QUALITY_OUTPUT",
            "124 frames",
            "$PWD/outputs/3070ti-fast-quality.mp4",
        ):
            self.assertIn(value, script + "\n" + profile)
        self.assertNotIn("--token-reduction", script)
        self.assertNotIn("--frames", script)
        self.assertIn("--reuse 1", script)
        self.assertNotIn("--reuse 2", script)
        self.assertIn("cannot prevent SSD stream-slot rereads", profile)
        self.assertIn('"$SCRIPT_DIR/h3cspeed.exe"', (
            ROOT / "scripts/run-3070ti-8gb.sh"
        ).read_text(encoding="utf-8"))
        self.assertIn('"$ROOT/bin/h3cspeed"', (
            ROOT / "scripts/run-3070ti-8gb.sh"
        ).read_text(encoding="utf-8"))

    def test_bilingual_docs_describe_preset_and_stream_reread_boundary(self) -> None:
        english = (ROOT / "README.md").read_text(encoding="utf-8")
        chinese = (ROOT / "README.zh-TW.md").read_text(encoding="utf-8")
        for document in (english, chinese):
            self.assertIn("fast-quality-3070ti-8gb.sh", document)
            self.assertIn("core-reuse", document)
            self.assertIn("H3_CUDA_HOST_CACHE_MIB", document)
            self.assertIn("480p", document)
            self.assertIn("288", document)
            self.assertIn("124", document)
        self.assertIn("cannot prevent SSD stream-slot rereads", english)
        self.assertIn("不能避免", chinese)


@unittest.skipUnless(os.name != "nt" and shutil.which("bash"),
                     "the launcher is a POSIX shell script")
class FastQualityProfileLauncherTest(unittest.TestCase):
    def test_positional_overrides_reach_runner_without_quality_drift(self) -> None:
        with tempfile.TemporaryDirectory(prefix="h3cspeed-fast-quality-") as temporary:
            root = Path(temporary)
            scripts = root / "scripts"
            scripts.mkdir()
            script = scripts / "fast-quality-3070ti-8gb.sh"
            runner = scripts / "run-3070ti-8gb.sh"
            binary = scripts / "h3cspeed"
            capture = root / "capture.json"
            shutil.copy2(ROOT / "scripts/fast-quality-3070ti-8gb.sh", script)
            shutil.copy2(ROOT / "scripts/run-3070ti-8gb.sh", runner)
            script.chmod(0o755)
            runner.chmod(0o755)
            binary.write_text(
                """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
Path(os.environ['H3CSPEED_TEST_CAPTURE']).write_text(json.dumps({
    'args': sys.argv[1:],
    'env': {name: os.environ.get(name) for name in (
        'H3_CUDA_LOW_VRAM', 'H3_CUDA_OFFLOAD',
        'H3_CUDA_VRAM_BUDGET_MIB', 'H3_CUDA_WEIGHT_CACHE_MIB',
        'H3_CUDA_PINNED_HOST_MIB', 'H3_CUDA_STAGING_MIB',
        'H3_CUDA_RELEASE_SCRATCH', 'H3_PROFILE')},
}), encoding='utf-8')
""",
                encoding="utf-8",
            )
            binary.chmod(0o755)
            environment = dict(os.environ)
            environment["H3CSPEED_TEST_CAPTURE"] = str(capture)
            completed = subprocess.run(
                ["bash", str(script), "model-A", "prompt with spaces", "out/A.mp4"],
                check=False,
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(capture.read_text(encoding="utf-8"))
            arguments = result["args"]
            self.assertEqual(arguments[0:6],
                             ["-d", "model-A", "-p", "prompt with spaces",
                              "-o", "out/A.mp4"])
            self.assertIn("--width", arguments)
            self.assertEqual(arguments[arguments.index("--width") + 1], "864")
            self.assertEqual(arguments[arguments.index("--height") + 1], "480")
            self.assertEqual(arguments[arguments.index("--render-width") + 1], "288")
            self.assertEqual(arguments[arguments.index("--render-height") + 1], "160")
            self.assertEqual(arguments[arguments.index("--seconds") + 1], "5")
            self.assertEqual(arguments[arguments.index("--steps") + 1], "20")
            self.assertEqual(arguments[arguments.index("--seed") + 1], "42")
            self.assertEqual(arguments[arguments.index("--layers") + 1], "50")
            self.assertEqual(arguments[arguments.index("--reuse") + 1], "1")
            self.assertEqual(arguments[arguments.index("--core-reuse") + 1], "4")
            self.assertNotIn("--token-reduction", arguments)
            self.assertEqual(arguments.count("--ssd-streaming"), 1)
            self.assertEqual(result["env"]["H3_CUDA_OFFLOAD"], "ram+file")

    def test_environment_overrides_work_without_positional_arguments(self) -> None:
        with tempfile.TemporaryDirectory(prefix="h3cspeed-fast-quality-env-") as temporary:
            root = Path(temporary)
            scripts = root / "scripts"
            scripts.mkdir()
            script = scripts / "fast-quality-3070ti-8gb.sh"
            runner = scripts / "run-3070ti-8gb.sh"
            binary = scripts / "h3cspeed"
            capture = root / "capture.json"
            shutil.copy2(ROOT / "scripts/fast-quality-3070ti-8gb.sh", script)
            shutil.copy2(ROOT / "scripts/run-3070ti-8gb.sh", runner)
            script.chmod(0o755)
            runner.chmod(0o755)
            binary.write_text(
                "#!/usr/bin/env python3\n"
                "import os, sys\n"
                "from pathlib import Path\n"
                "Path(os.environ['H3CSPEED_TEST_CAPTURE']).write_text(' '.join(sys.argv[1:]), encoding='utf-8')\n",
                encoding="utf-8",
            )
            binary.chmod(0o755)
            environment = dict(os.environ)
            environment.update({
                "H3CSPEED_TEST_CAPTURE": str(capture),
                "H3_FAST_QUALITY_MODEL_DIR": "model-env",
                "H3_FAST_QUALITY_PROMPT": "prompt-env",
                "H3_FAST_QUALITY_OUTPUT": "out/env.mp4",
            })
            completed = subprocess.run(
                ["bash", str(script)],
                check=False,
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            captured = capture.read_text(encoding="utf-8")
            self.assertIn("-d model-env", captured)
            self.assertIn("-p prompt-env", captured)
            self.assertIn("-o out/env.mp4", captured)


if __name__ == "__main__":
    unittest.main()
