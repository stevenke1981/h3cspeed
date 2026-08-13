#!/usr/bin/env python3
"""Real ffmpeg smoke for the PERF-002 media QA harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_perf002_ab.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("perf002_media", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Perf002RunnerTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                         "ffmpeg and ffprobe are required")
    def test_synthetic_124_frame_media_qa_is_harness_only(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            media = root / "synthetic.mp4"
            samples = root / "samples"
            samples.mkdir()
            command = [
                shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=864x480:rate=24",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=32000",
                "-t", "5.175", "-frames:v", "124", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "32000",
                "-ac", "2", "-movflags", "+faststart", str(media),
            ]
            completed = subprocess.run(command, capture_output=True, text=True,
                                       check=False, shell=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            evidence = runner.validate_media(
                shutil.which("ffmpeg") or "ffmpeg",
                shutil.which("ffprobe") or "ffprobe",
                media, samples,
            )
            self.assertTrue(evidence["full_decode"])
            self.assertTrue(evidence["audio"]["non_silent"])
            self.assertEqual(len(evidence["frame_hashes"]), 5)
            self.assertEqual(evidence["visual_review"], "MANUAL_REQUIRED")


if __name__ == "__main__":
    unittest.main()
