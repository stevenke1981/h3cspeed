"""Static safety checks for the Windows Comfy-conditioned quantized wrapper.

These tests intentionally do not start ComfyUI, CUDA, or h3cspeed.  They keep
the launch contract reviewable on a CPU-only checkout; the real 4-step and
20-step gates remain hardware acceptance tests.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "run-h3-quantized.ps1"


class QuantizedWrapperStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = WRAPPER.read_text(encoding="utf-8")

    def test_required_parameters_and_safe_defaults(self) -> None:
        for name in ("ModelRoot", "ComfyUIRoot", "TextEncoder", "Prompt", "Output"):
            self.assertRegex(
                self.source,
                rf"\[Parameter\(Mandatory\s*=\s*\$true\)\]\s*\[string\]\${name}",
            )
        self.assertRegex(self.source, r"\[int\]\$Steps\s*=\s*20")
        self.assertRegex(self.source, r"\[int\]\$Width\s*=\s*256")
        self.assertRegex(self.source, r"\[int\]\$Height\s*=\s*256")
        self.assertRegex(self.source, r"\[int\]\$Frames\s*=\s*22")

    def test_conditioning_bridge_and_hash_validation_are_explicit(self) -> None:
        self.assertIn("encode_h3_quantized_prompt.py", self.source)
        self.assertIn('"--comfyui", $comfyRoot', self.source)
        self.assertIn('"--text-encoder", $encoder', self.source)
        self.assertIn('"--output", $sidecar', self.source)
        self.assertIn('"--prompt", $Prompt', self.source)
        self.assertIn("model_sha256=([0-9a-fA-F]{64})", self.source)
        self.assertIn("Get-FileHash", self.source)
        self.assertIn("H3CSPEED_TEXT_EMBEDDING", self.source)
        self.assertIn("H3CSPEED_TEXT_ENCODER_SHA256", self.source)
        self.assertIn("finally", self.source)

    def test_fail_closed_cuda_and_argument_array_launch(self) -> None:
        self.assertIn('Set-StrictMode -Version Latest', self.source)
        self.assertIn('$ErrorActionPreference = "Stop"', self.source)
        self.assertIn('[Console]::Error.WriteLine', self.source)
        self.assertIn('StartsWith("cuda"', self.source)
        self.assertIn("CPU fallback is forbidden", self.source)
        self.assertNotIn("Invoke-Expression", self.source)
        self.assertNotIn("Start-Process", self.source)
        self.assertRegex(self.source, r"&\s*\$python\s+@helperArguments")
        self.assertRegex(self.source, r"&\s*\$binary\s+@cliArguments")

    def test_no_machine_specific_path_defaults(self) -> None:
        # Relative discovery from the supplied ComfyUI/repository roots is
        # allowed; user/model paths must never be embedded in the wrapper.
        self.assertNotRegex(self.source, re.compile(r"[A-Za-z]:\\(?:Users|models|minimax-h3)", re.I))
        self.assertNotIn("/home/", self.source.lower())


if __name__ == "__main__":
    unittest.main()
