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
        self.assertRegex(self.source, r"\[int\]\$RenderWidth\s*=\s*0")
        self.assertRegex(self.source, r"\[int\]\$RenderHeight\s*=\s*0")
        self.assertRegex(self.source, r"\[int\]\$Frames\s*=\s*22")
        self.assertRegex(self.source, r"\[UInt64\]\$Seed\s*=\s*42")

    def test_internal_render_geometry_and_seed_are_forwarded(self) -> None:
        self.assertIn("RenderWidth and RenderHeight must be supplied together", self.source)
        self.assertIn("Render dimensions must preserve the output aspect ratio", self.source)
        self.assertIn("Width and Height must be divisible by 32", self.source)
        self.assertIn("5 + 17n", self.source)
        self.assertIn("Reuse must be in 1..3", self.source)
        self.assertIn('"--width", $effectiveRenderWidth', self.source)
        self.assertIn('"--height", $effectiveRenderHeight', self.source)
        self.assertIn('"--render-width", $RenderWidth', self.source)
        self.assertIn('"--render-height", $RenderHeight', self.source)
        self.assertIn('"--seed", $Seed', self.source)
        self.assertIn('$env:H3_CUDA_DEVICE', self.source)
        self.assertIn('Restore-ProcessEnvironmentVariable "H3_CUDA_DEVICE"', self.source)
        self.assertIn('$env:H3_CUDA_ASYNC_REFILL = "1"', self.source)
        self.assertIn('$env:H3_CUDA_DIT_PREFETCH = "1"', self.source)
        self.assertIn('$env:H3_CUDA_ATTENTION = "sage"', self.source)
        self.assertIn('Restore-ProcessEnvironmentVariable "H3_CUDA_ASYNC_REFILL"', self.source)
        self.assertIn('Restore-ProcessEnvironmentVariable "H3_CUDA_DIT_PREFETCH"', self.source)
        self.assertIn('Restore-ProcessEnvironmentVariable "H3_CUDA_ATTENTION"', self.source)

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

    def test_i2v_keyframe_contract_is_explicit(self) -> None:
        for flag in ("FirstFrame", "LastFrame"):
            self.assertIn(f"${flag}", self.source)
        self.assertIn('"--mode", $mode', self.source)
        self.assertIn('"--first-frame", $canonicalFirstFrame', self.source)
        self.assertIn('$null -ne $firstFrameSource', self.source)
        self.assertIn('"--last-frame", $canonicalLastFrame', self.source)
        self.assertIn('"fl2va_i2v_first_frame"', self.source)
        self.assertIn('"fl2va_i2v_last_frame"', self.source)
        self.assertIn('"fl2va_i2v_first_and_last_frames"', self.source)
        self.assertIn("Quantized FL2VA manifest does not declare", self.source)
        self.assertIn('"--first-frame", $canonicalFirstFrame', self.source)
        self.assertIn("I2V width and height must both be at least 64", self.source)
        self.assertIn("sidecar path must end in .h3c", self.source)
        self.assertNotIn('$mode -eq "fl2va-i2v" -and\n        -not $sidecar.EndsWith', self.source)

    def test_fail_closed_cuda_and_argument_array_launch(self) -> None:
        self.assertIn('Set-StrictMode -Version Latest', self.source)
        self.assertIn('$ErrorActionPreference = "Stop"', self.source)
        self.assertIn('[Console]::Error.WriteLine', self.source)
        self.assertIn("^cuda(?::[0-9]+)?$", self.source)
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

    def test_extracted_runtime_layout_is_auto_discovered(self) -> None:
        self.assertIn('(Join-Path $RepositoryRoot "bin\\h3cspeed.exe")', self.source)


if __name__ == "__main__":
    unittest.main()
