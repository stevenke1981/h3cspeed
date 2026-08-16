#!/usr/bin/env python3
"""Portable contract checks for the resumable 60-second runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_h3_quantized_60s.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("h3cspeed_60s", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Quantized60SecondRunnerTests(unittest.TestCase):
    def test_defaults_are_exactly_sixty_seconds(self) -> None:
        runner = load_runner()
        parser = runner.build_parser()
        args = parser.parse_args([
            "--model-root", "model", "--comfyui", "comfy",
            "--text-encoder", "qwen.safetensors", "--prompt", "fox",
            "--output-dir", "output",
        ])
        self.assertEqual(args.segments, 12)
        self.assertEqual(args.segment_frames, 124)
        self.assertEqual(args.segments * args.segment_frames, 1488)
        self.assertEqual(args.steps, 20)
        self.assertEqual((args.width, args.height), (864, 480))
        self.assertEqual((args.render_width, args.render_height), (288, 160))
        runner.validate_args(args)

    def test_invalid_frame_math_fails_closed(self) -> None:
        runner = load_runner()
        parser = runner.build_parser()
        base = [
            "--model-root", "model", "--comfyui", "comfy",
            "--text-encoder", "qwen.safetensors", "--prompt", "fox",
            "--output-dir", "output",
        ]
        with self.assertRaisesRegex(runner.RunError, r"5 \+ 17n"):
            runner.validate_args(parser.parse_args(base + ["--segment-frames", "91"]))
        with self.assertRaisesRegex(runner.RunError, "must be at least 1440"):
            runner.validate_args(parser.parse_args(base + ["--segments", "11"]))
        with self.assertRaisesRegex(runner.RunError, "exceeds uint64"):
            runner.validate_args(parser.parse_args(base + [
                "--base-seed", str((1 << 64) - 2),
            ]))

    def test_resume_state_is_bound_to_the_full_configuration(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            state = runner.load_or_create_state(path, {"prompt_sha256": "a", "steps": 20})
            self.assertEqual(state["config"]["steps"], 20)
            with self.assertRaisesRegex(runner.RunError, "does not match"):
                runner.load_or_create_state(path, {"prompt_sha256": "b", "steps": 20})

    def test_runtime_identity_and_completed_artifacts_are_bound(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        for marker in (
            '"runner_sha256"', '"model_manifest_sha256"', '"comfy_python_sha256"',
            '"model_payload_sha256"', '"runtime_payload_sha256"',
            '"comfy_environment_sha256"',
            '"gpu_identity"',
            '"comfy_sources"', '"ffmpeg_sha256"', '"ffprobe_sha256"',
            'prior.get("sha256")', 'prior.get("sidecar_sha256")',
            'prior.get("input_anchor_sha256")', 'prior.get("output_anchor_sha256")',
            'verify_bound_inputs()',
        ):
            self.assertIn(marker, source)

    def test_lock_is_exclusive_and_reusable(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / ".run.lock"
            with runner.RunLock(lock_path):
                with self.assertRaisesRegex(runner.RunError, "run lock already exists"):
                    with runner.RunLock(lock_path):
                        pass
            self.assertFalse(lock_path.exists())
            with runner.RunLock(lock_path):
                self.assertTrue(lock_path.is_file())
            self.assertFalse(lock_path.exists())

    def test_preexisting_lock_is_refused_without_truncation(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / ".run.lock"
            lock_path.write_text("do not truncate", encoding="utf-8")
            with self.assertRaisesRegex(runner.RunError, "stale lock"):
                with runner.RunLock(lock_path):
                    pass
            self.assertEqual(lock_path.read_text(encoding="utf-8"), "do not truncate")

    def test_model_identity_binds_declared_root_and_payload_bytes(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            pack = Path(temporary)
            model_root = pack / "base"
            encoder = model_root / "FL2VA" / "text_encoder" / "qwen.safetensors"
            config = model_root / "config.json"
            encoder.parent.mkdir(parents=True)
            encoder.write_bytes(b"qwen")
            config.write_text("{}", encoding="utf-8")
            manifest = pack / "manifest.json"
            manifest.write_text(json.dumps({
                "schema_version": 2,
                "kind": "minimax-h3-comfy-fl2va-quantized-pack",
                "model_family": "FL2VA",
                "capabilities": [
                    "t2v", "fl2va_i2v_first_frame", "fl2va_i2v_last_frame",
                    "fl2va_i2v_first_and_last_frames",
                ],
                "conditioning_sidecar_versions": [1, 2],
                "model_root_relative": "base",
                "large_payloads": [{
                    "path": "base/FL2VA/text_encoder/qwen.safetensors",
                    "bytes": 4,
                    "role": "qwen_nvfp4",
                }],
                "small_configs": [{
                    "path": "base/config.json",
                    "bytes": 2,
                    "role": "config",
                }],
            }), encoding="utf-8")
            encoder_hash = runner.sha256_file(encoder)
            fingerprints = runner.model_payload_fingerprints(
                manifest, model_root, encoder, encoder_hash
            )
            self.assertEqual(
                fingerprints["base/FL2VA/text_encoder/qwen.safetensors"], encoder_hash
            )
            with self.assertRaisesRegex(runner.RunError, "does not match"):
                runner.model_payload_fingerprints(
                    manifest, pack / "sibling", encoder, encoder_hash
                )
            manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
            manifest_data["capabilities"].remove("fl2va_i2v_last_frame")
            manifest.write_text(json.dumps(manifest_data), encoding="utf-8")
            with self.assertRaisesRegex(runner.RunError, "lacks I2V capabilities"):
                runner.model_payload_fingerprints(
                    manifest, model_root, encoder, encoder_hash
                )

    def test_runtime_identity_includes_portable_private_payloads(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "bin" / "h3cspeed"
            executable = root / "libexec" / "h3cspeed"
            library = root / "lib" / "libcublas.so"
            for path, payload in ((binary, b"launcher"), (executable, b"elf"),
                                  (library, b"cuda")):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            fingerprints = runner.runtime_fingerprints(binary)
            self.assertEqual(
                set(fingerprints),
                {"bin/h3cspeed", "libexec/h3cspeed", "lib/libcublas.so"},
            )

    def test_comfy_environment_identity_hashes_package_contents(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("metadata.distributions()", source)
        self.assertIn('"sha256": sha256', source)
        self.assertIn("source.read(8 * 1024 * 1024)", source)
        self.assertIn("python_environment_inventory", source)
        self.assertIn("deep_content_check=True", source)
        self.assertIn("model_payload_inventory", source)
        self.assertIn("runtime_inventory", source)

    def test_source_uses_sage_low_vram_and_verification_gates(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        for marker in (
            '"H3_CUDA_ATTENTION": "sage"',
            '"H3_CUDA_DEVICE"',
            '"H3_CUDA_TF32": "0"',
            '"CUDA_DEVICE_ORDER": "PCI_BUS_ID"',
            'environment.pop("CUDA_VISIBLE_DEVICES", None)',
            'not key.upper().startswith(("H3_", "H3CSPEED_"))',
            'run_logged(helper_command, log, env=runtime_env)',
            'environment.pop("PYTHONPATH", None)',
            '"H3_CUDA_LOW_VRAM": "1"',
            '"H3_CUDA_OFFLOAD": "ram+file"',
            '"H3_CUDA_VRAM_BUDGET_MIB": "5888"',
            '"H3_CUDA_WEIGHT_CACHE_MIB": "1536"',
            '"H3_CUDA_ASYNC_REFILL": "1"',
            '"H3_CUDA_DIT_PREFETCH": "1"',
            '"--first-frame"',
            '"-xerror"',
            'trim=end_frame={FINAL_FRAMES}',
            'atrim=end_sample=1920000',
            '"samples_per_channel": expected_values // 2',
            'final audio is silent or non-finite',
        ):
            self.assertIn(marker, source)
        self.assertNotIn("shell=True", source)

    def test_output_must_be_outside_the_portable_tree(self) -> None:
        runner = load_runner()
        self.assertTrue(runner.path_within(ROOT / "generated", ROOT))
        self.assertFalse(runner.path_within(ROOT.parent / "external-output", ROOT))

    def test_output_subdirectories_reject_symlinks(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            link = root / "segments"
            target.mkdir()
            try:
                link.symlink_to(target, target_is_directory=True)
            except OSError:
                self.skipTest("directory symlinks are unavailable on this host")
            with self.assertRaisesRegex(runner.RunError, "non-symlink"):
                runner.safe_directory(link, "run output directory")

    def test_runtime_package_includes_runner(self) -> None:
        package_source = (ROOT / "scripts" / "package_runtime.py").read_text(encoding="utf-8")
        self.assertIn('"scripts/run_h3_quantized_60s.py"', package_source)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                         "FFmpeg tools are unavailable")
    def test_ffmpeg_final_is_exactly_1440_frames_and_1920000_audio_samples(self) -> None:
        runner = load_runner()
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        assert ffmpeg is not None and ffprobe is not None
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segment = root / "segment.mp4"
            subprocess.run([
                ffmpeg, "-v", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=24:duration=5.166666667",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=32000:duration=5.166666667",
                "-frames:v", "124", "-c:v", "libx264", "-preset", "ultrafast",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "32000", "-ac", "2",
                str(segment),
            ], check=True)
            concat = root / "concat.txt"
            concat.write_text(
                "".join(f"file '{runner.concat_path(segment)}'\n" for _ in range(12)),
                encoding="utf-8",
            )
            final = root / "final.mp4"
            subprocess.run(runner.build_final_command(ffmpeg, concat, final), check=True)
            runner.validate_media(ffmpeg, ffprobe, final, 64, 64, 1440,
                                  exact_duration=60.0)
            audio = runner.validate_final_audio(ffmpeg, final)
            self.assertEqual(audio["samples_per_channel"], 1_920_000)
            self.assertGreater(audio["peak_dbfs"], -100.0)


if __name__ == "__main__":
    unittest.main()
