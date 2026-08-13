#!/usr/bin/env python3
"""Contract tests for immutable PERF-002 benchmark inputs/results."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_perf002_ab.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("perf002", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def png(path: Path, width: int = 864, height: int = 480,
        filter_byte: int = 0, trailing_stream: bool = False) -> None:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + kind + payload +
                struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))
    rows = b"".join(bytes((filter_byte,)) + b"\x20\x40\x60" * width
                    for _ in range(height))
    compressed = zlib.compress(rows, 9) + (b"trailing" if trailing_stream else b"")
    payload = (b"\x89PNG\r\n\x1a\n" +
               chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
               chunk(b"IDAT", compressed) + chunk(b"IEND", b""))
    path.write_bytes(payload)


def specification() -> dict:
    schedule = {"sampler": "dual_clock_euler", "schedule": "native_flow",
                "video_shift": 12.0, "audio_shift": 3.0}
    return {
        "track": "algorithm_parity",
        "fixture": {"label": "codex-reference.png", "sha256": "0" * 64,
                    "width": 864, "height": 480, "mode": "fl2va_first_frame"},
        "prompt_sha256": "0" * 64,
        "render": {"width": 864, "height": 480, "render_width": 864,
                   "render_height": 480, "frames": 124, "fps": 24,
                   "steps": 8, "layers": 50, "reuse": 1, "core_reuse": 1,
                   "seed": 42},
        "scheduler": {"h3cspeed": schedule, "comfyui": schedule.copy(),
                      "sigma_tolerance": 1e-6},
        "attention": {"requested": "sage", "tf32": False,
                      "backend_hit_required": True, "fallback_allowed": False},
        "models": {"h3cspeed": ["fl2va", "qwen", "video_vae", "audio_vae"],
                   "comfyui": ["fl2va", "qwen", "video_vae", "audio_vae"]},
        "conditioning": {"h3cspeed": ["token_ids", "token_tags", "qwen_hidden"],
                         "comfyui": ["token_ids", "token_tags", "qwen_hidden"]},
        "engines": {"h3cspeed": ["binary", "source", "runtime"],
                    "comfyui": ["source", "python_env"]},
        "hardware": {"gpu_uuid": "GPU-deadbeef", "gpu_name": "RTX-3070-Ti",
                     "sm": "sm86", "vram_bytes": 8 * 1024**3,
                     "driver": "596.36", "toolkit": "CUDA-13.2"},
        "trials": {"cold": 1, "warm": 3},
    }


def binding_files(root: Path, spec: dict) -> dict:
    bindings: dict[str, dict[str, dict[str, str]]] = {
        "models": {}, "conditioning": {}, "engines": {}}
    for section in ("models", "conditioning", "engines"):
        for engine, labels in spec[section].items():
            bindings[section][engine] = {}
            for label in labels:
                path = root / f"{section}-{engine}-{label}.bin"
                path.write_bytes(f"{section}:{engine}:{label}".encode())
                bindings[section][engine][label] = str(path)
    return bindings


class Perf002ManifestTests(unittest.TestCase):
    def test_create_publish_and_load_are_canonical_and_private(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.png"
            prompt = root / "prompt.txt"
            png(reference)
            prompt.write_text("private benchmark prompt", encoding="utf-8")
            spec = specification()
            manifest = runner.create_input_manifest(
                spec, binding_files(root, spec), reference, prompt)
            manifest_path = root / "input-manifest.json"
            digest = runner.publish_json(manifest_path, manifest)
            loaded, loaded_digest = runner.load_input(manifest_path)
            self.assertEqual(digest, loaded_digest)
            self.assertEqual(loaded["fixture"]["sha256"], runner.sha256_file(reference))
            serialized = manifest_path.read_text(encoding="utf-8")
            self.assertNotIn("private benchmark prompt", serialized)
            self.assertNotIn(str(root), serialized)
            self.assertTrue(serialized.endswith("\n"))

    def test_tamper_and_overwrite_fail_closed(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.png"
            prompt = root / "prompt.txt"
            png(reference)
            prompt.write_text("fox", encoding="utf-8")
            spec = specification()
            manifest = runner.create_input_manifest(
                spec, binding_files(root, spec), reference, prompt)
            path = root / "input-manifest.json"
            runner.publish_json(path, manifest)
            with self.assertRaisesRegex(runner.ContractError, "overwrite"):
                runner.publish_json(path, manifest)
            path.write_bytes(path.read_bytes().replace(b'"seed":42', b'"seed":43'))
            with self.assertRaisesRegex(runner.ContractError, "checksum mismatch"):
                runner.load_input(path)

    def test_unmatched_scheduler_and_sensitive_labels_are_rejected(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.png"
            prompt = root / "prompt.txt"
            png(reference)
            prompt.write_text("fox", encoding="utf-8")
            bad = specification()
            bad["scheduler"]["comfyui"]["sampler"] = "res_multistep"
            with self.assertRaisesRegex(runner.ContractError, "identical dual-clock"):
                runner.create_input_manifest(
                    bad, binding_files(root, bad), reference, prompt)

    def test_required_bindings_and_complete_png_are_enforced(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompt = root / "prompt.txt"
            prompt.write_text("fox", encoding="utf-8")
            truncated = root / "truncated.png"
            truncated.write_bytes(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" +
                struct.pack(">II", 864, 480))
            spec = specification()
            with self.assertRaisesRegex(runner.ContractError, "PNG|truncated|incomplete"):
                runner.create_input_manifest(
                    spec, binding_files(root, spec), truncated, prompt)
            invalid_filter = root / "invalid-filter.png"
            png(invalid_filter, filter_byte=5)
            spec = specification()
            with self.assertRaisesRegex(runner.ContractError, "invalid row filter"):
                runner.create_input_manifest(
                    spec, binding_files(root, spec), invalid_filter, prompt)
            trailing = root / "trailing-zlib.png"
            png(trailing, trailing_stream=True)
            spec = specification()
            with self.assertRaisesRegex(runner.ContractError, "incomplete"):
                runner.create_input_manifest(
                    spec, binding_files(root, spec), trailing, prompt)
            reference = root / "reference.png"
            png(reference)
            spec = specification()
            spec["models"]["h3cspeed"] = ["placeholder"]
            with self.assertRaisesRegex(runner.ContractError, "lacks required"):
                runner.create_input_manifest(
                    spec, binding_files(root, spec), reference, prompt)
            bad = specification()
            bad["fixture"]["label"] = "E:/private/reference.png"
            with self.assertRaisesRegex(runner.ContractError, "portable non-sensitive"):
                runner.create_input_manifest(
                    bad, binding_files(root, bad), reference, prompt)

    def test_engine_result_cannot_claim_matched_pass(self) -> None:
        runner = load_runner()
        digest = "a" * 64
        result = {
            "schema_version": 1, "kind": runner.RESULT_KIND,
            "input_manifest_sha256": digest, "engine": "h3cspeed",
            "trial": {"kind": "cold", "index": 0},
            "timings": {stage: 1.0 for stage in runner.STAGES},
            "scheduler_evidence": {
                "sampler": "dual_clock_euler", "schedule": "native_flow",
                "video_shift": 12.0, "audio_shift": 3.0,
                "sigma_video_sha256": "b" * 64, "sigma_audio_sha256": "c" * 64,
                "sigma_max_abs_diff": 0.0, "raw_audio_protocol_verified": True,
            },
            "attention_evidence": {"requested": "sage", "selected": "sage",
                                   "backend_hits": 1, "fallbacks": 0,
                                   "backend_trace_sha256": "e" * 64},
            "media": {"sha256": "d" * 64, "bytes": 1,
                      "video": {"codec": "h264", "width": 864, "height": 480,
                                "fps": "24/1", "frames": 124},
                      "audio": {"codec": "aac", "sample_rate": 32000,
                                "channels": 2, "decoded_pcm_bytes": 4,
                                "non_silent": True},
                      "duration_seconds": 5.175,
                      "frame_hashes": {f"frame-{index:03d}.png": "f" * 64
                                       for index in (1, 32, 63, 94, 124)},
                      "full_decode": True,
                      "visual_review": "MANUAL_REQUIRED"},
            "status": "PASS",
        }
        with self.assertRaisesRegex(runner.ContractError, "only NOT_RUN"):
            runner.validate_result(result, digest)


if __name__ == "__main__":
    unittest.main()
