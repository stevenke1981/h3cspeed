#!/usr/bin/env python3
"""End-to-end synthetic process test for the isolated PERF-002C adapter."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
import zlib


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_png(path: Path) -> None:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + kind + payload +
                struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))
    rows = b"".join(b"\0" + b"\x20\x40\x60" * 864 for _ in range(480))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" +
        chunk(b"IHDR", struct.pack(">IIBBBBB", 864, 480, 8, 2, 0, 0, 0)) +
        chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


def write_fake_engine(path: Path) -> None:
    path.write_text(
        """import json, subprocess, sys
engine, media, scheduler, attention, ffmpeg = sys.argv[1:]
command = [ffmpeg, '-v', 'error', '-y', '-f', 'lavfi', '-i',
 'testsrc2=size=864x480:rate=24', '-f', 'lavfi', '-i',
 'sine=frequency=440:sample_rate=32000', '-t', '0.925', '-frames:v', '22',
 '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-ar', '32000',
 '-ac', '2', '-movflags', '+faststart', media]
raise_code = subprocess.run(command, check=False).returncode
if raise_code: raise SystemExit(raise_code)
with open(scheduler, 'w', encoding='utf-8') as f:
 json.dump({'schema_version':1,'engine':engine,'sampler':'dual_clock_euler',
 'schedule':'native_flow','video_shift':12.0,'audio_shift':3.0,
 'width':864,'height':480,'frames':22,'steps':2,'layers':50,'seed':42,
 'sigma_video':[1.0,0.5,0.0],'sigma_audio':[1.0,0.5,0.0],
 'raw_audio_protocol_verified':True}, f)
with open(attention, 'w', encoding='utf-8') as f:
 json.dump({'schema_version':1,'engine':engine,'requested':'sage',
 'selected':'sage','backend_hits':4,'fallbacks':0}, f)
""", encoding="utf-8")


def process_is_running(pid: int) -> bool:
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist.exe", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, check=False, timeout=10)
        return result.returncode == 0 and f'"{pid}"' in result.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def create_manifest(root: Path, perf002, command_driver: Path) -> tuple[Path, dict]:
    reference = root / "reference.png"
    prompt = root / "prompt.txt"
    write_png(reference)
    prompt.write_text("private fox prompt", encoding="utf-8")
    schedule = {"sampler": "dual_clock_euler", "schedule": "native_flow",
                "video_shift": 12.0, "audio_shift": 3.0}
    labels = {
        "models": ["fl2va", "qwen", "video_vae", "audio_vae"],
        "conditioning": ["token_ids", "token_tags", "qwen_hidden"],
        "engines": ["binary", "source", "runtime"],
    }
    spec = {
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
        "models": {engine: labels["models"] for engine in perf002.ENGINES},
        "conditioning": {engine: labels["conditioning"] for engine in perf002.ENGINES},
        "engines": {"h3cspeed": labels["engines"],
                    "comfyui": ["source", "python_env"]},
        "hardware": {"gpu_uuid": "GPU-test", "gpu_name": "RTX-3070-Ti",
                     "sm": "sm86", "vram_bytes": 8 * 1024**3,
                     "driver": "test", "toolkit": "CUDA-13.2"},
        "trials": {"cold": 1, "warm": 3},
    }
    bindings: dict = {"models": {}, "conditioning": {}, "engines": {}}
    for section in bindings:
        for engine, section_labels in spec[section].items():
            bindings[section][engine] = {}
            for label in section_labels:
                path = root / f"{section}-{engine}-{label}.bin"
                if section == "engines" and engine == "h3cspeed" and label == "binary":
                    path = Path(sys.executable)
                elif section == "engines" and engine == "comfyui" and label == "python_env":
                    path = Path(sys.executable)
                elif section == "engines" and engine == "comfyui" and label == "source":
                    path = command_driver
                else:
                    path.write_bytes(f"{section}:{engine}:{label}".encode())
                bindings[section][engine][label] = str(path)
    manifest = perf002.create_input_manifest(spec, bindings, reference, prompt)
    manifest_path = root / "input-manifest.json"
    perf002.publish_json(manifest_path, manifest)
    config = {"reference_png": str(reference), "prompt_file": str(prompt),
              "bindings": bindings}
    return manifest_path, config


class Perf002SmokeTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                         "ffmpeg and ffprobe are required")
    def test_isolated_process_smoke_binds_evidence_without_claiming_ab(self) -> None:
        perf002 = load_script("run_perf002_ab")
        smoke = load_script("run_perf002_smoke")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child = root / "fake_engine.py"
            write_fake_engine(child)
            manifest_path, config = create_manifest(root, perf002, child)
            engine_dir = root / "engine-output"
            evidence_dir = root / "evidence"
            engine_dir.mkdir()
            evidence_dir.mkdir()
            protected = []
            for name in ("source-root", "comfy-root", "model-root"):
                path = root / name
                path.mkdir()
                protected.append(str(path))
            media = engine_dir / "smoke.mp4"
            scheduler = engine_dir / "scheduler.json"
            attention = engine_dir / "attention.json"
            config.update({
                "schema_version": 1, "engine": "comfyui",
                "argv": [sys.executable, str(child), "comfyui", str(media), str(scheduler),
                         str(attention), shutil.which("ffmpeg") or "ffmpeg"],
                "environment": {"PYTHONUTF8": "1"},
                "output_media": str(media), "scheduler_trace": str(scheduler),
                "attention_trace": str(attention),
                "protected_roots": protected,
                "command_artifacts": {
                    "executable": {"label": "python_env", "argv_index": 0},
                    "driver": {"label": "source", "argv_index": 1},
                },
            })
            config_path = root / "command.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            manifest, _ = smoke.load_input(manifest_path)
            bad_config = dict(config)
            bypass = root / "unbound-fake.py"
            bypass.write_text("raise SystemExit(0)\n", encoding="utf-8")
            bad_config["argv"] = [sys.executable, str(bypass), str(child)]
            bad_config["command_artifacts"] = {
                "executable": {"label": "python_env", "argv_index": 0},
                "driver": {"label": "source", "argv_index": 2},
            }
            with self.assertRaisesRegex(smoke.ContractError, r"argv\[1\]"):
                smoke._validate_command(bad_config, manifest, "comfyui")
            result_path = smoke.run_smoke(
                manifest_path, config_path, evidence_dir,
                shutil.which("ffmpeg") or "ffmpeg",
                shutil.which("ffprobe") or "ffprobe", 60)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "SMOKE_PASS")
            self.assertEqual(result["matched_ab_status"], "NOT_RUN")
            self.assertEqual(result["contract"]["frames"], 22)
            self.assertEqual(result["engine"], "comfyui")
            self.assertEqual(result["attention_evidence"]["backend_hits"], 4)
            serialized = result_path.read_text(encoding="utf-8")
            self.assertNotIn("private fox prompt", serialized)
            self.assertNotIn(str(root), serialized)
            with self.assertRaisesRegex(smoke.ContractError, "outside protected"):
                smoke.run_smoke(
                    manifest_path, config_path, Path(protected[0]),
                    shutil.which("ffmpeg") or "ffmpeg",
                    shutil.which("ffprobe") or "ffprobe", 60)

    def test_attention_fallback_is_rejected(self) -> None:
        smoke = load_script("run_perf002_smoke")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "attention.json"
            path.write_text(json.dumps({
                "schema_version": 1, "engine": "comfyui", "requested": "sage",
                "selected": "sage", "backend_hits": 1, "fallbacks": 1,
            }), encoding="utf-8")
            with self.assertRaisesRegex(smoke.ContractError, "zero fallbacks"):
                smoke._attention_evidence(path, "comfyui")

    def test_timeout_stops_the_isolated_process_group(self) -> None:
        smoke = load_script("run_perf002_smoke")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = (
                "import pathlib,subprocess,sys,time; "
                "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(30)"
            )
            for attempt in range(2):
                with self.subTest(attempt=attempt):
                    log = root / f"timeout-{attempt}.log"
                    child_pid = root / f"child-{attempt}.pid"
                    with self.assertRaisesRegex(smoke.ContractError, "timed out"):
                        smoke._run_engine(
                            [sys.executable, "-c", source, str(child_pid)],
                            log, smoke._private_environment({"PYTHONUTF8": "1"}), 2)
                    self.assertTrue(log.is_file())
                    self.assertTrue(child_pid.is_file())
                    pid = int(child_pid.read_text(encoding="utf-8"))
                    deadline = time.monotonic() + 5
                    while process_is_running(pid) and time.monotonic() < deadline:
                        time.sleep(0.1)
                    self.assertFalse(
                        process_is_running(pid), "child process tree survived timeout")

    @unittest.skipUnless(os.name == "nt", "Job Object failure paths are Windows-only")
    def test_windows_job_assignment_failure_terminates_suspended_process(self) -> None:
        smoke = load_script("run_perf002_smoke")
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "assign-failure.log"
            with mock.patch.object(
                    smoke._WindowsJob, "assign_handle",
                    side_effect=OSError("simulated assignment failure")):
                with self.assertRaisesRegex(OSError, "simulated assignment failure"):
                    smoke._run_engine(
                        [sys.executable, "-c", "import time; time.sleep(30)"],
                        log, smoke._private_environment({"PYTHONUTF8": "1"}), 5)
            self.assertTrue(log.is_file())

    def test_windows_wait_gate_rejects_timeout_or_failure(self) -> None:
        smoke = load_script("run_perf002_smoke")
        smoke._require_windows_process_stopped(0, "unreachable")
        for result in (0x00000102, 0xFFFFFFFF):
            with self.subTest(result=result):
                with self.assertRaisesRegex(smoke.ContractError, "could not be stopped"):
                    smoke._require_windows_process_stopped(
                        result, "process could not be stopped")



if __name__ == "__main__":
    unittest.main()
