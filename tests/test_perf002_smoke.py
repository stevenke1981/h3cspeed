#!/usr/bin/env python3
"""End-to-end synthetic process test for the isolated PERF-002C adapter."""

from __future__ import annotations

import importlib.util
import copy
import hashlib
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
def value(flag):
    return sys.argv[sys.argv.index(flag) + 1]
engine = 'comfyui'
media, scheduler, attention, ffmpeg = (value('--output-media'),
    value('--scheduler-trace'), value('--attention-trace'), value('--ffmpeg'))
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
 'width':864,'height':480,'frames':22,'fps':24,'steps':2,'layers':50,'seed':42,
 'sigma_video':[1.0,0.5,0.0],'sigma_audio':[1.0,0.5,0.0],
 'raw_audio_protocol_verified':True}, f)
with open(attention, 'w', encoding='utf-8') as f:
 json.dump({'schema_version':1,'engine':engine,'requested':'sage',
 'selected':'sage','scope':'dit_bf16','backend_hits':4,
 'expected_native_calls':2,'unexpected_fallbacks':0}, f)
""", encoding="utf-8")


def write_fake_h3_engine(path: Path) -> None:
    path.write_text(
        """import hashlib, json, os, pathlib, shutil, subprocess, sys
def value(flag):
    indexes = [i for i, item in enumerate(sys.argv) if item == flag]
    if len(indexes) != 1: raise SystemExit('duplicate/missing ' + flag)
    return sys.argv[indexes[0] + 1]
model_root = pathlib.Path(value('-d'))
prompt = value('-p')
first = pathlib.Path(value('--first-frame'))
media = value('-o')
if prompt != 'private fox prompt': raise SystemExit('prompt mismatch')
if not first.is_file() or not model_root.is_dir(): raise SystemExit('input missing')
for relative in ('FL2VA/transformer/minimax_h3_fl2va_pruned_int8_convrot.safetensors',
 'FL2VA/text_encoder/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors',
 'FL2VA/video_vae/source/minimax_h3_video_vae_fp16.safetensors',
 'FL2VA/audio_vae/minimax_h3_audio_vae_fp32.safetensors'):
    if not (model_root / relative).is_file(): raise SystemExit('model missing ' + relative)
sidecar = pathlib.Path(os.environ['H3CSPEED_TEXT_EMBEDDING'])
if not sidecar.is_file(): raise SystemExit('sidecar missing')
qwen = model_root / 'FL2VA/text_encoder/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'
if hashlib.sha256(qwen.read_bytes()).hexdigest() != os.environ['H3CSPEED_TEXT_ENCODER_SHA256']:
    raise SystemExit('qwen hash mismatch')
if os.environ.get('H3_VAE_LAYER_MAJOR') not in (None, '1'):
    raise SystemExit('invalid layer-major selection')
ffmpeg = os.environ['H3_FFMPEG']
command = [ffmpeg, '-v', 'error', '-y', '-f', 'lavfi', '-i',
 'testsrc2=size=864x480:rate=24', '-f', 'lavfi', '-i',
 'sine=frequency=440:sample_rate=32000', '-t', '0.925', '-frames:v', '22',
 '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-ar', '32000',
 '-ac', '2', '-movflags', '+faststart', media]
if subprocess.run(command, check=False).returncode: raise SystemExit(1)
with open(os.environ['H3CSPEED_PERF002_SCHEDULER_TRACE'], 'w', encoding='utf-8') as f:
 json.dump({'schema_version':1,'engine':'h3cspeed','sampler':'dual_clock_euler',
 'schedule':'native_flow','video_shift':12.0,'audio_shift':3.0,
 'width':864,'height':480,'frames':22,'steps':2,'layers':50,'seed':42,
 'sigma_video':[1.0,0.5,0.0],'sigma_audio':[1.0,0.5,0.0],
 'raw_audio_protocol_verified':True}, f)
with open(os.environ['H3CSPEED_PERF002_ATTENTION_TRACE'], 'w', encoding='utf-8') as f:
 json.dump({'schema_version':1,'engine':'h3cspeed','requested':'sage',
 'selected':'sage','scope':'dit_bf16','backend_hits':4,
 'expected_native_calls':2,'unexpected_fallbacks':0}, f)
if os.environ.get('H3_VAE_LAYER_MAJOR') == '1':
 print('h3: layer-major video VAE 1 chunks x 8 tiles (8 states, 16224 KiB hidden each)',
       file=sys.stderr)
 print('h3cspeed CUDA [video VAE decoder]: device-live=0.00 MiB peak=2036.43 MiB '
       'resident-weights=0.00 MiB uploads=441/9.03 GiB linear=1176 conv=0 sdpa=288',
       file=sys.stderr)
""", encoding="utf-8")


def write_fake_h3_launcher(path: Path, driver: Path) -> None:
    if os.name == "nt":
        path.write_text(
            f'@echo off\r\n"{sys.executable}" "{driver}" %*\r\n',
            encoding="utf-8",
        )
    else:
        path.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{driver}" "$@"\n',
            encoding="utf-8",
        )
        path.chmod(0o755)


def write_h3_sidecar(path: Path, prompt: str, reference: Path, qwen: Path) -> None:
    recipe = b"h3cspeed-conditioning-v2"
    prompt_bytes = prompt.encode("utf-8")
    token_count = 1
    metadata = hashlib.sha256(reference.read_bytes()).digest()
    reserved = bytearray(24)
    reserved[:6] = bytes((1, 1, 1, 1, 0, 1))
    struct.pack_into("<III", reserved, 8, 864, 480, len(metadata))
    embedding = b"\0" * (token_count * 5120 * 2)
    header = struct.pack(
        "<8sIIQQQIIQQQ32s24s", b"H3CSEV01", 2, 128,
        len(prompt_bytes), token_count, len(recipe), 5120, 1,
        len(embedding), token_count, token_count * 4,
        hashlib.sha256(qwen.read_bytes()).digest(), bytes(reserved),
    )
    path.write_bytes(
        header + prompt_bytes + recipe + metadata +
        struct.pack("<I", 1) + embedding + b"\0"
    )


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


def create_manifest(root: Path, perf002, command_driver: Path,
                    h3_binary: Path | None = None) -> tuple[Path, dict]:
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
    conditioning_labels = {
        "h3cspeed": ["sidecar"],
        "comfyui": labels["conditioning"],
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
        "models": {
            "h3cspeed": labels["models"] + [
                "transformer_config", "tokenizer", "video_vae_config",
                "audio_vae_config",
            ],
            "comfyui": labels["models"],
        },
        "conditioning": conditioning_labels,
        "engines": {"h3cspeed": labels["engines"] + ["ffmpeg", "ffprobe"],
                    "comfyui": ["source", "python_env", "comfy_main",
                                 "t8_sampling", "t8_nodes", "comfy_attention",
                                 "model_file", "clip_file", "video_vae_file",
                                 "audio_vae_file"]},
        "hardware": {"gpu_uuid": "GPU-test", "gpu_name": "RTX-3070-Ti",
                     "sm": "sm86", "vram_bytes": 8 * 1024**3,
                     "driver": "test", "toolkit": "CUDA-13.2"},
        "trials": {"cold": 1, "warm": 3},
    }
    bindings: dict = {"models": {}, "conditioning": {}, "engines": {}}
    h3_model_root = root / "h3-model-root"
    h3_model_root.mkdir()
    h3_paths = {
        "fl2va": "FL2VA/transformer/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "qwen": "FL2VA/text_encoder/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "video_vae": "FL2VA/video_vae/source/minimax_h3_video_vae_fp16.safetensors",
        "audio_vae": "FL2VA/audio_vae/minimax_h3_audio_vae_fp32.safetensors",
        "transformer_config": "FL2VA/transformer/config.json",
        "tokenizer": "FL2VA/tokenizer/tokenizer.json",
        "video_vae_config": "FL2VA/video_vae/config.json",
        "audio_vae_config": "FL2VA/audio_vae/config.json",
    }
    for section in bindings:
        for engine, section_labels in spec[section].items():
            bindings[section][engine] = {}
            for label in section_labels:
                path = root / f"{section}-{engine}-{label}.bin"
                if section == "models" and engine == "h3cspeed":
                    path = h3_model_root / h3_paths[label]
                    path.parent.mkdir(parents=True, exist_ok=True)
                elif section == "conditioning" and engine == "h3cspeed" and label == "sidecar":
                    path = root / "h3-conditioning.h3c"
                if section == "engines" and engine == "h3cspeed" and label == "binary":
                    path = h3_binary or Path(sys.executable)
                elif section == "engines" and engine == "h3cspeed" and label in {
                        "ffmpeg", "ffprobe"}:
                    resolved_tool = shutil.which(label)
                    if resolved_tool is None:
                        raise RuntimeError(f"{label} is required for the smoke fixture")
                    path = Path(resolved_tool)
                elif section == "engines" and engine == "comfyui" and label == "python_env":
                    path = Path(sys.executable)
                elif section == "engines" and engine == "comfyui" and label == "source":
                    path = command_driver
                else:
                    path.write_bytes(f"{section}:{engine}:{label}".encode())
                bindings[section][engine][label] = str(path)
    write_h3_sidecar(
        Path(bindings["conditioning"]["h3cspeed"]["sidecar"]),
        "private fox prompt", reference,
        Path(bindings["models"]["h3cspeed"]["qwen"]),
    )
    manifest = perf002.create_input_manifest(spec, bindings, reference, prompt)
    manifest_path = root / "input-manifest.json"
    perf002.publish_json(manifest_path, manifest)
    config = {"reference_png": str(reference), "prompt_file": str(prompt),
            "bindings": bindings}
    return manifest_path, config


class Perf002SmokeTests(unittest.TestCase):
    def test_private_environment_preserves_username_but_drops_unapproved_host_values(self) -> None:
        smoke = load_script("run_perf002_smoke")
        with mock.patch.dict(os.environ, {
            "USERNAME": "perf002-test-user",
            "PERF002_PRIVATE_SECRET": "must-not-cross-boundary",
        }, clear=False):
            environment = smoke._private_environment({})
        self.assertEqual(environment.get("USERNAME"), "perf002-test-user")
        self.assertNotIn("PERF002_PRIVATE_SECRET", environment)

    def test_scheduler_evidence_engine_specific_fps_contract(self) -> None:
        smoke = load_script("run_perf002_smoke")
        base = {
            "schema_version": 1, "engine": "comfyui",
            "sampler": "dual_clock_euler", "schedule": "native_flow",
            "video_shift": 12.0, "audio_shift": 3.0,
            "sigma_video": [1.0, 0.5, 0.0],
            "sigma_audio": [1.0, 0.5, 0.0],
            "raw_audio_protocol_verified": True,
            "width": 864, "height": 480, "frames": 22, "fps": 24,
            "steps": 2, "layers": 50, "seed": 42,
        }
        h3 = dict(base)
        h3["engine"] = "h3cspeed"
        h3.pop("fps")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scheduler.json"
            path.write_text(json.dumps(h3), encoding="utf-8")
            self.assertEqual(
                smoke._scheduler_evidence(path, "h3cspeed")["sampler"],
                "dual_clock_euler")

            path.write_text(json.dumps(base), encoding="utf-8")
            self.assertEqual(
                smoke._scheduler_evidence(path, "comfyui")["sampler"],
                "dual_clock_euler")
            for label, mutation, message in (
                ("missing fps", lambda report: report.pop("fps"), "schema"),
                ("wrong fps", lambda report: report.update({"fps": 30}), "integer 24 fps"),
                ("float fps", lambda report: report.update({"fps": 24.0}), "integer 24 fps"),
                ("boolean fps", lambda report: report.update({"fps": True}), "integer 24 fps"),
                ("extra field", lambda report: report.update({"extra": True}), "schema"),
            ):
                with self.subTest(label=label):
                    invalid = dict(base)
                    mutation(invalid)
                    path.write_text(json.dumps(invalid), encoding="utf-8")
                    with self.assertRaisesRegex(smoke.ContractError, message):
                        smoke._scheduler_evidence(path, "comfyui")

    def test_runtime_directory_is_comfy_only(self) -> None:
        smoke = load_script("run_perf002_smoke")
        self.assertIn("model_root", smoke._required_config_fields("h3cspeed"))
        self.assertIn("command_inputs", smoke._required_config_fields("h3cspeed"))
        self.assertNotIn("runtime_dir", smoke._required_config_fields("h3cspeed"))
        self.assertIn("runtime_dir", smoke._required_config_fields("comfyui"))
        self.assertIn("command_inputs", smoke._required_config_fields("comfyui"))

    def test_perf006_profile_evidence_is_route_bound_and_matched(self) -> None:
        smoke = load_script("run_perf002_smoke")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_dir = root / "baseline"
            candidate_dir = root / "candidate"
            baseline_dir.mkdir()
            candidate_dir.mkdir()

            def write_profile(directory: Path, candidate: bool) -> None:
                counters = {
                    "prefetch_reserve_count": 10 if candidate else 0,
                    "prefetch_upload_count": 10 if candidate else 0,
                    "prefetch_consume_count": 9 if candidate else 0,
                    "prefetch_cancel_count": 0,
                    "prefetch_error_count": 0,
                    "prefetch_block_count": 2 if candidate else 0,
                }
                report = {
                    "schema_version": 1, "kind": "h3cspeed.cuda.profile",
                    "context": {"complete": True},
                    "perf006": {
                        "dit_prefetch_requested": candidate,
                        "dit_prefetch_mode": (
                            "one_ahead_convrot" if candidate else "disabled"),
                        "async_refill_requested": True,
                        "async_refill_active": True, "ssd_streaming": False,
                        "upload_wait_trace_requested": True,
                        "upload_wait_trace_complete": True,
                        "upload_wait_trace_overflow": False,
                        "upload_wait_trace_union_valid": True,
                        "scope": "dit_denoise",
                        "exclusive_upload_ready_wait_seconds": (
                            0.5 if candidate else 2.0),
                        "upload_ready_wait_count": 5,
                        **counters,
                    },
                }
                (directory / "h3-profile-1-2-test-H3_DiT.json").write_text(
                    json.dumps(report), encoding="utf-8")

            write_profile(baseline_dir, False)
            write_profile(candidate_dir, True)
            argv_a = [sys.executable, "-o", str(root / "a.mp4")]
            argv_b = [sys.executable, "-o", str(root / "b.mp4")]
            base_env = {
                "H3_CUDA_ASYNC_REFILL": "1", "H3_CUDA_UPLOAD_WAIT_TRACE": "1",
                "H3_PROFILE_JSON_DIR": str(baseline_dir),
                "H3CSPEED_PERF002_SCHEDULER_TRACE": str(root / "a-scheduler.json"),
                "H3CSPEED_PERF002_ATTENTION_TRACE": str(root / "a-attention.json"),
            }
            candidate_env = dict(base_env)
            candidate_env.update({
                "H3_CUDA_DIT_PREFETCH": "1",
                "H3_PROFILE_JSON_DIR": str(candidate_dir),
                "H3CSPEED_PERF002_SCHEDULER_TRACE": str(root / "b-scheduler.json"),
                "H3CSPEED_PERF002_ATTENTION_TRACE": str(root / "b-attention.json"),
            })
            baseline = smoke._perf006_profile_evidence(
                baseline_dir, base_env, argv_a)
            candidate = smoke._perf006_profile_evidence(
                candidate_dir, candidate_env, argv_b)
            self.assertEqual(baseline["variant"], "baseline")
            self.assertEqual(candidate["variant"], "candidate")
            self.assertEqual(baseline["matched_contract_sha256"],
                             candidate["matched_contract_sha256"])
            self.assertEqual(candidate["prefetch_upload_count"], 10)

            bad = json.loads(next(candidate_dir.glob("*.json")).read_text(
                encoding="utf-8"))
            bad["perf006"]["upload_wait_trace_overflow"] = True
            next(candidate_dir.glob("*.json")).write_text(
                json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(smoke.ContractError, "overflow"):
                smoke._perf006_profile_evidence(candidate_dir, candidate_env, argv_b)

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
            runtime = root / "comfy-runtime"
            config.update({
                "schema_version": 1, "engine": "comfyui",
                "argv": [sys.executable, str(child),
                         "--comfy-main", config["bindings"]["engines"]["comfyui"]["comfy_main"],
                         "--t8-sampling", config["bindings"]["engines"]["comfyui"]["t8_sampling"],
                         "--t8-nodes", config["bindings"]["engines"]["comfyui"]["t8_nodes"],
                         "--comfy-attention", config["bindings"]["engines"]["comfyui"]["comfy_attention"],
                         "--model-file", config["bindings"]["engines"]["comfyui"]["model_file"],
                         "--clip-file", config["bindings"]["engines"]["comfyui"]["clip_file"],
                         "--video-vae-file", config["bindings"]["engines"]["comfyui"]["video_vae_file"],
                         "--audio-vae-file", config["bindings"]["engines"]["comfyui"]["audio_vae_file"],
                         "--reference-png", config["reference_png"],
                         "--prompt-file", config["prompt_file"],
                         "--runtime-dir", str(runtime),
                         "--output-media", str(media), "--scheduler-trace", str(scheduler),
                         "--attention-trace", str(attention), "--ffmpeg",
                         shutil.which("ffmpeg") or "ffmpeg"],
                "environment": {"PYTHONUTF8": "1"},
                "output_media": str(media), "scheduler_trace": str(scheduler),
                "attention_trace": str(attention),
                "runtime_dir": str(runtime),
                "protected_roots": protected,
                "command_artifacts": {
                    "executable": {"label": "python_env", "argv_index": 0},
                    "driver": {"label": "source", "argv_index": 1},
                },
                "command_inputs": {
                    "--comfy-main": {"section": "engines", "label": "comfy_main"},
                    "--t8-sampling": {"section": "engines", "label": "t8_sampling"},
                    "--t8-nodes": {"section": "engines", "label": "t8_nodes"},
                    "--comfy-attention": {"section": "engines", "label": "comfy_attention"},
                    "--model-file": {"section": "engines", "label": "model_file"},
                    "--clip-file": {"section": "engines", "label": "clip_file"},
                    "--video-vae-file": {"section": "engines", "label": "video_vae_file"},
                    "--audio-vae-file": {"section": "engines", "label": "audio_vae_file"},
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
            bad_reference = dict(config)
            bad_reference["argv"] = list(config["argv"])
            alternate_reference = root / "alternate.png"
            write_png(alternate_reference)
            reference_index = bad_reference["argv"].index("--reference-png") + 1
            bad_reference["argv"][reference_index] = str(alternate_reference)
            with self.assertRaisesRegex(smoke.ContractError, "reference PNG"):
                smoke._validate_command(bad_reference, manifest, "comfyui")
            bad_environment = copy.deepcopy(config)
            bad_environment["environment"]["H3_VAE_LAYER_MAJOR"] = "1"
            with self.assertRaisesRegex(smoke.ContractError, "only for h3cspeed"):
                smoke._validate_command(bad_environment, manifest, "comfyui")
            bad_environment = copy.deepcopy(config)
            bad_environment["environment"]["H3_CUDA_DIT_PREFETCH"] = "1"
            with self.assertRaisesRegex(smoke.ContractError, "only for h3cspeed"):
                smoke._validate_command(bad_environment, manifest, "comfyui")
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
                protected_runtime = Path(protected[0]) / "runtime"
                config["runtime_dir"] = str(protected_runtime)
                runtime_index = config["argv"].index("--runtime-dir") + 1
                config["argv"][runtime_index] = str(protected_runtime)
                config_path.write_text(json.dumps(config), encoding="utf-8")
                smoke.run_smoke(
                    manifest_path, config_path, Path(protected[0]),
                    shutil.which("ffmpeg") or "ffmpeg",
                    shutil.which("ffprobe") or "ffprobe", 60)
            self.assertFalse(protected_runtime.exists())

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                         "ffmpeg and ffprobe are required")
    def test_h3_direct_binary_smoke_binds_sidecar_and_model_root(self) -> None:
        perf002 = load_script("run_perf002_ab")
        smoke = load_script("run_perf002_smoke")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = root / "fake_h3.py"
            write_fake_h3_engine(fake)
            launcher = root / ("fake_h3.cmd" if os.name == "nt" else "fake_h3")
            write_fake_h3_launcher(launcher, fake)
            manifest_path, config = create_manifest(
                root, perf002, fake, h3_binary=launcher)
            manifest, _ = smoke.load_input(manifest_path)
            model_root = root / "h3-model-root"
            media_dir = root / "h3-output"
            evidence_dir = root / "h3-evidence"
            media_dir.mkdir()
            evidence_dir.mkdir()
            source_root = root / "source-root"
            comfy_root = root / "comfy-root"
            source_root.mkdir()
            comfy_root.mkdir()
            media = media_dir / "smoke.mp4"
            scheduler = media_dir / "scheduler.json"
            attention = media_dir / "attention.json"
            sidecar = config["bindings"]["conditioning"]["h3cspeed"]["sidecar"]
            qwen_hash = manifest["models"]["h3cspeed"]["qwen"]
            config.update({
                "schema_version": 1, "engine": "h3cspeed",
                "model_root": str(model_root),
                "argv": [str(launcher), "-d", str(model_root),
                         "-p", "private fox prompt", "--first-frame",
                         config["reference_png"], "--width", "864", "--height", "480",
                         "--frames", "22", "--steps", "2", "--layers", "50",
                         "--reuse", "1", "--core-reuse", "1", "--seed", "42",
                         "-o", str(media)],
                "environment": {
                    "PYTHONUTF8": "1",
                    "H3_CUDA_ATTENTION": "sage",
                    "H3_CUDA_DIT_PREFETCH": "1",
                    "H3_CUDA_TF32": "0",
                    "H3_PROFILE": "1",
                    "H3_VAE_LAYER_MAJOR": "1",
                    "H3_FFMPEG": config["bindings"]["engines"]["h3cspeed"]["ffmpeg"],
                    "H3CSPEED_TEXT_EMBEDDING": sidecar,
                    "H3CSPEED_TEXT_ENCODER_SHA256": qwen_hash,
                    "H3CSPEED_PERF002_SCHEDULER_TRACE": str(scheduler),
                    "H3CSPEED_PERF002_ATTENTION_TRACE": str(attention),
                },
                "output_media": str(media), "scheduler_trace": str(scheduler),
                "attention_trace": str(attention),
                "protected_roots": [str(source_root), str(comfy_root), str(model_root)],
                "command_artifacts": {
                    "executable": {"label": "binary", "argv_index": 0},
                    "driver": {"label": "binary", "argv_index": 0},
                },
                "command_inputs": {
                    key: {"section": section, "label": label}
                    for key, (section, label) in smoke.H3_INPUT_BINDINGS.items()
                },
            })
            config_path = root / "h3-command.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            result_path = smoke.run_smoke(
                manifest_path, config_path, evidence_dir,
                shutil.which("ffmpeg") or "ffmpeg",
                shutil.which("ffprobe") or "ffprobe", 60)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "SMOKE_PASS")
            self.assertEqual(result["engine"], "h3cspeed")
            self.assertEqual(result["matched_ab_status"], "NOT_RUN")
            self.assertEqual(
                result["optimization_evidence"]["video_vae_traversal"],
                "layer_major")
            self.assertEqual(result["optimization_evidence"]["upload_gib"], 9.03)
            invalid_log = root / "invalid-layer-major.log"
            invalid_log.write_text(
                "h3: layer-major video VAE 1 chunks x 8 tiles "
                "(8 states, 16224 KiB hidden each)\n"
                "h3cspeed CUDA [video VAE decoder]: device-live=0.00 MiB "
                "peak=2036.43 MiB uploads=3528/72.23 GiB "
                "linear=1176 conv=0 sdpa=288\n",
                encoding="utf-8")
            with self.assertRaisesRegex(smoke.ContractError, "80% reduction"):
                smoke._h3_layer_major_evidence(invalid_log)

            baseline = copy.deepcopy(config)
            baseline_media_dir = root / "h3-baseline-output"
            baseline_evidence_dir = root / "h3-baseline-evidence"
            baseline_media_dir.mkdir()
            baseline_evidence_dir.mkdir()
            baseline["output_media"] = str(baseline_media_dir / "smoke.mp4")
            baseline["scheduler_trace"] = str(baseline_media_dir / "scheduler.json")
            baseline["attention_trace"] = str(baseline_media_dir / "attention.json")
            baseline["argv"][baseline["argv"].index("-o") + 1] = baseline["output_media"]
            baseline["environment"].pop("H3_VAE_LAYER_MAJOR")
            baseline["environment"]["H3CSPEED_PERF002_SCHEDULER_TRACE"] = baseline["scheduler_trace"]
            baseline["environment"]["H3CSPEED_PERF002_ATTENTION_TRACE"] = baseline["attention_trace"]
            baseline_path = root / "h3-baseline-command.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            baseline_result_path = smoke.run_smoke(
                manifest_path, baseline_path, baseline_evidence_dir,
                shutil.which("ffmpeg") or "ffmpeg",
                shutil.which("ffprobe") or "ffprobe", 60)
            baseline_result = json.loads(
                baseline_result_path.read_text(encoding="utf-8"))
            self.assertEqual(baseline_result["status"], "SMOKE_PASS")
            self.assertNotIn("optimization_evidence", baseline_result)

            def rejected(mutation, message: str) -> None:
                bad = copy.deepcopy(config)
                mutation(bad)
                with self.assertRaisesRegex(smoke.ContractError, message):
                    smoke._validate_command(bad, manifest, "h3cspeed")

            rejected(lambda bad: bad["environment"].update(
                {"H3CSPEED_TEXT_EMBEDDING": str(root / "other.h3c")}),
                "manifest sidecar")
            rejected(lambda bad: bad["environment"].update(
                {"H3CSPEED_TEXT_ENCODER_SHA256": "0" * 64}), "Qwen hash")
            rejected(lambda bad: bad["bindings"]["models"]["h3cspeed"].update(
                {"qwen": config["bindings"]["models"]["comfyui"]["qwen"]}),
                "native loader path")
            rejected(lambda bad: bad["argv"].__setitem__(
                bad["argv"].index("-p") + 1, "wrong prompt"), "prompt file")
            rejected(lambda bad: bad["argv"].__setitem__(
                bad["argv"].index("--width") + 1, "640"), "--width")
            alternate = root / "alternate.png"
            write_png(alternate)
            rejected(lambda bad: bad["argv"].__setitem__(
                bad["argv"].index("--first-frame") + 1, str(alternate)),
                "manifest reference")
            rejected(lambda bad: bad["environment"].update(
                {"H3_CUDA_TF32": "1"}), "TF32=0")
            rejected(lambda bad: bad["environment"].update(
                {"H3_VAE_LAYER_MAJOR": "0"}), "absent or exactly 1")
            rejected(lambda bad: bad["environment"].update(
                {"H3_CUDA_DIT_PREFETCH": "0"}), "absent or exactly 1")
            rejected(lambda bad: bad["argv"].insert(
                bad["argv"].index("-o"), "--ssd-streaming"),
                "requires non-SSD ConvRot")
            rejected(lambda bad: bad["environment"].pop("H3_PROFILE"),
                     "requires H3_PROFILE=1")
            rejected(lambda bad: bad["environment"].update(
                {"H3_FFMPEG": str(launcher)}), "manifest-bound ffmpeg")
            rejected(lambda bad: bad["argv"].insert(1, str(fake)),
                     "unbound positional")
            extra = model_root / "FL2VA/transformer/extra.safetensors"
            extra.write_bytes(b"extra")
            rejected(lambda bad: None, "inventory is not exact")
            extra.unlink()
            rejected(lambda bad: Path(
                bad["bindings"]["conditioning"]["h3cspeed"]["sidecar"]
            ).write_bytes(b"not-a-v2-sidecar"), "sidecar")

    def test_attention_fallback_is_rejected(self) -> None:
        smoke = load_script("run_perf002_smoke")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "attention.json"
            path.write_text(json.dumps({
                "schema_version": 1, "engine": "comfyui", "requested": "sage",
                "selected": "sage", "scope": "dit_bf16", "backend_hits": 1,
                "expected_native_calls": 2, "unexpected_fallbacks": 1,
            }), encoding="utf-8")
            with self.assertRaisesRegex(smoke.ContractError, "zero fallbacks"):
                smoke._attention_evidence(path, "comfyui")
            value = json.loads(path.read_text(encoding="utf-8"))
            value["unexpected_fallbacks"] = 0
            value["scope"] = "whole_pipeline"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(smoke.ContractError, "did not select Sage"):
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
