#!/usr/bin/env python3
"""Create and validate immutable PERF-002 benchmark evidence.

This is the portable walking skeleton for a matched ComfyUI/h3cspeed A/B run.
It deliberately does not launch either engine yet.  It binds the immutable
inputs and validates one engine's produced media without recording prompts,
absolute model paths, command lines, or ambient environment variables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import tempfile
from typing import Any
import zlib


SCHEMA_VERSION = 1
INPUT_KIND = "h3cspeed.perf002.input"
RESULT_KIND = "h3cspeed.perf002.result"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")
TRACKS = ("algorithm_parity", "engine_recommended")
ENGINES = ("h3cspeed", "comfyui")
REQUIRED_MODEL_LABELS = {"fl2va", "qwen", "video_vae", "audio_vae"}
REQUIRED_ENGINE_LABELS = {
    "h3cspeed": {"binary", "source", "runtime"},
    "comfyui": {"source", "python_env"},
}
REQUIRED_CONDITIONING_LABELS = {"token_ids", "token_tags", "qwen_hidden"}
STAGES = (
    "conditioning", "model_load", "keyframe_encode", "dit", "video_vae",
    "audio_vae", "mux", "total",
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ContractError(RuntimeError):
    """The benchmark evidence is incomplete, unsafe, or inconsistent."""


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, separators=(",", ":"),
                       sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _windows_reparse(path: Path) -> bool:
    return bool(getattr(path.lstat(), "st_file_attributes", 0) & 0x400)


def _reject_link_chain(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    for candidate in (absolute, *absolute.parents):
        if not candidate.exists() and not candidate.is_symlink():
            continue
        if candidate.is_symlink() or _windows_reparse(candidate):
            raise ContractError(f"{label} path must not contain links or reparse points")


def safe_regular(path: Path, label: str) -> Path:
    _reject_link_chain(path, label)
    candidate = path.resolve(strict=True)
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode) or not candidate.is_file():
        raise ContractError(f"{label} must be a regular file")
    return candidate


def safe_output_directory(path: Path) -> Path:
    _reject_link_chain(path, "output directory")
    candidate = path.resolve(strict=True)
    if not candidate.is_dir():
        raise ContractError("output directory must be an existing directory")
    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError:
        pass
    else:
        raise ContractError("benchmark evidence must be outside the source tree")
    return candidate


def png_dimensions(path: Path) -> tuple[int, int]:
    source = safe_regular(path, "reference PNG")
    size = source.stat().st_size
    if size > 64 * 1024 * 1024:
        raise ContractError("reference PNG exceeds the 64 MiB fixture limit")
    payload = source.read_bytes()
    if len(payload) < 45 or payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ContractError("reference fixture must be a PNG")
    offset = 8
    width = height = bit_depth = color_type = interlace = None
    compressed = bytearray()
    seen_iend = False
    while offset + 12 <= len(payload):
        length = struct.unpack(">I", payload[offset:offset + 4])[0]
        end = offset + 12 + length
        if end > len(payload):
            raise ContractError("reference PNG has a truncated chunk")
        kind = payload[offset + 4:offset + 8]
        data = payload[offset + 8:offset + 8 + length]
        expected_crc = struct.unpack(">I", payload[offset + 8 + length:end])[0]
        if zlib.crc32(kind + data) & 0xFFFFFFFF != expected_crc:
            raise ContractError("reference PNG chunk CRC mismatch")
        if kind == b"IHDR":
            if offset != 8 or length != 13:
                raise ContractError("reference PNG has an invalid IHDR")
            width, height, bit_depth, color_type, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", data))
            if compression != 0 or filtering != 0:
                raise ContractError("reference PNG uses unsupported compression")
        elif kind == b"IDAT":
            compressed.extend(data)
        elif kind == b"IEND":
            if length != 0 or end != len(payload):
                raise ContractError("reference PNG has an invalid IEND")
            seen_iend = True
            break
        offset = end
    if not seen_iend or not compressed or None in (width, height):
        raise ContractError("reference PNG is incomplete")
    if bit_depth != 8 or color_type not in (2, 6) or interlace != 0:
        raise ContractError("reference PNG must be non-interlaced 8-bit RGB/RGBA")
    channels = 3 if color_type == 2 else 4
    expected_bytes = int(height) * (1 + int(width) * channels)
    try:
        decoder = zlib.decompressobj()
        decoded = decoder.decompress(bytes(compressed), expected_bytes + 1)
        if len(decoded) > expected_bytes or decoder.unconsumed_tail:
            raise ContractError("reference PNG pixel payload exceeds its dimensions")
        decoded += decoder.flush()
    except zlib.error as error:
        raise ContractError("reference PNG IDAT is corrupt") from error
    if (len(decoded) != expected_bytes or decoder.unconsumed_tail or
            decoder.unused_data or not decoder.eof):
        raise ContractError("reference PNG pixel payload is incomplete")
    stride = 1 + int(width) * channels
    if any(decoded[row * stride] > 4 for row in range(int(height))):
        raise ContractError("reference PNG contains an invalid row filter")
    return int(width), int(height)


def _exclusive_write(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ContractError(f"refusing to overwrite evidence: {path.name}")
    path.parent.mkdir(parents=False, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="xb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
        temporary.unlink()
    except FileExistsError as error:
        temporary.unlink(missing_ok=True)
        raise ContractError(f"refusing to overwrite evidence: {path.name}") from error
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def publish_json(path: Path, value: dict[str, Any]) -> str:
    payload = canonical_bytes(value)
    digest = sha256_bytes(payload)
    _exclusive_write(path, payload)
    try:
        _exclusive_write(path.with_suffix(path.suffix + ".sha256"),
                         f"{digest}  {path.name}\n".encode("ascii"))
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return digest


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be an object")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ContractError(f"{name} has unexpected or missing fields")


def _label(value: Any, name: str) -> str:
    if not isinstance(value, str) or not LABEL_RE.fullmatch(value):
        raise ContractError(f"{name} must be a portable non-sensitive label")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ContractError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{name} must be a positive integer")
    return value


def _nonnegative_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ContractError(f"{name} must be finite and non-negative")
    return converted


def _validate_hash_map(value: Any, name: str) -> dict[str, str]:
    mapping = _mapping(value, name)
    if not mapping:
        raise ContractError(f"{name} must not be empty")
    for key, digest in mapping.items():
        _label(key, f"{name} label")
        _sha(digest, f"{name}.{key}")
    return mapping


def validate_input_manifest(value: Any) -> dict[str, Any]:
    root = _mapping(value, "manifest")
    _exact_keys(root, {
        "schema_version", "kind", "benchmark_id", "track", "fixture",
        "prompt_sha256", "render", "scheduler", "attention", "models",
        "conditioning", "engines", "hardware", "trials",
    }, "manifest")
    if root.get("schema_version") != SCHEMA_VERSION or root.get("kind") != INPUT_KIND:
        raise ContractError("unsupported PERF-002 input schema")
    _label(root.get("benchmark_id"), "benchmark_id")
    if root.get("track") not in TRACKS:
        raise ContractError("unsupported benchmark track")
    _sha(root.get("prompt_sha256"), "prompt_sha256")

    fixture = _mapping(root.get("fixture"), "fixture")
    _exact_keys(fixture, {"label", "sha256", "width", "height", "mode"}, "fixture")
    _label(fixture.get("label"), "fixture.label")
    _sha(fixture.get("sha256"), "fixture.sha256")
    if (fixture.get("width"), fixture.get("height"), fixture.get("mode")) != (
        864, 480, "fl2va_first_frame"
    ):
        raise ContractError("fixture must be native 864x480 FL2VA first-frame I2V")

    render = _mapping(root.get("render"), "render")
    expected_render = {
        "width": 864, "height": 480, "render_width": 864,
        "render_height": 480, "frames": 124, "fps": 24, "steps": 8,
        "layers": 50, "reuse": 1, "core_reuse": 1, "seed": 42,
    }
    if render != expected_render:
        raise ContractError("render contract is not the matched native 480p baseline")

    scheduler = _mapping(root.get("scheduler"), "scheduler")
    _exact_keys(scheduler, {"h3cspeed", "comfyui", "sigma_tolerance"}, "scheduler")
    for engine in ENGINES:
        item = _mapping(scheduler.get(engine), f"scheduler.{engine}")
        _exact_keys(item, {"sampler", "schedule", "video_shift", "audio_shift"},
                    f"scheduler.{engine}")
        _label(item.get("sampler"), f"scheduler.{engine}.sampler")
        _label(item.get("schedule"), f"scheduler.{engine}.schedule")
        _nonnegative_number(item.get("video_shift"), f"scheduler.{engine}.video_shift")
        _nonnegative_number(item.get("audio_shift"), f"scheduler.{engine}.audio_shift")
    tolerance = _nonnegative_number(scheduler.get("sigma_tolerance"),
                                    "scheduler.sigma_tolerance")
    if root["track"] == "algorithm_parity":
        expected = {"sampler": "dual_clock_euler", "schedule": "native_flow",
                    "video_shift": 12.0, "audio_shift": 3.0}
        if scheduler["h3cspeed"] != expected or scheduler["comfyui"] != expected:
            raise ContractError("algorithm_parity requires identical dual-clock schedules")
        if tolerance > 1e-6:
            raise ContractError("algorithm_parity sigma tolerance must be <= 1e-6")

    attention = _mapping(root.get("attention"), "attention")
    if attention != {"requested": "sage", "tf32": False,
                     "backend_hit_required": True, "fallback_allowed": False}:
        raise ContractError("attention contract must require verified Sage without fallback")

    models = _mapping(root.get("models"), "models")
    conditioning = _mapping(root.get("conditioning"), "conditioning")
    engines = _mapping(root.get("engines"), "engines")
    for engine in ENGINES:
        model_hashes = _validate_hash_map(models.get(engine), f"models.{engine}")
        if not REQUIRED_MODEL_LABELS.issubset(model_hashes):
            raise ContractError(f"models.{engine} lacks required FL2VA/Qwen/VAE bindings")
        engine_hashes = _validate_hash_map(engines.get(engine), f"engines.{engine}")
        if not REQUIRED_ENGINE_LABELS[engine].issubset(engine_hashes):
            raise ContractError(f"engines.{engine} lacks required runtime bindings")
        conditioning_hashes = _validate_hash_map(
            conditioning.get(engine), f"conditioning.{engine}")
        if not REQUIRED_CONDITIONING_LABELS.issubset(conditioning_hashes):
            raise ContractError(
                f"conditioning.{engine} lacks token IDs/tags/Qwen hidden bindings")

    hardware = _mapping(root.get("hardware"), "hardware")
    _exact_keys(hardware, {"gpu_uuid", "gpu_name", "sm", "vram_bytes",
                           "driver", "toolkit"}, "hardware")
    for key in ("gpu_uuid", "gpu_name", "sm", "driver", "toolkit"):
        _label(hardware.get(key), f"hardware.{key}")
    _positive_int(hardware.get("vram_bytes"), "hardware.vram_bytes")
    trials = _mapping(root.get("trials"), "trials")
    if trials != {"cold": 1, "warm": 3}:
        raise ContractError("matched A/B requires one cold and three warm trials")
    return root


def _hash_binding_group(labels: Any, bindings: Any, name: str) -> dict[str, str]:
    if not isinstance(labels, list) or not labels:
        raise ContractError(f"{name} labels must be a non-empty list")
    binding_map = _mapping(bindings, f"{name} bindings")
    if set(labels) != set(binding_map):
        raise ContractError(f"{name} labels and bindings must match exactly")
    result: dict[str, str] = {}
    for label in labels:
        _label(label, f"{name} label")
        source = safe_regular(Path(binding_map[label]), f"{name}.{label}")
        result[label] = sha256_file(source)
    return result


def create_input_manifest(specification: Any, bindings: Any, reference_png: Path,
                          prompt_file: Path) -> dict[str, Any]:
    spec = _mapping(specification, "specification").copy()
    if "benchmark_id" in spec:
        raise ContractError("specification must not choose benchmark_id")
    reference = safe_regular(reference_png, "reference PNG")
    prompt = safe_regular(prompt_file, "prompt file")
    if png_dimensions(reference) != (864, 480):
        raise ContractError("reference PNG must be native 864x480")
    fixture = _mapping(spec.get("fixture"), "fixture").copy()
    fixture["sha256"] = sha256_file(reference)
    fixture["width"] = 864
    fixture["height"] = 480
    spec["fixture"] = fixture
    spec["prompt_sha256"] = sha256_file(prompt)
    binding_root = _mapping(bindings, "bindings")
    if set(binding_root) != {"models", "conditioning", "engines"}:
        raise ContractError("bindings must contain models, conditioning, and engines")
    for section in ("models", "conditioning", "engines"):
        labels_by_engine = _mapping(spec.get(section), section)
        files_by_engine = _mapping(binding_root.get(section), f"bindings.{section}")
        if set(labels_by_engine) != set(ENGINES) or set(files_by_engine) != set(ENGINES):
            raise ContractError(f"{section} must bind both engines")
        spec[section] = {
            engine: _hash_binding_group(labels_by_engine[engine], files_by_engine[engine],
                                        f"{section}.{engine}")
            for engine in ENGINES
        }
    spec["schema_version"] = SCHEMA_VERSION
    spec["kind"] = INPUT_KIND
    identity_payload = canonical_bytes(spec)
    spec["benchmark_id"] = f"perf002-{sha256_bytes(identity_payload)[:16]}"
    return validate_input_manifest(spec)


def _run(command: list[str], *, binary: bool = False) -> bytes | str:
    completed = subprocess.run(command, check=False, capture_output=True,
                               text=not binary, shell=False, timeout=300)
    if completed.returncode != 0:
        raise ContractError("media command failed; inspect the engine-local log")
    return completed.stdout


def validate_media(ffmpeg: str, ffprobe: str, media: Path,
                   sample_directory: Path) -> dict[str, Any]:
    media = safe_regular(media, "media")
    sample_directory = safe_output_directory(sample_directory)
    raw = _run([ffprobe, "-v", "error", "-show_streams", "-show_format",
                "-count_frames", "-of", "json", str(media)])
    probe = json.loads(str(raw))
    streams = probe.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    if not video or not audio:
        raise ContractError("media must contain video and audio")
    if (video.get("codec_name"), video.get("width"), video.get("height"),
            video.get("r_frame_rate"), int(video.get("nb_read_frames", 0))) != (
            "h264", 864, 480, "24/1", 124):
        raise ContractError("video does not match 864x480 H.264 124f/24fps")
    if audio.get("codec_name") != "aac" or int(audio.get("sample_rate", 0)) != 32000:
        raise ContractError("audio must be AAC at 32 kHz")
    if int(audio.get("channels", 0)) != 2:
        raise ContractError("audio must be stereo")
    _run([ffmpeg, "-v", "error", "-xerror", "-i", str(media),
          "-map", "0:v:0", "-f", "null", os.devnull])
    pcm = _run([ffmpeg, "-v", "error", "-xerror", "-i", str(media),
                "-map", "0:a:0", "-f", "s16le", "-ac", "2", "-ar", "32000", "-"],
               binary=True)
    assert isinstance(pcm, bytes)
    if not pcm or not any(pcm):
        raise ContractError("decoded audio is empty or silent")
    frame_hashes: dict[str, str] = {}
    for frame in (0, 31, 62, 93, 123):
        name = f"frame-{frame + 1:03d}.png"
        destination = sample_directory / name
        if destination.exists() or destination.is_symlink():
            raise ContractError(f"refusing to overwrite sample frame: {name}")
        frame_png = _run([ffmpeg, "-v", "error", "-xerror", "-i", str(media),
                          "-vf", f"select=eq(n\\,{frame})", "-frames:v", "1",
                          "-map_metadata", "-1", "-f", "image2pipe", "-vcodec",
                          "png", "-"], binary=True)
        assert isinstance(frame_png, bytes)
        if not frame_png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ContractError("ffmpeg did not return a PNG sample")
        _exclusive_write(destination, frame_png)
        frame_hashes[name] = sha256_file(safe_regular(destination, name))
    duration = float(probe.get("format", {}).get("duration", 0.0))
    if not math.isfinite(duration) or not (5.14 <= duration <= 5.20):
        raise ContractError("media duration is outside the raw H3 124-frame contract")
    return {
        "sha256": sha256_file(media), "bytes": media.stat().st_size,
        "video": {"codec": "h264", "width": 864, "height": 480,
                  "fps": "24/1", "frames": 124},
        "audio": {"codec": "aac", "sample_rate": 32000, "channels": 2,
                  "decoded_pcm_bytes": len(pcm), "non_silent": True},
        "duration_seconds": duration, "frame_hashes": frame_hashes,
        "full_decode": True, "visual_review": "MANUAL_REQUIRED",
    }


def validate_result(value: Any, input_digest: str) -> dict[str, Any]:
    root = _mapping(value, "result")
    _exact_keys(root, {"schema_version", "kind", "input_manifest_sha256", "engine",
                       "trial", "timings", "scheduler_evidence",
                       "attention_evidence", "media", "status"}, "result")
    if root.get("schema_version") != SCHEMA_VERSION or root.get("kind") != RESULT_KIND:
        raise ContractError("unsupported PERF-002 result schema")
    if _sha(root.get("input_manifest_sha256"), "input_manifest_sha256") != input_digest:
        raise ContractError("result is not bound to this input manifest")
    if root.get("engine") not in ENGINES:
        raise ContractError("unsupported result engine")
    trial = _mapping(root.get("trial"), "trial")
    if (trial.get("kind") not in ("cold", "warm") or
            isinstance(trial.get("index"), bool) or
            not isinstance(trial.get("index"), int) or trial["index"] < 0):
        raise ContractError("invalid trial identity")
    timings = _mapping(root.get("timings"), "timings")
    _exact_keys(timings, set(STAGES), "timings")
    for stage in STAGES:
        _nonnegative_number(timings[stage], f"timings.{stage}")
    scheduler = _mapping(root.get("scheduler_evidence"), "scheduler_evidence")
    _exact_keys(scheduler, {"sampler", "schedule", "video_shift", "audio_shift",
                            "sigma_video_sha256", "sigma_audio_sha256",
                            "sigma_max_abs_diff", "raw_audio_protocol_verified"},
                "scheduler_evidence")
    _sha(scheduler.get("sigma_video_sha256"), "scheduler_evidence.sigma_video_sha256")
    _sha(scheduler.get("sigma_audio_sha256"), "scheduler_evidence.sigma_audio_sha256")
    _label(scheduler.get("sampler"), "scheduler_evidence.sampler")
    _label(scheduler.get("schedule"), "scheduler_evidence.schedule")
    for key in ("video_shift", "audio_shift", "sigma_max_abs_diff"):
        _nonnegative_number(scheduler.get(key), f"scheduler_evidence.{key}")
    if not isinstance(scheduler.get("raw_audio_protocol_verified"), bool):
        raise ContractError("raw_audio_protocol_verified must be boolean")
    attention = _mapping(root.get("attention_evidence"), "attention_evidence")
    _exact_keys(attention, {"requested", "selected", "backend_hits", "fallbacks",
                            "backend_trace_sha256"},
                "attention_evidence")
    if attention.get("requested") != "sage" or attention.get("selected") not in (
            "sage", "native", "pytorch", "unverified"):
        raise ContractError("invalid attention backend evidence")
    _sha(attention.get("backend_trace_sha256"),
         "attention_evidence.backend_trace_sha256")
    hits = attention.get("backend_hits")
    fallbacks = attention.get("fallbacks")
    if not isinstance(hits, int) or hits < 0 or not isinstance(fallbacks, int) or fallbacks < 0:
        raise ContractError("attention counters must be non-negative integers")
    status = root.get("status")
    if status not in ("NOT_RUN", "NOT_PASS"):
        raise ContractError("this walking skeleton accepts only NOT_RUN or NOT_PASS")
    if hits > 0 and attention["backend_trace_sha256"] == "0" * 64:
        raise ContractError("attention backend hits require a real trace digest")
    if hits == 0 or fallbacks != 0:
        if status not in ("NOT_PASS", "NOT_RUN"):
            raise ContractError("missing Sage hit or any fallback must fail closed")
    media = _mapping(root.get("media"), "media")
    _exact_keys(media, {"sha256", "bytes", "video", "audio", "duration_seconds",
                        "frame_hashes", "full_decode", "visual_review"}, "media")
    if media.get("visual_review") != "MANUAL_REQUIRED":
        raise ContractError("automated media QA cannot claim visual review")
    _sha(media.get("sha256"), "media.sha256")
    _positive_int(media.get("bytes"), "media.bytes")
    if media.get("video") != {"codec": "h264", "width": 864, "height": 480,
                              "fps": "24/1", "frames": 124}:
        raise ContractError("result video contract is incomplete")
    audio = _mapping(media.get("audio"), "media.audio")
    if (audio.get("codec"), audio.get("sample_rate"), audio.get("channels"),
            audio.get("non_silent")) != ("aac", 32000, 2, True):
        raise ContractError("result audio contract is incomplete")
    _positive_int(audio.get("decoded_pcm_bytes"), "media.audio.decoded_pcm_bytes")
    frame_hashes = _validate_hash_map(media.get("frame_hashes"), "media.frame_hashes")
    if set(frame_hashes) != {f"frame-{index:03d}.png" for index in (1, 32, 63, 94, 124)}:
        raise ContractError("result must bind the five prescribed QA frames")
    if media.get("full_decode") is not True:
        raise ContractError("result must prove a full decode")
    _nonnegative_number(media.get("duration_seconds"), "media.duration_seconds")
    if not 5.14 <= float(media["duration_seconds"]) <= 5.20:
        raise ContractError("result duration is outside the raw 124-frame contract")
    return root


def load_input(path: Path) -> tuple[dict[str, Any], str]:
    path = safe_regular(path, "input manifest")
    payload = path.read_bytes()
    if payload != canonical_bytes(json.loads(payload)):
        raise ContractError("input manifest is not canonical JSON")
    digest = sha256_bytes(payload)
    checksum = safe_regular(path.with_suffix(path.suffix + ".sha256"), "manifest checksum")
    if checksum.read_text(encoding="ascii") != f"{digest}  {path.name}\n":
        raise ContractError("input manifest checksum mismatch")
    return validate_input_manifest(json.loads(payload)), digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create-input")
    create.add_argument("--spec", required=True, type=Path)
    create.add_argument("--bindings", required=True, type=Path)
    create.add_argument("--reference-png", required=True, type=Path)
    create.add_argument("--prompt-file", required=True, type=Path)
    create.add_argument("--output-dir", required=True, type=Path)
    validate = subparsers.add_parser("validate-input")
    validate.add_argument("manifest", type=Path)
    media = subparsers.add_parser("validate-media")
    media.add_argument("--manifest", required=True, type=Path)
    media.add_argument("--engine", required=True, choices=ENGINES)
    media.add_argument("--media", required=True, type=Path)
    media.add_argument("--output-dir", required=True, type=Path)
    media.add_argument("--ffmpeg", default="ffmpeg")
    media.add_argument("--ffprobe", default="ffprobe")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "create-input":
        output = safe_output_directory(args.output_dir)
        spec_path = safe_regular(args.spec, "specification")
        bindings_path = safe_regular(args.bindings, "bindings")
        specification = json.loads(spec_path.read_text(encoding="utf-8"))
        bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
        manifest = create_input_manifest(specification, bindings, args.reference_png,
                                         args.prompt_file)
        destination = output / "input-manifest.json"
        digest = publish_json(destination, manifest)
        print(f"PERF-002 input created: {digest}")
        return 0
    _, digest = load_input(args.manifest)
    if args.command == "validate-input":
        print(f"PERF-002 input PASS: {digest}")
        return 0
    output = safe_output_directory(args.output_dir)
    evidence = validate_media(args.ffmpeg, args.ffprobe, args.media, output)
    result = {
        "schema_version": SCHEMA_VERSION, "kind": RESULT_KIND,
        "input_manifest_sha256": digest, "engine": args.engine,
        "trial": {"kind": "cold", "index": 0},
        "timings": {stage: 0.0 for stage in STAGES},
        "scheduler_evidence": {
            "sampler": "unverified", "schedule": "unverified", "video_shift": 0.0,
            "audio_shift": 0.0, "sigma_video_sha256": "0" * 64,
            "sigma_audio_sha256": "0" * 64, "sigma_max_abs_diff": 0.0,
            "raw_audio_protocol_verified": False,
        },
        "attention_evidence": {"requested": "sage", "selected": "unverified",
                               "backend_hits": 0, "fallbacks": 0,
                               "backend_trace_sha256": "0" * 64},
        "media": evidence, "status": "NOT_RUN",
    }
    validate_result(result, digest)
    destination = output / f"{args.engine}-run-result.json"
    result_digest = publish_json(destination, result)
    print(f"PERF-002 media harness PASS: {result_digest}")
    print("matched A/B status: NOT_RUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
