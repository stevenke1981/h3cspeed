#!/usr/bin/env python3
"""Build and optionally execute a private, no-clobber H3/Comfy matrix plan.

The default is a read-only contract plan for native 448x256, 864x480, and
1280x704 profiles.  ``--execute`` is an explicit GPU side effect: it runs the
H3 and ComfyUI child commands sequentially, keeps each engine's output private,
and does not convert process completion into media or quality acceptance.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable
import zlib


SCHEMA_VERSION = 1
CONFIG_KIND = "h3cspeed.resolution-matrix.config"
PLAN_KIND = "h3cspeed.resolution-matrix.plan"
EXECUTION_KIND = "h3cspeed.resolution-matrix.execution"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCER = PROJECT_ROOT / "scripts" / "perf002_comfy_trace.py"
GRID = 32
MAX_PIXELS = 768 * 1344
FRAMES = 124
FPS = 24
STEPS = 2
LAYERS = 50
SEED = 42
PROFILE_DIMENSIONS = {
    "240": (448, 256),
    "480": (864, 480),
    "720": (1280, 704),
}
PATH_FIELDS = (
    "python_executable",
    "powershell_executable",
    "comfy_main",
    "t8_sampling",
    "t8_nodes",
    "comfy_attention",
    "model_file",
    "clip_file",
    "video_vae_file",
    "audio_vae_file",
    "prompt_file",
    "h3_text_encoder",
    "h3_binary",
)
DIRECTORY_FIELDS = ("h3_model_root", "h3_comfy_root")
CONFIG_FIELDS = {"schema_version", "kind", "profiles", *PATH_FIELDS,
                 *DIRECTORY_FIELDS}
PROFILE_CONFIG_FIELDS = {"reference_png", "timeout_seconds"}


class ContractError(RuntimeError):
    """The matrix configuration or private output boundary is unsafe."""


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, separators=(",", ":"),
                       sort_keys=True) + "\n").encode("utf-8")


def _windows_reparse(path: Path) -> bool:
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError as error:
        raise ContractError(f"could not inspect {path}") from error
    return bool(attributes & 0x400)


def _reject_link_chain(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    for candidate in (absolute, *absolute.parents):
        if not candidate.exists() and not candidate.is_symlink():
            continue
        if candidate.is_symlink() or _windows_reparse(candidate):
            raise ContractError(f"{label} path must not contain links or reparse points")


def _regular_file(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be an absolute regular-file path")
    path = Path(value)
    _reject_link_chain(path, label)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ContractError(f"{label} must be an absolute regular-file path")
    return path.resolve(strict=True)


def _existing_directory(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be an absolute directory path")
    path = Path(value)
    _reject_link_chain(path, label)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ContractError(f"{label} must be an absolute directory path")
    return path.resolve(strict=True)


def _outside_project(path: Path, label: str) -> None:
    try:
        path.resolve(strict=False).relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return
    raise ContractError(f"{label} must be outside the source tree")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def png_dimensions(path: Path) -> tuple[int, int]:
    """Read a bounded, structurally valid PNG and return its IHDR dimensions."""
    source = _regular_file(str(path), "profile reference_png")
    if source.stat().st_size > 64 * 1024 * 1024:
        raise ContractError("profile reference_png exceeds the 64 MiB limit")
    payload = source.read_bytes()
    if len(payload) < 45 or payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ContractError("profile reference_png must be a PNG")
    offset = 8
    width = height = None
    seen_iend = False
    while offset + 12 <= len(payload):
        length = struct.unpack(">I", payload[offset:offset + 4])[0]
        end = offset + 12 + length
        if end > len(payload):
            raise ContractError("profile reference_png has a truncated chunk")
        kind = payload[offset + 4:offset + 8]
        data = payload[offset + 8:offset + 8 + length]
        expected_crc = struct.unpack(">I", payload[offset + 8 + length:end])[0]
        if zlib.crc32(kind + data) & 0xFFFFFFFF != expected_crc:
            raise ContractError("profile reference_png chunk CRC mismatch")
        if kind == b"IHDR":
            if offset != 8 or length != 13:
                raise ContractError("profile reference_png has an invalid IHDR")
            width, height, depth, colour, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", data))
            if (not width or not height or depth != 8 or colour not in (2, 6) or
                    compression != 0 or filtering != 0 or interlace != 0):
                raise ContractError("profile reference_png uses an unsupported format")
        elif kind == b"IEND":
            if length != 0 or end != len(payload):
                raise ContractError("profile reference_png has an invalid IEND")
            seen_iend = True
            break
        offset = end
    if width is None or height is None or not seen_iend:
        raise ContractError("profile reference_png is incomplete")
    return int(width), int(height)


def validate_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError("configuration must be a JSON object")
    if set(value) != CONFIG_FIELDS:
        raise ContractError("configuration has unexpected or missing fields")
    if value.get("schema_version") != SCHEMA_VERSION or value.get("kind") != CONFIG_KIND:
        raise ContractError("unsupported resolution-matrix configuration schema")
    validated = dict(value)
    for field in PATH_FIELDS:
        validated[field] = str(_regular_file(value.get(field), field))
    for field in DIRECTORY_FIELDS:
        validated[field] = str(_existing_directory(value.get(field), field))
    if Path(validated["prompt_file"]).stat().st_size > 1024 * 1024:
        raise ContractError("prompt_file exceeds the producer's 1 MiB limit")
    validated["prompt_sha256"] = hashlib.sha256(
        Path(validated["prompt_file"]).read_bytes()).hexdigest()
    profiles = value.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != set(PROFILE_DIMENSIONS):
        raise ContractError("profiles must define exactly 240, 480, and 720")
    validated_profiles: dict[str, dict[str, Any]] = {}
    for profile, dimensions in PROFILE_DIMENSIONS.items():
        record = profiles.get(profile)
        if not isinstance(record, dict) or set(record) != PROFILE_CONFIG_FIELDS:
            raise ContractError(f"profiles.{profile} has unexpected or missing fields")
        timeout = record.get("timeout_seconds")
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or
                not 1 <= float(timeout) <= 86400):
            raise ContractError(
                f"profiles.{profile}.timeout_seconds must be between 1 and 86400")
        reference = _regular_file(record.get("reference_png"),
                                  f"profiles.{profile}.reference_png")
        if png_dimensions(reference) != dimensions:
            width, height = dimensions
            raise ContractError(
                f"profiles.{profile}.reference_png must be exactly {width}x{height}; "
                "stretching is forbidden")
        validated_profiles[profile] = {
            "reference_png": str(reference),
            "reference_png_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
            "timeout_seconds": timeout,
        }
    validated["profiles"] = validated_profiles
    return validated


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    source = _regular_file(str(path.resolve()), "private configuration")
    if source.stat().st_size > 1024 * 1024:
        raise ContractError("private configuration exceeds the 1 MiB limit")
    payload = source.read_bytes()
    try:
        value = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("private configuration is not valid UTF-8 JSON") from error
    return validate_config(value), hashlib.sha256(payload).hexdigest()


def profile_contract(profile: str, reference_png_sha256: str = "0" * 64,
                     timeout_seconds: int | float = 7200,
                     prompt_sha256: str = "0" * 64) -> dict[str, Any]:
    if profile not in PROFILE_DIMENSIONS:
        raise ContractError(f"unsupported resolution profile: {profile}")
    width, height = PROFILE_DIMENSIONS[profile]
    contract = {
        "profile": profile,
        "label": f"{profile}p",
        "output": {"width": width, "height": height},
        "render": {"width": width, "height": height},
        "grid": GRID,
        "frames": FRAMES,
        "fps": FPS,
        "steps": STEPS,
        "layers": LAYERS,
        "seed": SEED,
        "latent": {"width": width // 16, "height": height // 16},
        "patch_tokens": width * height // 1024,
        "scaling": "none",
        "reference_png_sha256": reference_png_sha256,
        "prompt_sha256": prompt_sha256,
        "timeout_seconds": timeout_seconds,
    }
    validate_profile_contract(contract)
    return contract


def validate_profile_contract(value: Any) -> dict[str, Any]:
    fields = {"profile", "label", "output", "render", "grid", "frames", "fps",
              "steps", "layers", "seed", "latent", "patch_tokens", "scaling",
              "reference_png_sha256", "prompt_sha256", "timeout_seconds"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ContractError("profile contract has unexpected or missing fields")
    profile = value.get("profile")
    if profile not in PROFILE_DIMENSIONS or value.get("label") != f"{profile}p":
        raise ContractError("profile identity is invalid")
    width, height = PROFILE_DIMENSIONS[profile]
    dimensions = {"width": width, "height": height}
    if value.get("output") != dimensions or value.get("render") != dimensions:
        raise ContractError("render dimensions must equal the fixed output dimensions")
    if (width % GRID or height % GRID or width * height > MAX_PIXELS or
            value.get("grid") != GRID):
        raise ContractError("profile dimensions must use the H3 32-pixel grid")
    if value.get("frames") != FRAMES or value.get("fps") != FPS:
        raise ContractError("profile must use the 124-frame/24fps contract")
    if (value.get("steps") != STEPS or value.get("layers") != LAYERS or
            value.get("seed") != SEED):
        raise ContractError("profile must use steps=2, layers=50, and seed=42")
    if (value.get("latent") != {"width": width // 16, "height": height // 16} or
            value.get("patch_tokens") != width * height // 1024):
        raise ContractError("profile latent/patch geometry is invalid")
    if value.get("scaling") != "none":
        raise ContractError("resolution-matrix output scaling or stretching is forbidden")
    digest = value.get("reference_png_sha256")
    if (not isinstance(digest, str) or len(digest) != 64 or
            any(character not in "0123456789abcdef" for character in digest)):
        raise ContractError("profile reference_png_sha256 is invalid")
    prompt_digest = value.get("prompt_sha256")
    if (not isinstance(prompt_digest, str) or len(prompt_digest) != 64 or
            any(character not in "0123456789abcdef" for character in prompt_digest)):
        raise ContractError("profile prompt_sha256 is invalid")
    timeout = value.get("timeout_seconds")
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or
            not 1 <= float(timeout) <= 86400):
        raise ContractError("profile timeout_seconds is invalid")
    return value


def build_comfy_command(config: dict[str, Any], output_root: Path,
                        profile: str) -> list[str]:
    profile_config = config["profiles"][profile]
    contract = profile_contract(profile, profile_config["reference_png_sha256"],
                                profile_config["timeout_seconds"])
    profile_root = output_root / contract["label"] / "comfyui"
    width = contract["render"]["width"]
    height = contract["render"]["height"]
    return [
        config["python_executable"], str(PRODUCER),
        "--comfy-main", config["comfy_main"],
        "--t8-sampling", config["t8_sampling"],
        "--t8-nodes", config["t8_nodes"],
        "--comfy-attention", config["comfy_attention"],
        "--model-file", config["model_file"],
        "--clip-file", config["clip_file"],
        "--video-vae-file", config["video_vae_file"],
        "--audio-vae-file", config["audio_vae_file"],
        "--runtime-dir", str(profile_root / "runtime"),
        "--reference-png", profile_config["reference_png"],
        "--prompt-file", config["prompt_file"],
        "--output-media", str(profile_root / "output.mp4"),
        "--scheduler-trace", str(profile_root / "scheduler-trace.json"),
        "--attention-trace", str(profile_root / "attention-trace.json"),
        "--timeout", str(profile_config["timeout_seconds"]),
        "--width", str(width), "--height", str(height),
        "--frames", str(FRAMES), "--steps", str(STEPS),
        "--resolution-matrix",
    ]


def build_h3_command(config: dict[str, Any], output_root: Path,
                     profile: str) -> list[str]:
    profile_config = config["profiles"][profile]
    contract = profile_contract(profile, profile_config["reference_png_sha256"],
                                profile_config["timeout_seconds"])
    profile_root = output_root / contract["label"] / "h3cspeed"
    width = str(contract["render"]["width"])
    height = str(contract["render"]["height"])
    return [
        config["powershell_executable"], "-NoProfile", "-NonInteractive", "-File",
        str(PROJECT_ROOT / "scripts" / "run_perf007_h3.ps1"),
        "-ModelRoot", config["h3_model_root"],
        "-ComfyUIRoot", config["h3_comfy_root"],
        "-TextEncoder", config["h3_text_encoder"],
        "-PromptFile", config["prompt_file"],
        "-FirstFrame", profile_config["reference_png"],
        "-Output", str(profile_root / "output.mp4"),
        "-SidecarPath", str(profile_root / "conditioning.h3c"),
        "-BinaryPath", config["h3_binary"],
        "-ProfileDir", str(profile_root / "profile"),
        "-Width", width, "-Height", height,
        "-Frames", str(FRAMES), "-Steps", str(STEPS),
        "-Layers", str(LAYERS), "-Seed", str(SEED), "-LayerMajor", "-AsyncRefill",
        "-DitPrefetch", "-ResolutionMatrix",
    ]


def build_plan(config: dict[str, Any], config_digest: str, output_root: Path,
               profiles: Iterable[str]) -> dict[str, Any]:
    selected = list(profiles)
    if not selected or len(selected) != len(set(selected)):
        raise ContractError("profiles must be a non-empty unique selection")
    contracts = [
        profile_contract(profile,
                         config["profiles"][profile]["reference_png_sha256"],
                         config["profiles"][profile]["timeout_seconds"],
                         config["prompt_sha256"])
        for profile in selected
    ]
    plan = {
        "schema_version": SCHEMA_VERSION,
        "kind": PLAN_KIND,
        "mode": "dry-run",
        "status": "NOT_RUN",
        "config_sha256": config_digest,
        "contracts": contracts,
        "commands": [item for contract in contracts for item in (
            {"profile": contract["profile"], "engine": "h3cspeed",
             "argv": build_h3_command(config, output_root, contract["profile"])},
            {"profile": contract["profile"], "engine": "comfyui",
             "argv": build_comfy_command(config, output_root, contract["profile"])},
        )],
    }
    validate_plan(plan, output_root)
    return plan


def _flag_value(argv: list[str], flag: str) -> str:
    if argv.count(flag) != 1:
        raise ContractError(f"command must contain exactly one {flag}")
    index = argv.index(flag)
    if index + 1 >= len(argv):
        raise ContractError(f"command is missing the value for {flag}")
    return argv[index + 1]


def validate_plan(value: Any, output_root: Path) -> dict[str, Any]:
    fields = {"schema_version", "kind", "mode", "status", "config_sha256",
              "contracts", "commands"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ContractError("plan has unexpected or missing fields")
    if (value.get("schema_version") != SCHEMA_VERSION or value.get("kind") != PLAN_KIND or
            value.get("mode") != "dry-run" or value.get("status") != "NOT_RUN"):
        raise ContractError("plan cannot claim execution or use another schema")
    digest = value.get("config_sha256")
    if (not isinstance(digest, str) or len(digest) != 64 or
            any(character not in "0123456789abcdef" for character in digest)):
        raise ContractError("plan config_sha256 is invalid")
    contracts = value.get("contracts")
    commands = value.get("commands")
    if (not isinstance(contracts, list) or not isinstance(commands, list) or
            len(commands) != 2 * len(contracts) or not contracts):
        raise ContractError("plan must pair each contract with H3 and Comfy commands")
    for contract in contracts:
        validate_profile_contract(contract)
    if len({contract["profile"] for contract in contracts}) != len(contracts):
        raise ContractError("plan profile contracts must be unique")
    expected_order = [(contract["profile"], engine) for contract in contracts
                      for engine in ("h3cspeed", "comfyui")]
    actual_order = [(command.get("profile"), command.get("engine"))
                    for command in commands if isinstance(command, dict)]
    if actual_order != expected_order:
        raise ContractError("plan command order must be H3 then Comfy for every profile")
    contracts_by_profile = {contract["profile"]: contract for contract in contracts}
    for command in commands:
        if (not isinstance(command, dict) or
                set(command) != {"profile", "engine", "argv"} or
                command.get("engine") not in ("h3cspeed", "comfyui") or
                not isinstance(command.get("argv"), list) or
                not all(isinstance(item, str) for item in command["argv"])):
            raise ContractError("plan command is invalid")
        contract = contracts_by_profile[command["profile"]]
        argv = command["argv"]
        width = str(contract["render"]["width"])
        height = str(contract["render"]["height"])
        profile_root = output_root / contract["label"] / command["engine"]
        if command["engine"] == "comfyui":
            if (len(argv) < 2 or argv[1] != str(PRODUCER)):
                raise ContractError("Comfy matrix command has an unbound producer")
            planned_python = _regular_file(argv[0], "planned Python executable")
            if (str(planned_python) != argv[0] or
                    argv.count("--resolution-matrix") != 1 or
                    _flag_value(argv, "--width") != width or
                    _flag_value(argv, "--height") != height or
                    _flag_value(argv, "--frames") != str(FRAMES) or
                    _flag_value(argv, "--steps") != str(STEPS) or
                    _flag_value(argv, "--timeout") != str(contract["timeout_seconds"])):
                raise ContractError("Comfy matrix command does not match its contract")
            expected_outputs = {
                "--runtime-dir": profile_root / "runtime",
                "--output-media": profile_root / "output.mp4",
                "--scheduler-trace": profile_root / "scheduler-trace.json",
                "--attention-trace": profile_root / "attention-trace.json",
            }
            forbidden = ("--render-width", "--render-height", "--resize", "--scale",
                         "--stretch")
        else:
            expected_prefix = [
                argv[0], "-NoProfile", "-NonInteractive", "-File",
                str(PROJECT_ROOT / "scripts" / "run_perf007_h3.ps1"),
            ]
            planned_powershell = _regular_file(argv[0], "planned PowerShell executable")
            if (str(planned_powershell) != argv[0] or argv[:5] != expected_prefix or
                    argv.count("-ResolutionMatrix") != 1 or
                    _flag_value(argv, "-Width") != width or
                    _flag_value(argv, "-Height") != height or
                    _flag_value(argv, "-Frames") != str(FRAMES) or
                    _flag_value(argv, "-Steps") != str(STEPS) or
                    _flag_value(argv, "-Layers") != "50" or
                    _flag_value(argv, "-Seed") != "42" or
                    any(argv.count(flag) != 1 for flag in
                        ("-LayerMajor", "-AsyncRefill", "-DitPrefetch"))):
                raise ContractError("H3 matrix command does not match its contract")
            expected_outputs = {
                "-Output": profile_root / "output.mp4",
                "-SidecarPath": profile_root / "conditioning.h3c",
                "-ProfileDir": profile_root / "profile",
            }
            # The native runner defaults render dimensions to the requested
            # output dimensions.  Keep the no-resize contract structural by
            # omitting render overrides from the matrix command entirely.
            forbidden = ("-RenderWidth", "-RenderHeight", "--render-width",
                         "--render-height", "--resize", "--scale", "--stretch")
        for flag, expected in expected_outputs.items():
            if Path(_flag_value(argv, flag)) != expected:
                raise ContractError(f"matrix command has an unsafe {flag} destination")
        if any(flag in argv for flag in forbidden):
            raise ContractError("matrix command must not request output scaling or stretching")
    for contract in contracts:
        paired = {command["engine"]: command["argv"] for command in commands
                  if command["profile"] == contract["profile"]}
        h3_reference = Path(_flag_value(paired["h3cspeed"], "-FirstFrame"))
        comfy_reference = Path(_flag_value(paired["comfyui"], "--reference-png"))
        if h3_reference != comfy_reference:
            raise ContractError("H3 and Comfy must use the same exact reference PNG")
        if hashlib.sha256(h3_reference.read_bytes()).hexdigest() != \
                contract["reference_png_sha256"]:
            raise ContractError("paired reference PNG no longer matches the plan contract")
        if (_flag_value(paired["h3cspeed"], "-PromptFile") !=
                _flag_value(paired["comfyui"], "--prompt-file")):
            raise ContractError("H3 and Comfy must use the same prompt file")
        prompt_file = _regular_file(
            _flag_value(paired["h3cspeed"], "-PromptFile"), "paired prompt file")
        if hashlib.sha256(prompt_file.read_bytes()).hexdigest() != contract["prompt_sha256"]:
            raise ContractError("paired prompt file no longer matches the plan contract")
    return value


def _validate_output_boundary(path: Path, config: dict[str, Any]) -> None:
    _reject_link_chain(path, "private output directory")
    if not path.is_absolute():
        raise ContractError("private output directory must be absolute")
    _outside_project(path, "private output directory")
    protected_roots = {
        Path(config["comfy_main"]).parent.resolve(),
        Path(config["h3_comfy_root"]).resolve(),
        Path(config["h3_model_root"]).resolve(),
        *(Path(config[field]).parent.resolve() for field in (
            "model_file", "clip_file", "video_vae_file", "audio_vae_file")),
    }
    for protected in protected_roots:
        try:
            path.resolve(strict=False).relative_to(protected)
        except ValueError:
            continue
        raise ContractError(
            "private output directory must be outside ComfyUI and model roots")


def _prepare_private_output(path: Path, profiles: Iterable[str],
                            config: dict[str, Any]) -> Path:
    _validate_output_boundary(path, config)
    if path.exists() or path.is_symlink():
        raise ContractError("private output directory must be new (no-clobber)")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ContractError("private output parent must be an existing unlinked directory")
    path.mkdir(mode=0o700)
    if os.name == "nt":
        _restrict_windows_private_acl(path)
    else:
        path.chmod(0o700)
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise ContractError("private output directory permissions are too broad")
    for profile in profiles:
        child = path / f"{profile}p"
        child.mkdir(mode=0o700)
        if os.name != "nt":
            child.chmod(0o700)
        for engine in ("h3cspeed", "comfyui"):
            engine_child = child / engine
            engine_child.mkdir(mode=0o700)
            if os.name != "nt":
                engine_child.chmod(0o700)
    return path


_WINDOWS_PRIVATE_ACL_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$target = $env:H3CSPEED_PRIVATE_OUTPUT_DACL_PATH
if ([string]::IsNullOrWhiteSpace($target)) { exit 11 }
$current = [Security.Principal.WindowsIdentity]::GetCurrent().User
$icacls = $env:H3CSPEED_PRIVATE_OUTPUT_ICACLS_PATH
if ([string]::IsNullOrWhiteSpace($icacls)) { exit 14 }
& $icacls $target '/inheritance:r' '/grant:r' `
    "*$($current.Value):(OI)(CI)F" '/Q' | Out-Null
if ($LASTEXITCODE -ne 0) { exit 15 }
$before = Get-Acl -LiteralPath $target
$otherAllows = @($before.Access | Where-Object {
    $_.AccessControlType -eq [Security.AccessControl.AccessControlType]::Allow -and
    $_.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value -ne
        $current.Value
})
foreach ($entry in $otherAllows) {
    $sid = $entry.IdentityReference.Translate(
        [Security.Principal.SecurityIdentifier]).Value
    & $icacls $target '/remove:g' "*$sid" '/Q' | Out-Null
    if ($LASTEXITCODE -ne 0) { exit 16 }
}
$actual = Get-Acl -LiteralPath $target
if (-not $actual.AreAccessRulesProtected) { exit 12 }
$unexpected = @($actual.Access | Where-Object {
    $_.AccessControlType -eq [Security.AccessControl.AccessControlType]::Allow -and
    $_.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value -ne
        $current.Value
})
if ($unexpected.Count -ne 0) { exit 13 }
"""

_WINDOWS_VERIFY_PRIVATE_ACL_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$target = $env:H3CSPEED_PRIVATE_OUTPUT_DACL_PATH
if ([string]::IsNullOrWhiteSpace($target)) { exit 11 }
$current = [Security.Principal.WindowsIdentity]::GetCurrent().User
$actual = Get-Acl -LiteralPath $target
if (-not $actual.AreAccessRulesProtected) { exit 12 }
$unexpected = @($actual.Access | Where-Object {
    $_.AccessControlType -eq [Security.AccessControl.AccessControlType]::Allow -and
    $_.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value -ne
        $current.Value
})
if ($unexpected.Count -ne 0) { exit 13 }
$currentFull = @($actual.Access | Where-Object {
    $_.AccessControlType -eq [Security.AccessControl.AccessControlType]::Allow -and
    $_.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value -eq
        $current.Value -and
    ($_.FileSystemRights -band [Security.AccessControl.FileSystemRights]::FullControl) -eq
        [Security.AccessControl.FileSystemRights]::FullControl -and
    ($_.InheritanceFlags -band [Security.AccessControl.InheritanceFlags]::ContainerInherit) -ne 0 -and
    ($_.InheritanceFlags -band [Security.AccessControl.InheritanceFlags]::ObjectInherit) -ne 0
})
if ($currentFull.Count -eq 0) { exit 17 }
"""


def _windows_system_directory() -> Path:
    buffer = ctypes.create_unicode_buffer(32768)
    length = ctypes.windll.kernel32.GetSystemDirectoryW(buffer, len(buffer))
    if not length or length >= len(buffer):
        raise ContractError("could not resolve the Windows system directory")
    return Path(buffer.value)


def _run_system_acl_script(path: Path, script: str) -> None:
    # Dry planning must never execute a config-selected producer.  ACL setup is
    # therefore bound to the OS-owned Windows PowerShell under System32.
    system_directory = _windows_system_directory()
    windows_powershell = system_directory / "WindowsPowerShell" / "v1.0"
    powershell = _regular_file(
        str(windows_powershell / "powershell.exe"),
        "system Windows PowerShell executable")
    environment = dict(os.environ)
    environment["H3CSPEED_PRIVATE_OUTPUT_DACL_PATH"] = str(path)
    environment["H3CSPEED_PRIVATE_OUTPUT_ICACLS_PATH"] = str(
        _regular_file(str(system_directory / "icacls.exe"), "icacls executable"))
    # A parent pwsh process may replace PSModulePath with PowerShell 7 modules,
    # which prevents the OS-owned Windows PowerShell from loading Set-Acl.
    environment["PSModulePath"] = str(windows_powershell / "Modules")
    completed = subprocess.run(
        [str(powershell), "-NoProfile", "-NonInteractive", "-Command",
         script],
        capture_output=True, text=True, check=False, shell=False, timeout=30,
        env=environment,
    )
    if completed.returncode != 0:
        raise ContractError("could not establish a restrictive private-output DACL")


def _restrict_windows_private_acl(path: Path) -> None:
    """Apply and verify a current-user-only inheritable DACL, or fail closed."""
    _run_system_acl_script(path, _WINDOWS_PRIVATE_ACL_SCRIPT)


def _verify_windows_private_acl(path: Path) -> None:
    """Read back the existing private DACL without changing it."""
    _run_system_acl_script(path, _WINDOWS_VERIFY_PRIVATE_ACL_SCRIPT)


def _validate_existing_private_output(path: Path, profiles: Iterable[str],
                                      config: dict[str, Any]) -> Path:
    _validate_output_boundary(path, config)
    output = _existing_directory(str(path), "private output directory")
    if os.name == "nt":
        _verify_windows_private_acl(output)
    elif stat.S_IMODE(output.stat().st_mode) & 0o077:
        raise ContractError("private output directory permissions are too broad")
    for profile in profiles:
        profile_root = _existing_directory(
            str(output / f"{profile}p"), f"{profile}p private output")
        roots = (profile_root, *(
            _existing_directory(str(profile_root / engine),
                                f"{profile}p {engine} private output")
            for engine in ("h3cspeed", "comfyui")))
        if os.name != "nt" and any(
                stat.S_IMODE(root.stat().st_mode) & 0o077 for root in roots):
            raise ContractError(f"{profile}p private output permissions are too broad")
    return output


def _run_child(argv: list[str], stream: Any, timeout_seconds: float) -> int:
    """Run one producer and terminate its complete process tree on timeout."""
    options: dict[str, Any] = {
        "stdout": stream, "stderr": subprocess.STDOUT, "shell": False,
    }
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    process = subprocess.Popen(argv, **options)
    try:
        return int(process.wait(timeout=timeout_seconds))
    except subprocess.TimeoutExpired as error:
        if os.name == "nt":
            taskkill = _regular_file(
                str(_windows_system_directory() / "taskkill.exe"), "taskkill executable")
            killed = subprocess.run(
                [str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=False, shell=False, timeout=30,
            )
            if killed.returncode != 0:
                raise ContractError("producer process tree could not be terminated") from error
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired as stop_error:
            raise ContractError("producer process tree did not stop after timeout") from stop_error
        raise ContractError("producer process tree timed out and was terminated") from error


def _publish_no_clobber(path: Path, payload: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="xb", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise ContractError(f"refusing to overwrite {path.name}") from error
    except OSError as error:
        raise ContractError(f"could not publish {path.name}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def create_dry_plan(config_path: Path, output_directory: Path,
                    profiles: Iterable[str]) -> tuple[Path, str]:
    selected = list(profiles)
    config, digest = load_config(config_path)
    output = Path(os.path.abspath(output_directory))
    plan = build_plan(config, digest, output, selected)
    _prepare_private_output(output, selected, config)
    destination = output / "resolution-matrix-plan.json"
    payload = canonical_bytes(plan)
    _publish_no_clobber(destination, payload)
    return destination, hashlib.sha256(payload).hexdigest()


def load_existing_plan(path: Path) -> tuple[dict[str, Any], bytes]:
    source = _regular_file(str(path), "existing resolution-matrix plan")
    if source.stat().st_size > 4 * 1024 * 1024:
        raise ContractError("existing resolution-matrix plan exceeds the 4 MiB limit")
    payload = source.read_bytes()
    try:
        value = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("existing resolution-matrix plan is not valid UTF-8 JSON") from error
    if not isinstance(value, dict) or payload != canonical_bytes(value):
        raise ContractError("existing resolution-matrix plan must be canonical JSON")
    return value, payload


def execute_plan(plan: dict[str, Any], output_root: Path, config: dict[str, Any],
                 config_digest: str) -> dict[str, Any]:
    """Execute after explicit opt-in and publish per-engine wall-time evidence."""
    validate_plan(plan, output_root)
    selected = [contract["profile"] for contract in plan["contracts"]]
    expected = build_plan(config, config_digest, output_root, selected)
    if canonical_bytes(plan) != canonical_bytes(expected):
        raise ContractError("execution plan is not bound to the private configuration")
    summary_path = output_root / "resolution-matrix-execution.json"
    if summary_path.exists() or summary_path.is_symlink():
        raise ContractError("execution summary already exists (no-clobber)")
    _reject_link_chain(summary_path, "resolution-matrix execution summary")
    contracts = {contract["profile"]: contract for contract in plan["contracts"]}
    executions: list[dict[str, Any]] = []
    for command in plan["commands"]:
        # Re-hash shared inputs immediately before each child.  A long H3 run
        # must not let a modified reference/prompt reach the paired Comfy run.
        validate_plan(plan, output_root)
        contract = contracts[command["profile"]]
        profile_root = output_root / contract["label"] / command["engine"]
        _existing_directory(str(profile_root), f"{contract['label']} private engine output")
        protected = (
            profile_root / "runtime",
            profile_root / "output.mp4",
            profile_root / "scheduler-trace.json",
            profile_root / "attention-trace.json",
            profile_root / "conditioning.h3c",
            profile_root / "conditioning.h3c.first.png",
            profile_root / "conditioning.h3c.last.png",
            profile_root / "profile",
            profile_root / "producer-private.log",
        )
        if any(path.exists() or path.is_symlink() for path in protected):
            raise ContractError(
                f"{contract['label']} execution target already exists (no-clobber)")
        log_path = profile_root / "producer-private.log"
        _reject_link_chain(log_path, f"{contract['label']} private producer log")
        with log_path.open("xb") as stream:
            started = time.perf_counter()
            returncode = _run_child(
                command["argv"], stream,
                float(contract["timeout_seconds"]) + 900)
            wall_seconds = time.perf_counter() - started
        if returncode != 0:
            raise ContractError(
                f"{contract['label']} producer failed; inspect its private log")
        executions.append({
            "profile": contract["profile"],
            "label": contract["label"],
            "engine": command["engine"],
            "wall_seconds": wall_seconds,
            "returncode": returncode,
        })
    by_profile: list[dict[str, Any]] = []
    for contract in plan["contracts"]:
        profile_runs = [item for item in executions
                        if item["profile"] == contract["profile"]]
        engine_runs = {item["engine"]: item for item in profile_runs}
        if set(engine_runs) != {"h3cspeed", "comfyui"}:
            raise ContractError("execution did not produce both engine timings")
        h3_wall = float(engine_runs["h3cspeed"]["wall_seconds"])
        comfy_wall = float(engine_runs["comfyui"]["wall_seconds"])
        if h3_wall <= 0 or comfy_wall <= 0:
            raise ContractError("engine wall time must be positive")
        by_profile.append({
            "profile": contract["profile"],
            "label": contract["label"],
            "output": contract["output"],
            "engines": {
                engine: {
                    "wall_seconds": float(engine_runs[engine]["wall_seconds"]),
                    "returncode": engine_runs[engine]["returncode"],
                }
                for engine in ("h3cspeed", "comfyui")
            },
            "h3cspeed_over_comfyui_wall_ratio": h3_wall / comfy_wall,
        })
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": EXECUTION_KIND,
        "status": "EXECUTED_UNVERIFIED",
        "plan_sha256": hashlib.sha256(canonical_bytes(plan)).hexdigest(),
        "config_sha256": config_digest,
        "profiles": by_profile,
        "acceptance": {
            "process_completion": "PASS",
            "media_contract": "NOT_RUN",
            "quality_parity": "NOT_RUN",
            "speed_alignment": "OBSERVED_ONLY",
        },
    }
    _publish_no_clobber(summary_path, canonical_bytes(summary))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path,
                        help="private resolution-matrix JSON configuration")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="new private directory outside the source tree")
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILE_DIMENSIONS),
                        default=list(PROFILE_DIMENSIONS))
    parser.add_argument("--execute", action="store_true",
                        help="explicitly launch the GPU producer after dry validation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = Path(os.path.abspath(args.output_dir))
        plan_path = output / "resolution-matrix-plan.json"
        if args.execute and output.exists():
            # A prior dry-run owns an existing output root.  Never recreate it
            # (or overwrite its plan); re-read and re-bind the existing plan.
            if not plan_path.is_file():
                raise ContractError(
                    "--execute requires an existing dry-run resolution-matrix-plan.json")
            plan, plan_payload = load_existing_plan(plan_path)
            digest = hashlib.sha256(plan_payload).hexdigest()
            config, current_digest = load_config(args.config)
            _validate_existing_private_output(output, args.profiles, config)
            fresh = build_plan(config, current_digest, output, args.profiles)
            if plan_payload != canonical_bytes(fresh):
                raise ContractError(
                    "existing resolution-matrix plan does not match the current config")
            execute_plan(fresh, output, config, current_digest)
        else:
            plan_path, digest = create_dry_plan(args.config, output, args.profiles)
            if args.execute:
                # Fresh one-shot execution remains supported for callers that
                # explicitly opt in without a preceding dry-run command.
                config, current_digest = load_config(args.config)
                _validate_existing_private_output(output, args.profiles, config)
                plan, plan_payload = load_existing_plan(plan_path)
                fresh = build_plan(config, current_digest, output, args.profiles)
                if plan_payload != canonical_bytes(fresh):
                    raise ContractError("resolution-matrix dry plan changed before execution")
                execute_plan(fresh, output, config, current_digest)
    except (ContractError, OSError) as error:
        print(f"resolution-matrix dry plan failed: {error}", file=sys.stderr)
        return 2
    print(f"resolution-matrix dry plan created: {digest}")
    print(f"GPU execution status: {'EXECUTED_UNVERIFIED' if args.execute else 'NOT_RUN'}")
    if args.execute:
        print(f"timing summary: {(output / 'resolution-matrix-execution.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
