#!/usr/bin/env python3
"""Run one isolated PERF-002C 22-frame/2-step engine smoke adapter.

The command configuration is private input. Raw argv, prompt, paths,
environment and child output are never copied into the machine-readable result.
An individual smoke may pass, but this script never claims matched A/B or speed
parity; that requires both engines and the full 124-frame trial matrix.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import math
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_perf002_ab import (  # noqa: E402
    ContractError, ENGINES, _reject_link_chain, canonical_bytes, load_input, png_dimensions,
    publish_json, safe_existing_directory, safe_output_directory, safe_regular,
    sha256_bytes, sha256_file, validate_media_contract,
)


SCHEMA_VERSION = 1
KIND = "h3cspeed.perf002.smoke"
ALLOWED_ENVIRONMENT = {
    "CUDA_DEVICE_ORDER", "H3_CUDA_ATTENTION", "H3_CUDA_DEVICE",
    "H3_CUDA_LOW_VRAM", "H3_CUDA_OFFLOAD", "H3_CUDA_PINNED_HOST_MIB",
    "H3_CUDA_RAM_CACHE_MIB", "H3_CUDA_STAGING_MIB", "H3_CUDA_TF32",
    "H3_CUDA_VRAM_BUDGET_MIB", "H3_CUDA_WEIGHT_CACHE_MIB", "H3_PROFILE",
    "H3_PROFILE_JSON_DIR", "PYTHONIOENCODING", "PYTHONUTF8",
    "H3CSPEED_PERF002_ATTENTION_TRACE", "H3CSPEED_PERF002_SCHEDULER_TRACE",
    "H3CSPEED_TEXT_EMBEDDING", "H3CSPEED_TEXT_ENCODER_SHA256", "H3_FFMPEG",
}
BASE_ENVIRONMENT = {
    "APPDATA", "COMSPEC", "LOCALAPPDATA", "PATH", "PATHEXT", "PROGRAMDATA",
    "SYSTEMDRIVE", "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE", "WINDIR",
}
COMFY_INPUT_FLAGS = {
    "--comfy-main": "comfy_main",
    "--t8-sampling": "t8_sampling",
    "--t8-nodes": "t8_nodes",
    "--comfy-attention": "comfy_attention",
    "--model-file": "model_file",
    "--clip-file": "clip_file",
    "--video-vae-file": "video_vae_file",
    "--audio-vae-file": "audio_vae_file",
}
H3_INPUT_BINDINGS = {
    "-d": ("config", "model_root"),
    "-p": ("config", "prompt_file"),
    "--first-frame": ("fixture", "reference_png"),
    "-o": ("config", "output_media"),
    "H3CSPEED_TEXT_EMBEDDING": ("conditioning", "sidecar"),
    "H3CSPEED_TEXT_ENCODER_SHA256": ("models", "qwen"),
    "H3_FFMPEG": ("engines", "ffmpeg"),
    "H3CSPEED_PERF002_SCHEDULER_TRACE": ("config", "scheduler_trace"),
    "H3CSPEED_PERF002_ATTENTION_TRACE": ("config", "attention_trace"),
}
H3_VALUE_FLAGS = {
    "-d", "-p", "--first-frame", "-o", "--width", "--height", "--frames",
    "--steps", "--layers", "--reuse", "--core-reuse", "--seed",
}
H3_SWITCH_FLAGS = {"--ssd-streaming"}
H3_FORBIDDEN_FLAGS = {"--last-frame", "--render-width", "--render-height"}
H3_MODEL_PATHS = {
    "fl2va": Path("FL2VA/transformer/minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
    "qwen": Path("FL2VA/text_encoder/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
    "video_vae": Path("FL2VA/video_vae/source/minimax_h3_video_vae_fp16.safetensors"),
    "audio_vae": Path("FL2VA/audio_vae/minimax_h3_audio_vae_fp32.safetensors"),
    "transformer_config": Path("FL2VA/transformer/config.json"),
    "tokenizer": Path("FL2VA/tokenizer/tokenizer.json"),
    "video_vae_config": Path("FL2VA/video_vae/config.json"),
    "audio_vae_config": Path("FL2VA/audio_vae/config.json"),
}
H3_SIDECAR_HEADER = struct.Struct("<8sIIQQQIIQQQ32s24s")
H3_SIDECAR_RECIPE = b"h3cspeed-conditioning-v2"
H3_SIDECAR_MAX_BYTES = 64 * 1024 * 1024
BASE_CONFIG_FIELDS = {
    "schema_version", "engine", "argv", "environment", "output_media",
    "scheduler_trace", "attention_trace", "protected_roots", "reference_png",
    "prompt_file", "bindings", "command_artifacts",
}


def _required_config_fields(engine: str) -> set[str]:
    if engine == "h3cspeed":
        return BASE_CONFIG_FIELDS | {"model_root", "command_inputs"}
    return BASE_CONFIG_FIELDS | {"runtime_dir", "command_inputs"}


if os.name == "nt":
    class _JobBasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]


    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]


    class _JobExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JobBasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]


    class _StartupInfoW(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR),
            ("lpDesktop", wintypes.LPWSTR), ("lpTitle", wintypes.LPWSTR),
            ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
            ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD),
            ("dwXCountChars", wintypes.DWORD), ("dwYCountChars", wintypes.DWORD),
            ("dwFillAttribute", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
            ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD),
            ("lpReserved2", ctypes.POINTER(ctypes.c_ubyte)),
            ("hStdInput", wintypes.HANDLE), ("hStdOutput", wintypes.HANDLE),
            ("hStdError", wintypes.HANDLE),
        ]


    class _ProcessInformation(ctypes.Structure):
        _fields_ = [
            ("hProcess", wintypes.HANDLE), ("hThread", wintypes.HANDLE),
            ("dwProcessId", wintypes.DWORD), ("dwThreadId", wintypes.DWORD),
        ]


class _WindowsJob:
    """Own a Windows process tree with kill-on-close semantics."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise RuntimeError("Windows Job Objects are unavailable")
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32 = kernel32
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        self._handle = kernel32.CreateJobObjectW(None, None)
        if not self._handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        information = _JobExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = 0x00002000
        if not kernel32.SetInformationJobObject(
                self._handle, 9, ctypes.byref(information), ctypes.sizeof(information)):
            error = ctypes.get_last_error()
            self.close()
            raise OSError(error, "SetInformationJobObject failed")

    def assign_handle(self, process_handle: wintypes.HANDLE) -> None:
        if not self._kernel32.AssignProcessToJobObject(self._handle, process_handle):
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


def _require_windows_process_stopped(wait_result: int, message: str) -> None:
    if wait_result != 0:
        raise ContractError(message)


def _run_engine_windows(argv: list[str], stream: Any, environment: dict[str, str],
                        timeout_seconds: int) -> int:
    """Create suspended, assign to a kill-on-close job, then execute."""
    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateProcessW.argtypes = [
        wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p, ctypes.c_void_p,
        wintypes.BOOL, wintypes.DWORD, ctypes.c_void_p, wintypes.LPCWSTR,
        ctypes.POINTER(_StartupInfoW), ctypes.POINTER(_ProcessInformation),
    ]
    kernel32.CreateProcessW.restype = wintypes.BOOL
    kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
    kernel32.ResumeThread.restype = wintypes.DWORD
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline(argv))
    environment_block = ctypes.create_unicode_buffer(
        "\0".join(f"{key}={value}" for key, value in sorted(
            environment.items(), key=lambda item: item[0].upper())) + "\0\0")
    output_handle = wintypes.HANDLE(msvcrt.get_osfhandle(stream.fileno()))
    os.set_handle_inheritable(int(output_handle.value), True)
    startup = _StartupInfoW()
    startup.cb = ctypes.sizeof(startup)
    startup.dwFlags = 0x00000100
    startup.hStdInput = wintypes.HANDLE(0)
    startup.hStdOutput = output_handle
    startup.hStdError = output_handle
    information = _ProcessInformation()
    job = _WindowsJob()
    try:
        created = kernel32.CreateProcessW(
            argv[0], command_line, None, None, True,
            0x00000004 | 0x00000200 | 0x00000400,
            ctypes.cast(environment_block, ctypes.c_void_p), None,
            ctypes.byref(startup), ctypes.byref(information),
        )
        if not created:
            raise OSError(ctypes.get_last_error(), "CreateProcessW failed")
        try:
            try:
                job.assign_handle(information.hProcess)
            except Exception:
                if not kernel32.TerminateProcess(information.hProcess, 1):
                    raise OSError(
                        ctypes.get_last_error(),
                        "AssignProcessToJobObject and TerminateProcess failed",
                    )
                _require_windows_process_stopped(
                    kernel32.WaitForSingleObject(information.hProcess, 30000),
                    "unassigned suspended engine process could not be stopped",
                )
                raise
            if kernel32.ResumeThread(information.hThread) == 0xFFFFFFFF:
                raise OSError(ctypes.get_last_error(), "ResumeThread failed")
            wait_result = kernel32.WaitForSingleObject(
                information.hProcess, int(timeout_seconds * 1000))
            if wait_result == 0x00000102:
                job.close()
                stopped = kernel32.WaitForSingleObject(information.hProcess, 30000)
                _require_windows_process_stopped(
                    stopped,
                    "engine smoke timed out and its process group could not be stopped",
                )
                raise ContractError(
                    "engine smoke timed out and its process group was stopped")
            if wait_result != 0:
                raise OSError(ctypes.get_last_error(), "WaitForSingleObject failed")
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(information.hProcess,
                                               ctypes.byref(exit_code)):
                raise OSError(ctypes.get_last_error(), "GetExitCodeProcess failed")
            return int(exit_code.value)
        finally:
            kernel32.CloseHandle(information.hThread)
            kernel32.CloseHandle(information.hProcess)
    finally:
        os.set_handle_inheritable(int(output_handle.value), False)
        job.close()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be an object")
    return value


def _load_json(path: Path, label: str, maximum: int = 4 * 1024 * 1024) -> Any:
    source = safe_regular(path, label)
    if source.stat().st_size > maximum:
        raise ContractError(f"{label} exceeds its size limit")
    return json.loads(source.read_text(encoding="utf-8"))


def _hash_binding_group(labels: dict[str, str], paths: Any,
                        section: str) -> dict[str, str]:
    binding_map = _mapping(paths, f"bindings.{section}")
    if set(binding_map) != set(labels):
        raise ContractError(f"bindings.{section} does not match the manifest")
    actual: dict[str, str] = {}
    for name in sorted(labels):
        source = safe_regular(Path(binding_map[name]), f"bindings.{section}.{name}")
        actual[name] = sha256_file(source)
    if actual != labels:
        raise ContractError(f"bindings.{section} content does not match the manifest")
    return actual


def verify_bound_inputs(manifest: dict[str, Any], config: dict[str, Any],
                        engine: str) -> dict[str, str]:
    reference = safe_regular(Path(config["reference_png"]), "reference PNG")
    prompt = safe_regular(Path(config["prompt_file"]), "prompt file")
    if png_dimensions(reference) != (864, 480):
        raise ContractError("smoke reference PNG must be native 864x480")
    if sha256_file(reference) != manifest["fixture"]["sha256"]:
        raise ContractError("reference PNG changed after manifest creation")
    if sha256_file(prompt) != manifest["prompt_sha256"]:
        raise ContractError("private prompt changed after manifest creation")
    bindings = _mapping(config.get("bindings"), "bindings")
    fingerprints: dict[str, str] = {
        "reference_png": sha256_file(reference), "prompt": sha256_file(prompt),
    }
    for section in ("models", "conditioning", "engines"):
        section_bindings = _mapping(bindings.get(section), f"bindings.{section}")
        current = _hash_binding_group(
            manifest[section][engine], section_bindings.get(engine),
            f"{section}.{engine}",
        )
        fingerprints.update({f"{section}.{name}": digest
                             for name, digest in current.items()})
    return fingerprints


def _private_environment(overrides: Any) -> dict[str, str]:
    requested = _mapping(overrides, "environment")
    unknown = set(requested) - ALLOWED_ENVIRONMENT
    if unknown:
        raise ContractError("command environment contains non-allowlisted keys")
    environment = {key: value for key, value in os.environ.items()
                   if key.upper() in BASE_ENVIRONMENT}
    for key, value in requested.items():
        if not isinstance(value, str) or "\x00" in value:
            raise ContractError("environment values must be strings without NUL")
        environment[key] = value
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    return environment


def _single_h3_flag(argv: list[str], flag: str) -> tuple[int, str]:
    indices = [index for index, value in enumerate(argv) if value == flag]
    if len(indices) != 1:
        raise ContractError(f"h3cspeed flag {flag} must occur exactly once")
    index = indices[0]
    if index + 1 >= len(argv) or not argv[index + 1]:
        raise ContractError(f"h3cspeed flag {flag} must have one value")
    return index, argv[index + 1]


def _validate_h3_sidecar(path: Path, prompt: bytes, reference_sha256: str,
                         qwen_sha256: str) -> None:
    size = path.stat().st_size
    if size < H3_SIDECAR_HEADER.size or size > H3_SIDECAR_MAX_BYTES:
        raise ContractError("h3cspeed conditioning sidecar size is invalid")
    payload = path.read_bytes()
    if len(payload) != size:
        raise ContractError("h3cspeed conditioning sidecar changed while reading")
    try:
        (magic, version, header_size, prompt_bytes, token_count, recipe_bytes,
         width, flags, embedding_bytes, tag_bytes, token_id_bytes, model_sha,
         reserved) = H3_SIDECAR_HEADER.unpack_from(payload)
    except struct.error as error:
        raise ContractError("h3cspeed conditioning sidecar header is invalid") from error
    if (magic != b"H3CSEV01" or version != 2 or
            header_size != H3_SIDECAR_HEADER.size or width != 5120 or flags != 1 or
            token_count <= 0 or recipe_bytes != len(H3_SIDECAR_RECIPE) or
            embedding_bytes != token_count * width * 2 or
            tag_bytes != token_count or token_id_bytes != token_count * 4 or
            model_sha.hex() != qwen_sha256):
        raise ContractError("h3cspeed conditioning sidecar v2 contract is invalid")
    mode, role, count, order, first_resize, last_resize = reserved[:6]
    render_width, render_height, metadata_bytes = struct.unpack_from("<III", reserved, 8)
    if (reserved[6:8] != b"\0\0" or reserved[20:24] != b"\0\0\0\0" or
            (mode, role, count, order, first_resize, last_resize) != (1, 1, 1, 1, 0, 1) or
            (render_width, render_height, metadata_bytes) != (864, 480, 32)):
        raise ContractError("h3cspeed conditioning sidecar is not first-frame 864x480 FL2VA")
    expected_size = (header_size + prompt_bytes + recipe_bytes + metadata_bytes +
                     token_id_bytes + embedding_bytes + tag_bytes)
    if expected_size != len(payload):
        raise ContractError("h3cspeed conditioning sidecar payload length is invalid")
    offset = header_size
    sidecar_prompt = payload[offset:offset + prompt_bytes]
    offset += prompt_bytes
    recipe = payload[offset:offset + recipe_bytes]
    offset += recipe_bytes
    reference_hash = payload[offset:offset + metadata_bytes]
    if (sidecar_prompt != prompt or recipe != H3_SIDECAR_RECIPE or
            reference_hash.hex() != reference_sha256):
        raise ContractError("h3cspeed sidecar prompt, recipe, or first-frame hash mismatches")


def _validate_h3_command(config: dict[str, Any], manifest: dict[str, Any],
                         argv: list[str]) -> None:
    command_inputs = _mapping(config.get("command_inputs"), "command_inputs")
    expected_inputs = {
        key: {"section": section, "label": label}
        for key, (section, label) in H3_INPUT_BINDINGS.items()
    }
    if command_inputs != expected_inputs:
        raise ContractError("h3cspeed command_inputs do not match the direct-binary schema")

    # Parse the public CLI grammar without allowing a hidden render downgrade,
    # Ref2VA last frame, or duplicate/unknown option that could change the run.
    consumed: set[int] = set()
    for index, value in enumerate(argv[1:], start=1):
        if index in consumed:
            continue
        if not value.startswith("-"):
            raise ContractError("h3cspeed argv contains an unbound positional argument")
        if value in H3_FORBIDDEN_FLAGS:
            raise ContractError(f"h3cspeed flag {value} is forbidden for the smoke contract")
        if value in H3_VALUE_FLAGS:
            _, argument = _single_h3_flag(argv, value)
            consumed.add(index)
            consumed.add(index + 1)
            if not argument:
                raise ContractError(f"h3cspeed flag {value} must have one value")
            continue
        if value in H3_SWITCH_FLAGS:
            indices = [item for item, candidate in enumerate(argv) if candidate == value]
            if len(indices) != 1:
                raise ContractError(f"h3cspeed flag {value} must occur exactly once")
            consumed.add(index)
            continue
        raise ContractError(f"unknown h3cspeed option {value}")

    required_values = {
        "--width": "864", "--height": "480", "--frames": "22",
        "--steps": "2", "--layers": "50", "--reuse": "1",
        "--core-reuse": "1", "--seed": "42",
    }
    for flag, expected in required_values.items():
        _, actual = _single_h3_flag(argv, flag)
        if actual != expected:
            raise ContractError(f"h3cspeed flag {flag} must be {expected}")
    _, model_root_value = _single_h3_flag(argv, "-d")
    _, prompt_value = _single_h3_flag(argv, "-p")
    _, first_frame_value = _single_h3_flag(argv, "--first-frame")
    _, output_value = _single_h3_flag(argv, "-o")

    model_root_config = Path(config.get("model_root", ""))
    if not model_root_config.is_absolute():
        raise ContractError("h3cspeed model_root must be absolute")
    model_root = safe_existing_directory(model_root_config, "h3cspeed model root")
    model_root_argument = safe_existing_directory(Path(model_root_value),
                                                  "h3cspeed -d model root")
    if model_root_argument != model_root:
        raise ContractError("h3cspeed -d does not match model_root")

    model_bindings = _mapping(
        _mapping(_mapping(config.get("bindings"), "bindings").get("models"),
                 "bindings.models").get("h3cspeed"),
        "bindings.models.h3cspeed",
    )
    manifest_models = _mapping(manifest["models"].get("h3cspeed"),
                               "manifest.models.h3cspeed")
    resolved_models: dict[str, Path] = {}
    for label, relative in H3_MODEL_PATHS.items():
        if label not in model_bindings or label not in manifest_models:
            raise ContractError(f"h3cspeed model binding lacks {label}")
        model_file = safe_regular(Path(model_bindings[label]),
                                  f"h3cspeed model {label}")
        expected = (model_root / relative).resolve()
        if model_file != expected:
            raise ContractError(f"h3cspeed model {label} is not at its native loader path")
        resolved_models[label] = model_file
    for label in ("fl2va", "qwen", "video_vae", "audio_vae"):
        component = resolved_models[label].parent
        inventory = {
            safe_regular(candidate, f"h3cspeed {label} inventory")
            for candidate in component.glob("*.safetensors")
        }
        if inventory != {resolved_models[label]}:
            raise ContractError(f"h3cspeed {label} safetensors inventory is not exact")
    ref2va_index = model_root / "Ref2VA/transformer/model.safetensors.index.json"
    if ref2va_index.exists() or ref2va_index.is_symlink():
        raise ContractError("h3cspeed FL2VA smoke model root must not enable Ref2VA")

    reference = Path(config.get("reference_png", ""))
    prompt = Path(config.get("prompt_file", ""))
    if not reference.is_absolute() or not prompt.is_absolute():
        raise ContractError("h3cspeed reference_png and prompt_file must be absolute")
    reference = safe_regular(reference, "reference PNG")
    prompt = safe_regular(prompt, "prompt file")
    first_frame = safe_regular(Path(first_frame_value), "h3cspeed first frame")
    if first_frame != reference or sha256_file(first_frame) != manifest["fixture"]["sha256"]:
        raise ContractError("h3cspeed --first-frame is not the manifest reference")
    try:
        prompt_bytes = prompt.read_bytes()
        prompt_text = prompt_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError("h3cspeed prompt file must be UTF-8") from error
    if prompt_value != prompt_text:
        raise ContractError("h3cspeed -p must equal the prompt file bytes")

    output = Path(config.get("output_media", ""))
    if not output.is_absolute() or Path(os.path.abspath(output_value)) != Path(os.path.abspath(output)):
        raise ContractError("h3cspeed -o does not match output_media")

    environment = _mapping(config.get("environment"), "environment")
    for key, config_key in (
            ("H3CSPEED_PERF002_SCHEDULER_TRACE", "scheduler_trace"),
            ("H3CSPEED_PERF002_ATTENTION_TRACE", "attention_trace")):
        configured = Path(config.get(config_key, ""))
        value = environment.get(key)
        if (not configured.is_absolute() or not isinstance(value, str) or
                not Path(value).is_absolute() or
                Path(os.path.abspath(value)) != Path(os.path.abspath(configured))):
            raise ContractError(f"{key} must equal the configured trace path")

    conditioning = _mapping(
        _mapping(config.get("bindings"), "bindings").get("conditioning"),
        "bindings.conditioning",
    )
    h3_conditioning = _mapping(conditioning.get("h3cspeed"),
                                "bindings.conditioning.h3cspeed")
    manifest_conditioning = _mapping(manifest["conditioning"].get("h3cspeed"),
                                     "manifest.conditioning.h3cspeed")
    sidecar_path = h3_conditioning.get("sidecar")
    if "sidecar" not in manifest_conditioning or not isinstance(sidecar_path, str):
        raise ContractError("h3cspeed manifest must bind a conditioning sidecar")
    sidecar = safe_regular(Path(sidecar_path), "h3cspeed conditioning sidecar")
    if sha256_file(sidecar) != manifest_conditioning["sidecar"]:
        raise ContractError("h3cspeed conditioning sidecar does not match the manifest")
    sidecar_env = environment.get("H3CSPEED_TEXT_EMBEDDING")
    if (not isinstance(sidecar_env, str) or not Path(sidecar_env).is_absolute() or
            Path(os.path.abspath(sidecar_env)) != Path(os.path.abspath(sidecar))):
        raise ContractError("H3CSPEED_TEXT_EMBEDDING must equal the manifest sidecar")

    qwen_hash = manifest_models.get("qwen")
    encoder_hash = environment.get("H3CSPEED_TEXT_ENCODER_SHA256")
    if (not isinstance(qwen_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", qwen_hash) or
            not isinstance(encoder_hash, str) or encoder_hash.lower() != qwen_hash):
        raise ContractError("H3CSPEED_TEXT_ENCODER_SHA256 must equal the manifest Qwen hash")
    _validate_h3_sidecar(sidecar, prompt_bytes, manifest["fixture"]["sha256"], qwen_hash)
    if environment.get("H3_CUDA_TF32") != "0":
        raise ContractError("h3cspeed algorithm-parity smoke requires H3_CUDA_TF32=0")
    if environment.get("H3_CUDA_ATTENTION") != "sage":
        raise ContractError("h3cspeed algorithm-parity smoke requires Sage attention")
    engine_bindings = _mapping(
        _mapping(_mapping(config.get("bindings"), "bindings").get("engines"),
                 "bindings.engines").get("h3cspeed"),
        "bindings.engines.h3cspeed",
    )
    native_ffmpeg_value = engine_bindings.get("ffmpeg")
    if not isinstance(native_ffmpeg_value, str):
        raise ContractError("h3cspeed engine bindings lack ffmpeg")
    native_ffmpeg = safe_regular(Path(native_ffmpeg_value), "h3cspeed native ffmpeg")
    ffmpeg_env = environment.get("H3_FFMPEG")
    if (not isinstance(ffmpeg_env, str) or not Path(ffmpeg_env).is_absolute() or
            Path(os.path.abspath(ffmpeg_env)) != Path(os.path.abspath(native_ffmpeg))):
        raise ContractError("H3_FFMPEG must equal the manifest-bound ffmpeg")


def _validate_h3_qa_tools(config: dict[str, Any], manifest: dict[str, Any],
                          ffmpeg: str, ffprobe: str) -> None:
    bindings = _mapping(
        _mapping(_mapping(config.get("bindings"), "bindings").get("engines"),
                 "bindings.engines").get("h3cspeed"),
        "bindings.engines.h3cspeed",
    )
    manifest_engines = _mapping(manifest["engines"].get("h3cspeed"),
                                "manifest.engines.h3cspeed")
    for label, value in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)):
        binding = bindings.get(label)
        if label not in manifest_engines or not isinstance(binding, str):
            raise ContractError(f"h3cspeed engine bindings lack {label}")
        bound = safe_regular(Path(binding), f"h3cspeed bound {label}")
        actual = safe_regular(Path(value), f"h3cspeed QA {label}")
        if actual != bound or sha256_file(actual) != manifest_engines[label]:
            raise ContractError(f"h3cspeed QA {label} is not manifest-bound")


def _validate_command(config: dict[str, Any], manifest: dict[str, Any],
                      engine: str) -> list[str]:
    if config.get("schema_version") != SCHEMA_VERSION or config.get("engine") != engine:
        raise ContractError("private command config does not match this engine")
    argv = config.get("argv")
    if (not isinstance(argv, list) or not argv or
            not all(isinstance(item, str) and item and "\x00" not in item for item in argv)):
        raise ContractError("argv must be a non-empty string list")
    raw_executable = Path(argv[0])
    if not raw_executable.is_absolute():
        raise ContractError("engine executable must be absolute")
    executable = safe_regular(raw_executable, "engine executable")
    if os.name != "nt" and not os.access(executable, os.X_OK):
        raise ContractError("engine executable is not executable")
    argv[0] = str(executable)
    command_artifacts = _mapping(config.get("command_artifacts"),
                                 "command_artifacts")
    if set(command_artifacts) != {"executable", "driver"}:
        raise ContractError("command_artifacts must bind executable and driver")
    engine_bindings = _mapping(
        _mapping(config.get("bindings"), "bindings").get("engines"),
        "bindings.engines",
    )
    bound_paths = _mapping(engine_bindings.get(engine), f"bindings.engines.{engine}")
    manifest_hashes = _mapping(manifest["engines"].get(engine),
                               f"manifest.engines.{engine}")
    resolved_indices: dict[str, int] = {}
    for role in ("executable", "driver"):
        item = _mapping(command_artifacts[role], f"command_artifacts.{role}")
        if set(item) != {"label", "argv_index"}:
            raise ContractError(f"command_artifacts.{role} has invalid fields")
        label = item.get("label")
        index = item.get("argv_index")
        if (not isinstance(label, str) or label not in manifest_hashes or
                label not in bound_paths or isinstance(index, bool) or
                not isinstance(index, int) or index < 0 or index >= len(argv)):
            raise ContractError(f"command_artifacts.{role} is not manifest-bound")
        artifact = safe_regular(Path(bound_paths[label]),
                                f"command_artifacts.{role}")
        argument = Path(argv[index])
        if not argument.is_absolute():
            raise ContractError(f"command_artifacts.{role} argv item must be absolute")
        argument = safe_regular(argument, f"command_artifacts.{role} argv item")
        if argument != artifact or sha256_file(artifact) != manifest_hashes[label]:
            raise ContractError(f"command_artifacts.{role} does not match the manifest")
        argv[index] = str(argument)
        resolved_indices[role] = index
    if resolved_indices["executable"] != 0:
        raise ContractError("command executable must be argv[0]")
    if engine == "h3cspeed":
        if (resolved_indices["driver"] != 0 or
                command_artifacts["executable"]["label"] != "binary" or
                command_artifacts["driver"]["label"] != "binary"):
            raise ContractError("h3cspeed smoke must directly execute its bound binary")
        _validate_h3_command(config, manifest, argv)
    elif (resolved_indices["driver"] != 1 or
          command_artifacts["executable"]["label"] != "python_env" or
          command_artifacts["driver"]["label"] != "source" or
          len(argv) < 2 or argv[1].startswith("-")):
        raise ContractError("ComfyUI smoke must execute its bound driver as argv[1]")
    command_inputs = _mapping(config.get("command_inputs", {}), "command_inputs")
    if engine == "comfyui":
        if set(command_inputs) != set(COMFY_INPUT_FLAGS):
            raise ContractError("ComfyUI smoke must bind all source/model input flags")
        for flag, expected_label in COMFY_INPUT_FLAGS.items():
            entry = _mapping(command_inputs.get(flag), f"command_inputs.{flag}")
            if set(entry) != {"section", "label"} or entry.get("section") != "engines":
                raise ContractError(f"command_inputs.{flag} must bind engines label")
            label = entry.get("label")
            if label != expected_label or label not in manifest_hashes or label not in bound_paths:
                raise ContractError(f"command_inputs.{flag} is not manifest-bound")
            indices = [index for index, value in enumerate(argv) if value == flag]
            if len(indices) != 1 or indices[0] + 1 >= len(argv):
                raise ContractError(f"ComfyUI smoke flag {flag} must have one path")
            argument = Path(argv[indices[0] + 1])
            if not argument.is_absolute():
                raise ContractError(f"ComfyUI smoke path after {flag} must be absolute")
            argument = safe_regular(argument, f"command_inputs.{flag}")
            bound = safe_regular(Path(bound_paths[label]), f"command_inputs.{flag}")
            if argument != bound or sha256_file(bound) != manifest_hashes[label]:
                raise ContractError(f"command_inputs.{flag} does not match the manifest")
            argv[indices[0] + 1] = str(argument)

        immutable_paths = (
            ("--reference-png", "reference_png", "reference PNG",
             manifest["fixture"]["sha256"]),
            ("--prompt-file", "prompt_file", "prompt file",
             manifest["prompt_sha256"]),
        )
        for flag, config_key, label, expected_hash in immutable_paths:
            indices = [index for index, value in enumerate(argv) if value == flag]
            if len(indices) != 1 or indices[0] + 1 >= len(argv):
                raise ContractError(f"ComfyUI smoke flag {flag} must have one path")
            configured = safe_regular(Path(config[config_key]), label)
            argument = safe_regular(Path(argv[indices[0] + 1]), f"{label} argv item")
            if argument != configured or sha256_file(configured) != expected_hash:
                raise ContractError(f"{label} does not match the immutable manifest")
            argv[indices[0] + 1] = str(argument)

        for flag, config_key, label in (
                ("--output-media", "output_media", "output media"),
                ("--scheduler-trace", "scheduler_trace", "scheduler trace"),
                ("--attention-trace", "attention_trace", "attention trace")):
            indices = [index for index, value in enumerate(argv) if value == flag]
            if len(indices) != 1 or indices[0] + 1 >= len(argv):
                raise ContractError(f"ComfyUI smoke flag {flag} must have one path")
            configured = Path(config[config_key])
            argument = Path(argv[indices[0] + 1])
            if not configured.is_absolute() or not argument.is_absolute():
                raise ContractError(f"{label} must be an absolute path")
            _reject_link_chain(configured, label)
            _reject_link_chain(argument, f"{label} argv item")
            if Path(os.path.abspath(configured)) != Path(os.path.abspath(argument)):
                raise ContractError(f"{label} does not match the config")
            argv[indices[0] + 1] = str(Path(os.path.abspath(argument)))

        runtime_indices = [index for index, value in enumerate(argv)
                           if value == "--runtime-dir"]
        if len(runtime_indices) != 1 or runtime_indices[0] + 1 >= len(argv):
            raise ContractError("ComfyUI smoke flag --runtime-dir must have one path")
        runtime = Path(config["runtime_dir"])
        argument = Path(argv[runtime_indices[0] + 1])
        if not runtime.is_absolute() or not argument.is_absolute():
            raise ContractError("ComfyUI runtime directory must be absolute")
        _reject_link_chain(runtime, "ComfyUI runtime directory")
        _reject_link_chain(argument, "ComfyUI runtime argv item")
        if Path(os.path.abspath(runtime)) != Path(os.path.abspath(argument)):
            raise ContractError("ComfyUI runtime directory does not match the config")
        if runtime.exists():
            if not runtime.is_dir() or any(runtime.iterdir()):
                raise ContractError("ComfyUI runtime directory must be new or empty")
        else:
            safe_existing_directory(runtime.parent, "ComfyUI runtime parent")
        argv[runtime_indices[0] + 1] = str(Path(os.path.abspath(runtime)))
    elif engine != "h3cspeed" and command_inputs:
        raise ContractError("unsupported engine command_inputs")
    return argv


def _run_engine(argv: list[str], log: Path, environment: dict[str, str],
                timeout_seconds: int) -> int:
    if os.name == "nt":
        with log.open("xb") as stream:
            try:
                return _run_engine_windows(argv, stream, environment, timeout_seconds)
            finally:
                stream.flush()
                os.fsync(stream.fileno())
    popen_kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL, "stdout": None, "stderr": subprocess.STDOUT,
        "env": environment, "shell": False,
    }
    popen_kwargs["start_new_session"] = True
    with log.open("xb") as stream:
        popen_kwargs["stdout"] = stream
        process = subprocess.Popen(argv, **popen_kwargs)
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired as cleanup_error:
                    raise ContractError(
                        "engine smoke timed out and could not be stopped"
                    ) from cleanup_error
            raise ContractError("engine smoke timed out and its process group was stopped") from error
        finally:
            stream.flush()
            os.fsync(stream.fileno())
    return return_code


def _scheduler_evidence(path: Path, engine: str) -> dict[str, Any]:
    report = _mapping(_load_json(path, "scheduler trace"), "scheduler trace")
    required = {"schema_version", "engine", "sampler", "schedule", "video_shift",
                "audio_shift", "sigma_video", "sigma_audio",
                "raw_audio_protocol_verified", "width", "height", "frames",
                "steps", "layers", "seed"}
    if set(report) != required or report.get("schema_version") != 1 or report.get("engine") != engine:
        raise ContractError("scheduler trace schema or engine is invalid")
    if (report.get("sampler"), report.get("schedule"), report.get("video_shift"),
            report.get("audio_shift"), report.get("raw_audio_protocol_verified")) != (
            "dual_clock_euler", "native_flow", 12.0, 3.0, True):
        raise ContractError("scheduler trace is not the algorithm-parity contract")
    if (report.get("width"), report.get("height"), report.get("frames"),
            report.get("steps"), report.get("layers"), report.get("seed")) != (
            864, 480, 22, 2, 50, 42):
        raise ContractError("scheduler trace is not the 22-frame/2-step smoke contract")
    for name in ("sigma_video", "sigma_audio"):
        values = report.get(name)
        if (not isinstance(values, list) or len(values) != 3 or
                not all(isinstance(value, (int, float)) and not isinstance(value, bool) and
                        math.isfinite(float(value)) for value in values)):
            raise ContractError(f"{name} must contain the three 2-step sigma points")
        if float(values[0]) != 1.0 or float(values[-1]) != 0.0:
            raise ContractError(f"{name} must span sigma 1.0 to 0.0")
        if not all(float(values[index]) >= float(values[index + 1])
                   for index in range(len(values) - 1)):
            raise ContractError(f"{name} must be monotonically non-increasing")
    return {
        "sampler": report["sampler"], "schedule": report["schedule"],
        "video_shift": 12.0, "audio_shift": 3.0,
        "sigma_video_sha256": sha256_bytes(canonical_bytes(report["sigma_video"])),
        "sigma_audio_sha256": sha256_bytes(canonical_bytes(report["sigma_audio"])),
        "raw_audio_protocol_verified": True,
        "trace_sha256": sha256_file(path),
    }


def _attention_evidence(path: Path, engine: str) -> dict[str, Any]:
    report = _mapping(_load_json(path, "attention trace"), "attention trace")
    required = {"schema_version", "engine", "requested", "selected",
                "scope", "backend_hits", "expected_native_calls",
                "unexpected_fallbacks"}
    if set(report) != required or report.get("schema_version") != 1 or report.get("engine") != engine:
        raise ContractError("attention trace schema or engine is invalid")
    hits = report.get("backend_hits")
    expected_native = report.get("expected_native_calls")
    fallbacks = report.get("unexpected_fallbacks")
    if (report.get("requested"), report.get("selected"), report.get("scope")) != (
            "sage", "sage", "dit_bf16"):
        raise ContractError("attention trace did not select Sage")
    if (isinstance(hits, bool) or not isinstance(hits, int) or hits <= 0 or
            isinstance(expected_native, bool) or not isinstance(expected_native, int) or
            expected_native < 0 or
            isinstance(fallbacks, bool) or not isinstance(fallbacks, int) or fallbacks != 0):
        raise ContractError("Sage smoke requires backend hits and zero fallbacks")
    return {"requested": "sage", "selected": "sage", "scope": "dit_bf16",
            "backend_hits": hits, "expected_native_calls": expected_native,
            "unexpected_fallbacks": 0, "trace_sha256": sha256_file(path)}


def run_smoke(manifest_path: Path, config_path: Path, output_directory: Path,
              ffmpeg: str, ffprobe: str, timeout_seconds: int) -> Path:
    if not 1 <= timeout_seconds <= 14400:
        raise ContractError("timeout must be between 1 and 14400 seconds")
    manifest, manifest_digest = load_input(manifest_path)
    if manifest["track"] != "algorithm_parity":
        raise ContractError("PERF-002C smoke requires the algorithm_parity track")
    output = safe_output_directory(output_directory)
    config = _mapping(_load_json(config_path, "private command config"),
                      "private command config")
    engine = config.get("engine")
    if engine not in ENGINES:
        raise ContractError("unsupported engine")
    if set(config) != _required_config_fields(engine):
        raise ContractError("private command config has unexpected or missing fields")
    protected_values = config.get("protected_roots")
    if (not isinstance(protected_values, list) or len(protected_values) < 3 or
            not all(isinstance(item, str) and item for item in protected_values)):
        raise ContractError("protected_roots must list source, ComfyUI, and model roots")
    protected = [safe_existing_directory(Path(item), "protected root")
                 for item in protected_values]
    if len(set(protected)) != len(protected):
        raise ContractError("protected_roots must be distinct")

    runtime_candidate: Path | None = None
    if engine == "comfyui":
        runtime_candidate = Path(config["runtime_dir"])
        if not runtime_candidate.is_absolute():
            raise ContractError("ComfyUI runtime directory must be absolute")
        _reject_link_chain(runtime_candidate, "ComfyUI runtime directory")
        runtime_absolute = Path(os.path.abspath(runtime_candidate))
        for root in protected:
            try:
                runtime_absolute.relative_to(root)
            except ValueError:
                continue
            raise ContractError("smoke outputs must be outside protected roots")
        if runtime_candidate.exists():
            runtime = safe_existing_directory(runtime_candidate,
                                              "ComfyUI runtime directory")
            if any(runtime.iterdir()):
                raise ContractError("ComfyUI runtime directory must be new or empty")
        else:
            safe_existing_directory(runtime_candidate.parent,
                                    "ComfyUI runtime parent")

    before = verify_bound_inputs(manifest, config, engine)
    argv = _validate_command(config, manifest, engine)
    if engine == "h3cspeed":
        _validate_h3_qa_tools(config, manifest, ffmpeg, ffprobe)
    environment = _private_environment(config.get("environment", {}))
    media = Path(config["output_media"])
    scheduler_trace = Path(config["scheduler_trace"])
    attention_trace = Path(config["attention_trace"])
    destinations = [output, media.parent, scheduler_trace.parent,
                    attention_trace.parent]
    if runtime_candidate is not None and runtime_candidate.exists():
        destinations.append(safe_existing_directory(
            runtime_candidate, "ComfyUI runtime directory"))
    for destination in destinations:
        resolved_destination = safe_output_directory(destination)
        for root in protected:
            try:
                resolved_destination.relative_to(root)
            except ValueError:
                continue
            raise ContractError("smoke outputs must be outside protected roots")
    for artifact, label in ((media, "output media"), (scheduler_trace, "scheduler trace"),
                            (attention_trace, "attention trace")):
        safe_output_directory(artifact.parent)
        if artifact.exists() or artifact.is_symlink():
            raise ContractError(f"{label} must not exist before the run")
    log = output / f"{engine}-smoke-private.log"
    if log.exists() or log.is_symlink():
        raise ContractError("private smoke log already exists")
    if runtime_candidate is not None and not runtime_candidate.exists():
        parent = safe_existing_directory(runtime_candidate.parent,
                                         "ComfyUI runtime parent")
        runtime_candidate.mkdir(mode=0o700)
        _reject_link_chain(runtime_candidate, "ComfyUI runtime directory")
        runtime = safe_existing_directory(runtime_candidate,
                                          "ComfyUI runtime directory")
        if runtime.parent != parent:
            raise ContractError("ComfyUI runtime parent changed during setup")
    started = time.perf_counter()
    return_code = _run_engine(argv, log, environment, timeout_seconds)
    wall = time.perf_counter() - started
    if return_code != 0:
        raise ContractError("engine smoke failed; inspect the private log")
    media_evidence = validate_media_contract(
        ffmpeg, ffprobe, media, output, frames=22, duration_range=(0.90, 0.96),
        sample_frames=(0, 5, 10, 16, 21),
    )
    scheduler = _scheduler_evidence(safe_regular(scheduler_trace, "scheduler trace"), engine)
    attention = _attention_evidence(safe_regular(attention_trace, "attention trace"), engine)
    after = verify_bound_inputs(manifest, config, engine)
    if before != after:
        raise ContractError("bound inputs changed while the smoke was running")
    result = {
        "schema_version": SCHEMA_VERSION, "kind": KIND,
        "input_manifest_sha256": manifest_digest, "engine": engine,
        "contract": {"width": 864, "height": 480, "frames": 22,
                     "fps": 24, "steps": 2, "layers": 50, "seed": 42},
        "command_sha256": sha256_bytes(canonical_bytes(argv)),
        "environment_sha256": sha256_bytes(canonical_bytes(environment)),
        "bindings_sha256": sha256_bytes(canonical_bytes(after)),
        "private_log_sha256": sha256_file(log), "wall_seconds": wall,
        "scheduler_evidence": scheduler, "attention_evidence": attention,
        "media": media_evidence, "status": "SMOKE_PASS",
        "matched_ab_status": "NOT_RUN",
    }
    destination = output / f"{engine}-smoke-result.json"
    publish_json(destination, result)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--command-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    if not 1 <= args.timeout_seconds <= 14400:
        raise ContractError("timeout must be between 1 and 14400 seconds")
    result = run_smoke(args.manifest, args.command_config, args.output_dir,
                       args.ffmpeg, args.ffprobe, args.timeout_seconds)
    print(f"PERF-002C engine smoke PASS: {result.name}")
    print("matched A/B status: NOT_RUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
