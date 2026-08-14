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
import signal
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_perf002_ab import (  # noqa: E402
    ContractError, ENGINES, canonical_bytes, load_input, png_dimensions,
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
}
BASE_ENVIRONMENT = {
    "APPDATA", "COMSPEC", "LOCALAPPDATA", "PATH", "PATHEXT", "PROGRAMDATA",
    "SYSTEMDRIVE", "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE", "WINDIR",
}


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
    elif (resolved_indices["driver"] != 1 or
          command_artifacts["executable"]["label"] != "python_env" or
          command_artifacts["driver"]["label"] != "source" or
          len(argv) < 2 or argv[1].startswith("-")):
        raise ContractError("ComfyUI smoke must execute its bound driver as argv[1]")
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
    required_config = {"schema_version", "engine", "argv", "environment",
                       "output_media", "scheduler_trace", "attention_trace",
                       "protected_roots", "reference_png", "prompt_file", "bindings",
                       "command_artifacts"}
    if set(config) != required_config:
        raise ContractError("private command config has unexpected or missing fields")
    engine = config.get("engine")
    if engine not in ENGINES:
        raise ContractError("unsupported engine")
    before = verify_bound_inputs(manifest, config, engine)
    argv = _validate_command(config, manifest, engine)
    environment = _private_environment(config.get("environment", {}))
    media = Path(config["output_media"])
    scheduler_trace = Path(config["scheduler_trace"])
    attention_trace = Path(config["attention_trace"])
    protected_values = config.get("protected_roots")
    if (not isinstance(protected_values, list) or len(protected_values) < 3 or
            not all(isinstance(item, str) and item for item in protected_values)):
        raise ContractError("protected_roots must list source, ComfyUI, and model roots")
    protected = [safe_existing_directory(Path(item), "protected root")
                 for item in protected_values]
    if len(set(protected)) != len(protected):
        raise ContractError("protected_roots must be distinct")
    for destination in (output, media.parent, scheduler_trace.parent,
                        attention_trace.parent):
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
