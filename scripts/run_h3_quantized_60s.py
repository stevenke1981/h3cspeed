#!/usr/bin/env python3
"""Generate a resumable 60-second 480p quantized H3 video.

The default plan is twelve validated-shape 124-frame H3 clips at 24 fps.  Clip one is
T2V; every later clip is FL2VA I2V conditioned on the preceding clip's last
decoded frame.  Each clip is verified before it becomes an input to the next
one, then FFmpeg concatenates and normalizes the result to exactly 1,440
frames.  Model weights and conditioning sidecars remain outside the bundle.
"""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import subprocess
import sys
import time
from typing import Any, TextIO


FPS = 24
DEFAULT_SEGMENTS = 12
DEFAULT_SEGMENT_FRAMES = 124
FINAL_FRAMES = 60 * FPS
MODEL_HASH_PATTERN = "model_sha256="


class RunError(RuntimeError):
    pass


class RunLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: TextIO | None = None
        self.identity: tuple[int, int] | None = None

    def _remove_owned_file(self) -> None:
        if self.identity is None:
            return
        try:
            metadata = self.path.lstat()
        except FileNotFoundError:
            return
        if (metadata.st_dev, metadata.st_ino) == self.identity:
            self.path.unlink()

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600
            )
        except FileExistsError as exc:
            raise RunError(
                f"run lock already exists at {self.path}; if no process is active, "
                "inspect and remove the stale lock explicitly"
            ) from exc
        self.handle = os.fdopen(descriptor, "w+b")  # type: ignore[assignment]
        metadata = os.fstat(self.handle.fileno())
        self.identity = (metadata.st_dev, metadata.st_ino)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            self._remove_owned_file()
            raise RunError(f"another 60-second run holds {self.path}") from exc
        try:
            self.handle.write(
                json.dumps({"pid": os.getpid(), "host": socket.gethostname()}).encode("utf-8")
            )
            self.handle.flush()
        except OSError:
            self.handle.close()
            self.handle = None
            self._remove_owned_file()
            raise
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        if self.handle is None:
            return
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None
        self._remove_owned_file()
        self.identity = None


def absolute_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise RunError(f"{label} is not a file: {path}")
    return path


def absolute_directory(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise RunError(f"{label} is not a directory: {path}")
    return path


def discover_binary(root: Path, explicit: str | None) -> Path:
    if explicit:
        return absolute_file(explicit, "h3cspeed binary")
    suffix = ".exe" if os.name == "nt" else ""
    candidates = (
        root / "bin" / f"h3cspeed{suffix}",
        root / "build-quant" / f"h3cspeed{suffix}",
        root / "build-native" / f"h3cspeed{suffix}",
        root / "build" / f"h3cspeed{suffix}",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RunError("h3cspeed binary was not found; pass --binary")


def discover_comfy_python(comfy: Path, explicit: str | None) -> Path:
    if explicit:
        return absolute_file(explicit, "ComfyUI Python")
    candidates = (
        comfy / ".venv" / "Scripts" / "python.exe",
        comfy / "venv" / "Scripts" / "python.exe",
        comfy.parent / ".venv" / "Scripts" / "python.exe",
        comfy.parent / "venv" / "Scripts" / "python.exe",
        comfy / ".venv" / "bin" / "python",
        comfy / "venv" / "bin" / "python",
        comfy.parent / ".venv" / "bin" / "python",
        comfy.parent / "venv" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RunError("ComfyUI Python was not found; pass --comfy-python")


def executable(value: str, label: str) -> str:
    resolved = shutil.which(value)
    if resolved is None:
        raise RunError(f"{label} executable was not found: {value}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_regular(path: Path, label: str, *, allow_missing: bool = False) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        if allow_missing:
            return
        raise RunError(f"{label} does not exist: {path}")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RunError(f"{label} must be a regular non-symlink file: {path}")


def safe_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        path.mkdir(mode=0o700)
        metadata = path.lstat()
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    is_reparse = bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)
    if stat.S_ISLNK(metadata.st_mode) or is_reparse or not stat.S_ISDIR(metadata.st_mode):
        raise RunError(f"{label} must be a real non-symlink directory: {path}")


def private_temp(parent: Path, suffix: str) -> tuple[Any, Path]:
    import tempfile

    directory = tempfile.TemporaryDirectory(prefix=".h3cspeed-", dir=parent)
    return directory, Path(directory.name) / f"payload{suffix}"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    safe_regular(path, "run state", allow_missing=True)
    temporary_directory, temporary = private_temp(path.parent, ".json")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary_directory.cleanup()


def load_or_create_state(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    if path.is_file():
        safe_regular(path, "run state")
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunError(f"run state is unreadable: {path}") from exc
        if state.get("schema") != 1 or state.get("config") != config:
            raise RunError(
                "existing run state does not match this prompt/model/binary/geometry; "
                "choose another output directory"
            )
        return state
    state = {
        "schema": 1,
        "status": "running",
        "config": config,
        "completed_segments": [],
        "updated_unix": time.time(),
    }
    atomic_json(path, state)
    return state


def source_fingerprint(root: Path, paths: tuple[str, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in paths:
        path = root / relative
        candidates: list[Path]
        if path.is_dir():
            candidates = sorted(
                candidate for candidate in path.rglob("*")
                if candidate.is_file()
                and "__pycache__" not in candidate.parts
                and candidate.suffix.lower() not in {".pyc", ".pyo"}
            )
        elif path.is_file():
            candidates = [path]
        else:
            raise RunError(f"expected ComfyUI runtime source is missing: {path}")
        for candidate in candidates:
            safe_regular(candidate, "ComfyUI runtime source")
            result[candidate.relative_to(root).as_posix()] = sha256_file(candidate)
    if not result:
        raise RunError(f"no expected ComfyUI runtime sources were found below {root}")
    return result


def python_environment_lock(python: Path) -> dict[str, Any]:
    script = r"""
import hashlib
import importlib.metadata as metadata
import json
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor

def fingerprint(distribution):
    name = distribution.metadata.get("Name", "").lower()
    if not name:
        return None
    digest = hashlib.sha256()
    files = sorted(distribution.files or [], key=str)
    actual_files = 0
    for relative in files:
        path = pathlib.Path(distribution.locate_file(relative))
        if not path.is_file():
            continue
        digest.update(str(relative).replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
        actual_files += 1
    return name, distribution.version, digest.hexdigest(), actual_files

distributions = sorted(
    metadata.distributions(),
    key=lambda item: (item.metadata.get("Name", "").lower(), item.version),
)
workers = min(8, max(1, os.cpu_count() or 1))
with ThreadPoolExecutor(max_workers=workers) as executor:
    fingerprints = list(executor.map(fingerprint, distributions))

out = {}
for item in fingerprints:
    if item is None:
        continue
    name, version, sha256, actual_files = item
    key = name if name not in out else f"{name}@{version}"
    out[key] = {
        "version": version,
        "sha256": sha256,
        "files": actual_files,
    }
print(json.dumps(out, sort_keys=True, separators=(",", ":")))
"""
    completed = subprocess.run(
        [str(python), "-I", "-c", script], capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        detail = completed.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise RunError(f"could not fingerprint the ComfyUI Python environment{suffix}")
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RunError("ComfyUI Python environment fingerprint is invalid") from exc
    if not isinstance(parsed, dict) or parsed.get("torch") is None:
        raise RunError("ComfyUI Python environment is missing required torch metadata")
    return parsed


def python_environment_inventory(python: Path) -> dict[str, Any]:
    """Return a fast mutation detector for per-segment dependency checks.

    The full content lock above is still computed when a run starts/resumes and
    again before final acceptance.  Re-reading every installed wheel twice per
    segment can take minutes on a large CUDA environment, so the inner loop
    checks each distribution's version plus every installed file's size and
    nanosecond mtime instead.
    """
    script = r"""
import hashlib
import importlib.metadata as metadata
import json
import pathlib

out = {}
for distribution in sorted(
        metadata.distributions(),
        key=lambda item: (item.metadata.get("Name", "").lower(), item.version)):
    name = distribution.metadata.get("Name", "").lower()
    if not name:
        continue
    digest = hashlib.sha256()
    actual_files = 0
    for relative in sorted(distribution.files or [], key=str):
        path = pathlib.Path(distribution.locate_file(relative))
        if not path.is_file():
            continue
        stat = path.stat()
        digest.update(str(relative).replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\0")
        actual_files += 1
    key = name if name not in out else f"{name}@{distribution.version}"
    out[key] = {
        "version": distribution.version,
        "inventory_sha256": digest.hexdigest(),
        "files": actual_files,
    }
print(json.dumps(out, sort_keys=True, separators=(",", ":")))
"""
    completed = subprocess.run(
        [str(python), "-I", "-c", script], capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        detail = completed.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise RunError(f"could not inventory the ComfyUI Python environment{suffix}")
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RunError("ComfyUI Python environment inventory is invalid") from exc
    if not isinstance(parsed, dict) or parsed.get("torch") is None:
        raise RunError("ComfyUI Python environment inventory is missing torch")
    return parsed


def build_runtime_environment(
    device: str, ffmpeg: str, ffprobe: str
) -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items()
        if not key.upper().startswith(("H3_", "H3CSPEED_"))
    }
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "H3_CUDA_ATTENTION": "sage",
        "H3_CUDA_DEVICE": device.split(":", 1)[1] if ":" in device else "0",
        "H3_CUDA_TF32": "0",
        "H3_CUDA_LOW_VRAM": "1",
        "H3_CUDA_OFFLOAD": "ram+file",
        "H3_CUDA_VRAM_BUDGET_MIB": "5888",
        "H3_CUDA_WEIGHT_CACHE_MIB": "1536",
        "H3_CUDA_PINNED_HOST_MIB": "128",
        "H3_CUDA_STAGING_MIB": "64",
        "H3_CUDA_RELEASE_SCRATCH": "1",
        "H3_PROFILE": "1",
        "H3_FFMPEG": ffmpeg,
        "H3_FFPROBE": ffprobe,
    })
    return environment


def gpu_identity(
    python: Path, binary: Path, device: str, environment: dict[str, str]
) -> dict[str, Any]:
    index = int(device.split(":", 1)[1]) if ":" in device else 0
    script = (
        "import json,torch;"
        f"i={index};torch.cuda.set_device(i);p=torch.cuda.get_device_properties(i);"
        "print(json.dumps({'uuid':str(p.uuid),'name':p.name,'major':p.major,"
        "'minor':p.minor,'memory':p.total_memory,'pci_bus_id':p.pci_bus_id,"
        "'torch':torch.__version__,'torch_cuda':torch.version.cuda},sort_keys=True))"
    )
    completed = subprocess.run(
        [str(python), "-I", "-c", script], capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=environment, check=False,
    )
    if completed.returncode != 0:
        raise RunError(f"cannot resolve CUDA device {device} with ComfyUI Python")
    try:
        identity = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RunError("ComfyUI CUDA device identity is invalid") from exc
    sm = f"sm_{int(identity['major'])}{int(identity['minor'])}"
    nvidia_smi = shutil.which("nvidia-smi", path=environment.get("PATH"))
    if nvidia_smi is None:
        raise RunError("nvidia-smi is required to bind the driver and GPU UUID")
    smi = subprocess.run([
        nvidia_smi,
        "--query-gpu=uuid,name,driver_version,compute_cap,pci.bus_id",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, encoding="utf-8", errors="replace",
       env=environment, check=False)
    if smi.returncode != 0:
        raise RunError("nvidia-smi failed while binding the CUDA device identity")
    expected_uuid = str(identity["uuid"]).lower().removeprefix("gpu-")
    match: list[str] | None = None
    for line in smi.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 5 and fields[0].lower().removeprefix("gpu-") == expected_uuid:
            match = fields
            break
    if match is None:
        raise RunError("PyTorch CUDA device UUID was not reported by nvidia-smi")
    if match[1] != identity["name"] or match[3].replace(".", "") != sm[3:]:
        raise RunError("PyTorch and NVIDIA device identities do not match")
    info_name = "h3cspeed-cuda-info.exe" if os.name == "nt" else "h3cspeed-cuda-info"
    native_probe = binary.parent / info_name
    safe_regular(native_probe, "native CUDA identity probe")
    probe = subprocess.run(
        [str(native_probe)], capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=environment, check=False,
    )
    native_lines = {}
    for line in probe.stdout.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            native_lines[key.strip().lower()] = value.strip()
    if (probe.returncode != 0 or native_lines.get("device") != identity["name"] or
            sm not in native_lines.get("architecture", "")):
        raise RunError("native and PyTorch CUDA device identities do not match")
    identity.update({
        "driver": match[2],
        "compute_capability": match[3],
        "pci_bus": match[4],
        "native_device": native_lines["device"],
        "native_architecture": native_lines["architecture"],
    })
    return identity


def model_payload_fingerprints(
    manifest: Path, model_root: Path, encoder: Path, encoder_hash: str
) -> dict[str, str]:
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunError(f"quantized model manifest is unreadable: {manifest}") from exc
    if data.get("schema_version") != 2:
        raise RunError("quantized FL2VA model manifest must use schema version 2")
    if data.get("kind") != "minimax-h3-comfy-fl2va-quantized-pack":
        raise RunError("quantized model manifest is not an FL2VA pack")
    if data.get("model_family") != "FL2VA":
        raise RunError("quantized model manifest has the wrong model family")
    required_capabilities = {
        "t2v", "fl2va_i2v_first_frame", "fl2va_i2v_last_frame",
        "fl2va_i2v_first_and_last_frames",
    }
    capabilities = data.get("capabilities")
    if not isinstance(capabilities, list) or not required_capabilities.issubset(capabilities):
        raise RunError("quantized FL2VA model manifest lacks I2V capabilities")
    if 2 not in data.get("conditioning_sidecar_versions", []):
        raise RunError("quantized FL2VA model manifest lacks sidecar v2 support")
    root = manifest.parent.resolve()
    relative_model_root = data.get("model_root_relative")
    if not isinstance(relative_model_root, str) or not relative_model_root:
        raise RunError("quantized model manifest has no relative model root")
    declared_model_root = (root / relative_model_root).resolve()
    if model_root.resolve() != declared_model_root:
        raise RunError(
            f"model root does not match the prepared manifest: {declared_model_root}"
        )
    large_payloads = data.get("large_payloads")
    small_configs = data.get("small_configs")
    if not isinstance(large_payloads, list) or not large_payloads:
        raise RunError("quantized model manifest has no large payloads")
    if not isinstance(small_configs, list) or not small_configs:
        raise RunError("quantized model manifest has no small configs")
    result: dict[str, str] = {}
    qwen_hash: str | None = None
    for entry in large_payloads + small_configs:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise RunError("quantized model manifest contains an invalid path entry")
        relative = entry["path"].replace("\\", "/")
        path = (root / relative).resolve()
        if path != root and root not in path.parents:
            raise RunError(f"quantized model manifest path escapes its root: {relative}")
        safe_regular(path, f"quantized model payload {relative}")
        if path.stat().st_size != int(entry.get("bytes", -1)):
            raise RunError(f"quantized model payload size mismatch: {relative}")
        digest = encoder_hash if path.samefile(encoder) else sha256_file(path)
        result[relative] = digest
        if entry.get("role") == "qwen_nvfp4":
            qwen_hash = digest
    if qwen_hash != encoder_hash:
        raise RunError("text encoder does not match the prepared model root Qwen payload")
    return result


def file_inventory(paths: dict[str, Path], label: str) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for name, path in sorted(paths.items()):
        safe_regular(path, f"{label} {name}")
        metadata = path.stat()
        result[name] = {
            "bytes": metadata.st_size,
            "mtime_ns": metadata.st_mtime_ns,
        }
    return result


def model_payload_inventory(
    manifest: Path, model_root: Path, encoder: Path
) -> dict[str, dict[str, int]]:
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunError(f"quantized model manifest is unreadable: {manifest}") from exc
    if data.get("kind") != "minimax-h3-comfy-fl2va-quantized-pack":
        raise RunError("quantized model manifest is not an FL2VA pack")
    root = manifest.parent.resolve()
    relative_model_root = data.get("model_root_relative")
    if not isinstance(relative_model_root, str) or (
        model_root.resolve() != (root / relative_model_root).resolve()
    ):
        raise RunError("model root does not match the prepared manifest")
    entries = data.get("large_payloads", []) + data.get("small_configs", [])
    paths: dict[str, Path] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise RunError("quantized model manifest contains an invalid path entry")
        relative = entry["path"].replace("\\", "/")
        path = (root / relative).resolve()
        if path != root and root not in path.parents:
            raise RunError(f"quantized model manifest path escapes its root: {relative}")
        paths[relative] = path
    if not paths or not any(path.samefile(encoder) for path in paths.values()):
        raise RunError("text encoder is not part of the prepared model root")
    return file_inventory(paths, "quantized model payload")


def runtime_fingerprints(binary: Path) -> dict[str, str]:
    binary = binary.resolve()
    if binary.parent.name.lower() != "bin":
        paths = {
            path.name: path
            for path in sorted(binary.parent.iterdir())
            if path.is_file() and path.suffix.lower() in {
                "", ".exe", ".dll", ".so", ".dylib",
            }
        }
        if binary.name not in paths:
            paths[binary.name] = binary
        return {name: sha256_file(path) for name, path in paths.items()}
    root = binary.parent.parent
    directories = [binary.parent]
    for name in ("libexec", "lib"):
        candidate = root / name
        if candidate.is_dir():
            directories.append(candidate)
    result: dict[str, str] = {}
    for directory in directories:
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            safe_regular(path, "portable runtime payload")
            result[path.relative_to(root).as_posix()] = sha256_file(path)
    if not result:
        raise RunError(f"portable runtime payload is empty below {root}")
    return result


def runtime_inventory(binary: Path) -> dict[str, dict[str, int]]:
    binary = binary.resolve()
    if binary.parent.name.lower() != "bin":
        paths = {
            path.name: path
            for path in sorted(binary.parent.iterdir())
            if path.is_file() and path.suffix.lower() in {
                "", ".exe", ".dll", ".so", ".dylib",
            }
        }
        paths.setdefault(binary.name, binary)
        return file_inventory(paths, "runtime payload")
    root = binary.parent.parent
    paths: dict[str, Path] = {}
    for directory in (binary.parent, root / "libexec", root / "lib"):
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                paths[path.relative_to(root).as_posix()] = path
    if not paths:
        raise RunError(f"portable runtime payload is empty below {root}")
    return file_inventory(paths, "runtime payload")


def path_within(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


def run_logged(command: list[str], log: Path, env: dict[str, str] | None = None) -> str:
    log.parent.mkdir(parents=True, exist_ok=True)
    safe_regular(log, "log output", allow_missing=True)
    lines: list[str] = []
    with log.open("a", encoding="utf-8", errors="replace") as sink:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            sink.write(line)
            sink.flush()
            lines.append(line)
        return_code = process.wait()
    if return_code != 0:
        raise RunError(f"command failed with exit code {return_code}; see {log}")
    return "".join(lines)


def probe_media(ffprobe: str, path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-count_frames", "-show_entries",
            "stream=index,codec_type,codec_name,width,height,r_frame_rate,nb_read_frames,sample_rate,channels:format=duration",
            "-of", "json", str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RunError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RunError(f"ffprobe returned invalid JSON for {path}") from exc


def validate_media(
    ffmpeg: str,
    ffprobe: str,
    path: Path,
    width: int,
    height: int,
    frames: int,
    exact_duration: float | None = None,
) -> dict[str, Any]:
    safe_regular(path, "media output")
    if not path.is_file() or path.stat().st_size == 0:
        raise RunError(f"media output is missing or empty: {path}")
    probe = probe_media(ffprobe, path)
    streams = probe.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if video is None or audio is None:
        raise RunError(f"expected both video and audio streams: {path}")
    if video.get("codec_name") != "h264" or audio.get("codec_name") != "aac":
        raise RunError(f"expected H.264 video and AAC audio in {path}")
    if int(video.get("width", 0)) != width or int(video.get("height", 0)) != height:
        raise RunError(f"unexpected video dimensions in {path}")
    if video.get("r_frame_rate") != f"{FPS}/1":
        raise RunError(f"expected {FPS} fps in {path}")
    if str(audio.get("sample_rate")) != "32000" or int(audio.get("channels", 0)) != 2:
        raise RunError(f"expected 32 kHz stereo audio in {path}")
    try:
        decoded_frames = int(video.get("nb_read_frames", 0))
    except (TypeError, ValueError) as exc:
        raise RunError(f"ffprobe did not report a frame count for {path}") from exc
    if decoded_frames != frames:
        raise RunError(f"expected {frames} frames in {path}, found {decoded_frames}")
    duration = float(probe.get("format", {}).get("duration", 0.0))
    expected = frames / FPS if exact_duration is None else exact_duration
    tolerance = 0.001 if exact_duration is not None else (1.0 / FPS + 0.02)
    if abs(duration - expected) > tolerance:
        raise RunError(f"unexpected duration {duration:.6f}s in {path}; expected {expected:.6f}s")
    null_sink = "NUL" if os.name == "nt" else "/dev/null"
    decode = subprocess.run(
        [ffmpeg, "-v", "error", "-xerror", "-i", str(path), "-f", "null", null_sink],
        check=False,
    )
    if decode.returncode != 0:
        raise RunError(f"full FFmpeg decode failed for {path}")
    return probe


def validate_final_audio(ffmpeg: str, path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            ffmpeg, "-v", "error", "-xerror", "-i", str(path),
            "-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le",
            "-ar", "32000", "-ac", "2", "pipe:1",
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RunError(f"final audio decode failed for {path}")
    expected_values = 60 * 32000 * 2
    if len(result.stdout) != expected_values * 4:
        raise RunError(
            f"expected {expected_values // 2} samples/channel, decoded {len(result.stdout) // 8}"
        )
    values = array("f")
    values.frombytes(result.stdout)
    if sys.byteorder != "little":
        values.byteswap()
    peak = max((abs(value) for value in values), default=0.0)
    mean_square = sum(value * value for value in values) / max(1, len(values))
    rms = math.sqrt(mean_square)
    if not math.isfinite(rms) or not math.isfinite(peak) or peak <= 1.0e-6:
        raise RunError("final audio is silent or non-finite")
    return {
        "sample_rate": 32000,
        "channels": 2,
        "samples_per_channel": expected_values // 2,
        "rms_dbfs": 20.0 * math.log10(max(rms, 1.0e-30)),
        "peak_dbfs": 20.0 * math.log10(max(peak, 1.0e-30)),
    }


def extract_last_frame(ffmpeg: str, source: Path, destination: Path, frame: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    safe_regular(destination, "anchor output", allow_missing=True)
    temporary_directory, temporary = private_temp(destination.parent, ".png")
    try:
        result = subprocess.run(
            [
                ffmpeg, "-v", "error", "-y", "-i", str(source),
                "-vf", f"select=eq(n\\,{frame})", "-frames:v", "1", str(temporary),
            ],
            check=False,
        )
        if result.returncode != 0 or not temporary.is_file():
            raise RunError(f"failed to extract last frame from {source}")
        os.replace(temporary, destination)
    finally:
        temporary_directory.cleanup()


def concat_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "'\\''")


def build_final_command(ffmpeg: str, concat_file: Path, output: Path) -> list[str]:
    return [
        ffmpeg, "-v", "error", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_file), "-map", "0:v:0", "-map", "0:a:0",
        "-vf", f"fps={FPS},trim=end_frame={FINAL_FRAMES},setpts=N/({FPS}*TB)",
        "-af", "aresample=32000,apad,atrim=end_sample=1920000,asetpts=N/SR/TB",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "32000", "-ac", "2",
        "-movflags", "+faststart", str(output),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--comfyui", required=True)
    parser.add_argument("--text-encoder", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--final-output")
    parser.add_argument("--binary")
    parser.add_argument("--comfy-python")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--render-width", type=int, default=288)
    parser.add_argument("--render-height", type=int, default=160)
    parser.add_argument("--segments", type=int, default=DEFAULT_SEGMENTS)
    parser.add_argument("--segment-frames", type=int, default=DEFAULT_SEGMENT_FRAMES)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--layers", type=int, default=50)
    parser.add_argument("--reuse", type=int, default=1)
    parser.add_argument("--core-reuse", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=4242)
    parser.add_argument("--stop-after", type=int, default=0,
                        help="stop after this segment (0 runs all; useful for smoke tests)")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.prompt.strip():
        raise RunError("prompt must not be empty")
    if re.fullmatch(r"cuda(?::[0-9]+)?", args.device, flags=re.IGNORECASE) is None:
        raise RunError("device must be CUDA; CPU fallback is forbidden")
    if min(args.width, args.height, args.render_width, args.render_height) < 64:
        raise RunError("output and render dimensions must be at least 64")
    if args.width % 32 or args.height % 32:
        raise RunError("output dimensions must be divisible by 32")
    if args.width * args.height > 768 * 1344:
        raise RunError("output canvas exceeds the released 768*1344 pixel limit")
    if args.render_width > args.width or args.render_height > args.height:
        raise RunError("render dimensions must not exceed output dimensions")
    if args.render_width % 32 or args.render_height % 32:
        raise RunError("render dimensions must be divisible by 32")
    if args.render_width * args.height != args.render_height * args.width:
        raise RunError("render dimensions must preserve the output aspect ratio")
    if args.segment_frames < 22 or args.segment_frames > 362 or (args.segment_frames - 5) % 17:
        raise RunError("segment frames must follow H3's 5 + 17n layout and be at least 22")
    if args.segments < 1 or not 2 <= args.steps <= 1000 or not 1 <= args.layers <= 50:
        raise RunError("segments must be positive, steps in [2,1000], and layers in [1,50]")
    if not 1 <= args.reuse <= 3 or not 1 <= args.core_reuse <= 6:
        raise RunError("reuse must be in [1,3] and core-reuse in [1,6]")
    if args.reuse > 1 and args.core_reuse > 1:
        raise RunError("reuse > 1 and core-reuse > 1 cannot be combined")
    if args.segments * args.segment_frames < FINAL_FRAMES:
        raise RunError(
            f"segments * segment-frames must be at least {FINAL_FRAMES}; final encoding trims to 60 seconds"
        )
    if args.stop_after < 0 or args.stop_after > args.segments:
        raise RunError("stop-after must be zero or within the segment count")
    if args.base_seed < 0:
        raise RunError("base-seed must be non-negative")
    if args.base_seed + args.segments - 1 > (1 << 64) - 1:
        raise RunError("base-seed plus segment count exceeds uint64")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
        root = Path(__file__).resolve().parents[1]
        helper = absolute_file(str(root / "scripts" / "encode_h3_quantized_prompt.py"),
                               "conditioning helper")
        model_root = absolute_directory(args.model_root, "model root")
        comfy_root = absolute_directory(args.comfyui, "ComfyUI root")
        encoder = absolute_file(args.text_encoder, "quantized text encoder")
        binary = discover_binary(root, args.binary)
        comfy_python = discover_comfy_python(comfy_root, args.comfy_python)
        ffmpeg = executable(args.ffmpeg, "ffmpeg")
        ffprobe = executable(args.ffprobe, "ffprobe")
        output_root = Path(args.output_dir).expanduser().resolve()
        if path_within(output_root, root):
            raise RunError("output directory must be outside the portable/source tree")
        output_root.mkdir(parents=True, exist_ok=True)
        segments_root = output_root / "segments"
        anchors_root = output_root / "anchors"
        sidecars_root = output_root / "conditioning"
        logs_root = output_root / "logs"
        for directory in (segments_root, anchors_root, sidecars_root, logs_root):
            safe_directory(directory, "run output directory")
            if directory.resolve().parent != output_root:
                raise RunError(f"run output directory escapes its root: {directory}")
        final_output = (
            Path(args.final_output).expanduser().resolve()
            if args.final_output else output_root / "h3-quantized-60s-864x480.mp4"
        )
        if final_output.suffix.lower() != ".mp4":
            raise RunError("final output must end in .mp4")
        if path_within(final_output, root):
            raise RunError("final output must be outside the portable/source tree")
        state_path = output_root / "run-state.json"
        reserved = (segments_root, anchors_root, sidecars_root, logs_root)
        if final_output == state_path or any(
            final_output == directory or directory in final_output.parents for directory in reserved
        ):
            raise RunError("final output must not collide with run state or reserved directories")
        lock = RunLock(output_root / ".run.lock")
        lock.__enter__()
        try:
            encoder_hash = sha256_file(encoder)
            manifest = model_root.parent / "manifest.json"
            safe_regular(manifest, "quantized model manifest")
            model_identity = sha256_file(manifest)
            model_payloads = model_payload_fingerprints(
                manifest, model_root, encoder, encoder_hash
            )
            runtime_env = build_runtime_environment(args.device, ffmpeg, ffprobe)
            selected_gpu = gpu_identity(
                comfy_python, binary, args.device, runtime_env
            )
            config = {
                "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
                "text_encoder_sha256": encoder_hash,
                "model_root": str(model_root),
                "model_manifest_sha256": model_identity,
                "model_payload_sha256": model_payloads,
                "model_payload_inventory": model_payload_inventory(
                    manifest, model_root, encoder
                ),
                "binary_sha256": sha256_file(binary),
                "runtime_payload_sha256": runtime_fingerprints(binary),
                "runtime_payload_inventory": runtime_inventory(binary),
                "helper_sha256": sha256_file(helper),
                "runner_sha256": sha256_file(Path(__file__).resolve()),
                "comfy_python_sha256": sha256_file(comfy_python),
                "comfy_environment_sha256": python_environment_lock(comfy_python),
                "comfy_environment_inventory": python_environment_inventory(comfy_python),
                "comfy_root": str(comfy_root),
                "comfy_sources": source_fingerprint(comfy_root, (
                    "comfy", "comfy_extras/nodes_minimax_h3.py", "folder_paths.py",
                )),
                "ffmpeg": str(Path(ffmpeg).resolve()),
                "ffprobe": str(Path(ffprobe).resolve()),
                "ffmpeg_sha256": sha256_file(Path(ffmpeg).resolve()),
                "ffprobe_sha256": sha256_file(Path(ffprobe).resolve()),
                "device": args.device,
                "gpu_identity": selected_gpu,
                "width": args.width,
                "height": args.height,
                "render_width": args.render_width,
                "render_height": args.render_height,
                "segments": args.segments,
                "segment_frames": args.segment_frames,
                "steps": args.steps,
                "layers": args.layers,
                "reuse": args.reuse,
                "core_reuse": args.core_reuse,
                "base_seed": args.base_seed,
                "attention": "sage",
                "tf32": False,
            }

            def verify_bound_inputs(*, deep_content_check: bool = False) -> None:
                if sha256_file(manifest) != config["model_manifest_sha256"]:
                    raise RunError("quantized model manifest changed during the resumable run")
                if model_payload_inventory(
                    manifest, model_root, encoder
                ) != config["model_payload_inventory"]:
                    raise RunError("quantized model payload inventory changed during the resumable run")
                if runtime_inventory(binary) != config["runtime_payload_inventory"]:
                    raise RunError("portable runtime inventory changed during the resumable run")
                if deep_content_check:
                    current_encoder_hash = sha256_file(encoder)
                    if current_encoder_hash != config["text_encoder_sha256"]:
                        raise RunError("text encoder changed during the resumable run")
                    if model_payload_fingerprints(
                        manifest, model_root, encoder, current_encoder_hash
                    ) != config["model_payload_sha256"]:
                        raise RunError("quantized model payload changed during the resumable run")
                    if runtime_fingerprints(binary) != config["runtime_payload_sha256"]:
                        raise RunError("portable runtime payload changed during the resumable run")
                if sha256_file(helper) != config["helper_sha256"]:
                    raise RunError("conditioning helper changed during the resumable run")
                if sha256_file(Path(__file__).resolve()) != config["runner_sha256"]:
                    raise RunError("60-second runner changed during the resumable run")
                if sha256_file(comfy_python) != config["comfy_python_sha256"]:
                    raise RunError("ComfyUI Python changed during the resumable run")
                if python_environment_inventory(comfy_python) != config["comfy_environment_inventory"]:
                    raise RunError("ComfyUI Python package inventory changed during the resumable run")
                if deep_content_check and (
                    python_environment_lock(comfy_python) != config["comfy_environment_sha256"]
                ):
                    raise RunError("ComfyUI Python package contents changed during the resumable run")
                if source_fingerprint(comfy_root, (
                    "comfy", "comfy_extras/nodes_minimax_h3.py", "folder_paths.py",
                )) != config["comfy_sources"]:
                    raise RunError("ComfyUI conditioning sources changed during the resumable run")
                if sha256_file(Path(ffmpeg).resolve()) != config["ffmpeg_sha256"]:
                    raise RunError("FFmpeg changed during the resumable run")
                if sha256_file(Path(ffprobe).resolve()) != config["ffprobe_sha256"]:
                    raise RunError("FFprobe changed during the resumable run")
                if gpu_identity(
                    comfy_python, binary, args.device, runtime_env
                ) != config["gpu_identity"]:
                    raise RunError("CUDA GPU or driver changed during the resumable run")

            state = load_or_create_state(state_path, config)
            state["status"] = "running"
            state["updated_unix"] = time.time()
            atomic_json(state_path, state)

            recorded = {
                int(item["index"]): item
                for item in state.get("completed_segments", [])
                if isinstance(item, dict) and "index" in item
            }
            completed: list[dict[str, Any]] = []
            for number in range(1, args.segments + 1):
                verify_bound_inputs()
                segment = segments_root / f"segment-{number:02d}.mp4"
                anchor = anchors_root / f"segment-{number:02d}-last.png"
                sidecar = sidecars_root / f"segment-{number:02d}.h3c"
                log = logs_root / f"segment-{number:02d}.log"
                resumed = False
                try:
                    probe = validate_media(
                        ffmpeg, ffprobe, segment, args.width, args.height,
                        args.segment_frames,
                    )
                    prior = recorded.get(number)
                    if prior is None or sha256_file(segment) != prior.get("sha256"):
                        raise RunError(f"segment {number:02d} content hash does not match run state")
                    if number > 1:
                        previous_anchor = anchors_root / f"segment-{number - 1:02d}-last.png"
                        if not previous_anchor.is_file() or (
                            sha256_file(previous_anchor) != prior.get("input_anchor_sha256")
                        ):
                            raise RunError(f"segment {number:02d} anchor lineage does not match run state")
                    if not anchor.is_file() or sha256_file(anchor) != prior.get("output_anchor_sha256"):
                        raise RunError(f"segment {number:02d} output anchor does not match run state")
                    if not sidecar.is_file() or sha256_file(sidecar) != prior.get("sidecar_sha256"):
                        raise RunError(f"segment {number:02d} sidecar does not match run state")
                    print(f"[h3cspeed-60s] resume: segment {number:02d} already verified")
                    resumed = True
                except RunError:
                    if segment.exists():
                        quarantined = segment.with_suffix(f".invalid-{int(time.time())}.mp4")
                        segment.replace(quarantined)
                        print(f"[h3cspeed-60s] preserved invalid segment as {quarantined}")
                    first_frame: Path | None = None
                    if number > 1:
                        previous = segments_root / f"segment-{number - 1:02d}.mp4"
                        validate_media(
                            ffmpeg, ffprobe, previous, args.width, args.height,
                            args.segment_frames,
                        )
                        extract_last_frame(
                            ffmpeg, previous, anchors_root / f"segment-{number - 1:02d}-last.png",
                            args.segment_frames - 1,
                        )
                        first_frame = anchors_root / f"segment-{number - 1:02d}-last.png"
                    helper_command = [
                        str(comfy_python), str(helper), "--comfyui", str(comfy_root),
                        "--text-encoder", str(encoder), "--output", str(sidecar),
                        "--prompt", args.prompt, "--device", args.device,
                        "--mode", "fl2va-i2v" if first_frame else "t2v",
                        "--width", str(args.render_width), "--height", str(args.render_height),
                    ]
                    if first_frame:
                        helper_command.extend(("--first-frame", str(first_frame)))
                    print(f"[h3cspeed-60s] segment {number:02d}: conditioning")
                    helper_output = run_logged(helper_command, log, env=runtime_env)
                    reported = {
                        token.split()[0].lower()
                        for token in helper_output.split(MODEL_HASH_PATTERN)[1:]
                        if len(token.split()[0]) == 64
                    }
                    if reported != {encoder_hash}:
                        raise RunError(
                            f"conditioning helper model hash mismatch for segment {number:02d}"
                        )
                    canonical_first = Path(str(sidecar) + ".first.png") if first_frame else None
                    if canonical_first is not None and not canonical_first.is_file():
                        raise RunError(f"canonical first frame is missing: {canonical_first}")
                    segment_env = runtime_env.copy()
                    segment_env["H3CSPEED_TEXT_EMBEDDING"] = str(sidecar)
                    segment_env["H3CSPEED_TEXT_ENCODER_SHA256"] = encoder_hash
                    native_command = [
                        str(binary), "-d", str(model_root), "-p", args.prompt,
                        "--width", str(args.width), "--height", str(args.height),
                        "--render-width", str(args.render_width),
                        "--render-height", str(args.render_height),
                        "--frames", str(args.segment_frames), "--steps", str(args.steps),
                        "--layers", str(args.layers), "--reuse", str(args.reuse),
                        "--core-reuse", str(args.core_reuse),
                        "--seed", str(args.base_seed + number - 1),
                        "--ssd-streaming", "-o", str(segment),
                    ]
                    if canonical_first is not None:
                        native_command.extend(("--first-frame", str(canonical_first)))
                    print(f"[h3cspeed-60s] segment {number:02d}: native generation")
                    run_logged(native_command, log, env=segment_env)
                    probe = validate_media(
                        ffmpeg, ffprobe, segment, args.width, args.height,
                        args.segment_frames,
                    )
                    verify_bound_inputs()
                if not resumed:
                    extract_last_frame(ffmpeg, segment, anchor, args.segment_frames - 1)
                record = {
                    "index": number,
                    "path": str(segment),
                    "sha256": sha256_file(segment),
                    "bytes": segment.stat().st_size,
                    "duration": float(probe.get("format", {}).get("duration", 0.0)),
                    "seed": args.base_seed + number - 1,
                    "sidecar_sha256": sha256_file(sidecar),
                    "input_anchor_sha256": (
                        sha256_file(anchors_root / f"segment-{number - 1:02d}-last.png")
                        if number > 1 else None
                    ),
                    "output_anchor_sha256": sha256_file(anchor),
                }
                completed.append(record)
                state["completed_segments"] = completed
                state["updated_unix"] = time.time()
                atomic_json(state_path, state)
                if args.stop_after and number >= args.stop_after:
                    state["status"] = "partial"
                    atomic_json(state_path, state)
                    print(f"[h3cspeed-60s] stopped after verified segment {number:02d}")
                    return 0

            verify_bound_inputs(deep_content_check=True)
            concat_lines = "".join(
                    f"file '{concat_path(segments_root / f'segment-{number:02d}.mp4')}'\n"
                    for number in range(1, args.segments + 1)
            )
            concat_directory, concat_file = private_temp(output_root, ".txt")
            concat_file.write_text(concat_lines, encoding="utf-8")
            final_output.parent.mkdir(parents=True, exist_ok=True)
            safe_regular(final_output, "final output", allow_missing=True)
            final_directory, temporary_final = private_temp(final_output.parent, ".mp4")
            final_command = build_final_command(ffmpeg, concat_file, temporary_final)
            print("[h3cspeed-60s] concatenating exact 60-second final")
            try:
                run_logged(final_command, logs_root / "final-concat.log")
                os.replace(temporary_final, final_output)
            finally:
                final_directory.cleanup()
                concat_directory.cleanup()
            final_probe = validate_media(
                ffmpeg, ffprobe, final_output, args.width, args.height,
                FINAL_FRAMES, exact_duration=60.0,
            )
            audio = validate_final_audio(ffmpeg, final_output)
            verify_bound_inputs(deep_content_check=True)
            state.update({
                "status": "complete",
                "final_output": str(final_output),
                "final_sha256": sha256_file(final_output),
                "final_bytes": final_output.stat().st_size,
                "final_duration": float(final_probe.get("format", {}).get("duration", 0.0)),
                "final_audio": audio,
                "updated_unix": time.time(),
            })
            atomic_json(state_path, state)
            print(f"[h3cspeed-60s] PASS {final_output}")
            return 0
        finally:
            lock.__exit__(None, None, None)
    except (OSError, RunError, ValueError) as exc:
        print(f"run_h3_quantized_60s: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
