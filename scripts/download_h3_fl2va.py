"""Download and validate the pinned MiniMax-H3 FL2VA snapshot.

The helper intentionally keeps all of its working state on the model volume.
It does not remove an existing local checkout: Hugging Face's snapshot helper
can resume incomplete files from its cache/local directory, and validation is
performed only after a complete, pinned manifest has been fetched.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPOSITORY = "MiniMaxAI/MiniMax-H3"
# Do not make this configurable: the downloader is deliberately tied to the
# revision whose file sizes are checked below through the HF API.
REVISION = "939557dc319dd91227e30195a763f272ba7f8765"


def _path_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


# H3_MODEL_ROOT is the one path contract shared with the PowerShell launcher.
# Keep the historical E: default on Windows, but use the platform user cache
# on POSIX so direct invocation cannot create a literal ``E:\models`` folder.
DEFAULT_MODEL_ROOT = (
    Path(r"E:\models") if os.name == "nt"
    else Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
         / "h3cspeed" / "models"
)
MODEL_ROOT = _path_env("H3_MODEL_ROOT", DEFAULT_MODEL_ROOT)
LOCAL_DIR = _path_env("H3_LOCAL_DIR", MODEL_ROOT / "MiniMax-H3")
CACHE_DIR = _path_env("H3_CACHE_DIR", MODEL_ROOT / "hf-cache")
TEMP_DIR = _path_env("H3_TEMP_DIR", MODEL_ROOT / "hf-tmp")
XET_CACHE_DIR = _path_env("H3_XET_CACHE", MODEL_ROOT / "hf-xet")
HF_HOME_DIR = _path_env("H3_HF_HOME", MODEL_ROOT / "hf-home")
try:
    MAX_WORKERS = int(os.environ.get("H3_DOWNLOAD_WORKERS", "4"))
except ValueError:
    # Defer the error until main() so the failed invocation still gets a
    # timestamped status record instead of dying during module import.
    MAX_WORKERS = 0


def _configure_huggingface_environment() -> None:
    """Set model-volume defaults before *any* huggingface_hub import occurs.

    The H3_* variables are intentional: the PowerShell launcher can provide
    an explicit path (for example, a dedicated cache volume) and it wins.
    """

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    HF_HOME_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    XET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # The H3_* paths are the explicit override surface used by the launcher.
    # Assign the derived HF/TEMP variables so an inherited Windows TEMP or HF
    # cache cannot silently move this large job back to the system volume.
    os.environ["HF_HOME"] = str(HF_HOME_DIR)
    os.environ["HF_HUB_CACHE"] = str(CACHE_DIR)
    os.environ["HF_XET_CACHE"] = str(XET_CACHE_DIR)
    os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR / "datasets")
    os.environ["TEMP"] = str(TEMP_DIR)
    os.environ["TMP"] = str(TEMP_DIR)


# This call must remain before the lazy import in _download_snapshot and
# _fetch_manifest.  It is also safe when the module is imported by offline
# tests that replace huggingface_hub with a stub.
_configure_huggingface_environment()


REQUIRED_FILES = (
    "FL2VA/transformer/config.json",
    "FL2VA/tokenizer/tokenizer.json",
)
REQUIRED_DIRECTORIES = (
    "FL2VA/audio_vae",
    "FL2VA/text_encoder",
    "FL2VA/transformer",
    "FL2VA/video_vae/source",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


START_TIME = _utc_now()
STATUS_PATH = Path(
    os.environ.get(
        "H3_STATUS_PATH",
        str(
            MODEL_ROOT
            / f"download-h3-fl2va-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.status.json"
        ),
    )
)
LOG_PATH = os.environ.get("H3_LOG_PATH")
ERROR_LOG_PATH = os.environ.get("H3_ERROR_PATH")
LOCK_PATH = Path(
    os.environ.get("H3_LOCK_PATH", str(MODEL_ROOT / ".download-h3-fl2va.lock"))
)


class DuplicateInstanceError(RuntimeError):
    """Raised when another downloader owns the job-wide lock."""


class ValidationError(RuntimeError):
    """Raised when the local checkout is not a complete pinned snapshot."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details


class SingleInstanceLock:
    """Cross-platform advisory lock that remains valid until process exit.

    The lock file is deliberately retained (with the last owner metadata) so
    a failed run never destroys evidence.  The OS lock itself is released by
    ``release`` and also automatically when the process exits.
    """

    def __init__(self, path: Path):
        self.path = path
        self._handle: Any | None = None

    def acquire(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Do not use ``a+`` here: Windows opens it with O_APPEND and
        # msvcrt.locking can then fail even though the file is ours.  A plain
        # read/write descriptor works for both the advisory lock and metadata.
        descriptor = os.open(str(self.path), os.O_CREAT | os.O_RDWR)
        handle = os.fdopen(descriptor, "r+", encoding="utf-8")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(" ")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    try:
                        handle.seek(0)
                        owner = handle.read().strip()
                    except OSError:
                        owner = ""
                    raise DuplicateInstanceError(
                        f"another FL2VA download already owns {self.path}"
                        + (f": {owner}" if owner else "")
                    ) from exc
            else:
                import fcntl

                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    try:
                        handle.seek(0)
                        owner = handle.read().strip()
                    except OSError:
                        owner = ""
                    raise DuplicateInstanceError(
                        f"another FL2VA download already owns {self.path}"
                        + (f": {owner}" if owner else "")
                    ) from exc
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            handle.flush()
            self._handle = handle
        except Exception:
            handle.close()
            raise

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _base_status() -> dict[str, Any]:
    return {
        "state": "starting",
        "pid": os.getpid(),
        "repository": REPOSITORY,
        "revision": REVISION,
        "model_root": str(MODEL_ROOT),
        "local_dir": str(LOCAL_DIR),
        "cache_dir": str(CACHE_DIR),
        "hf_home_dir": str(HF_HOME_DIR),
        "temp_dir": str(TEMP_DIR),
        "xet_cache_dir": str(XET_CACHE_DIR),
        "lock_path": str(LOCK_PATH),
        "status_path": str(STATUS_PATH),
        "log_path": LOG_PATH,
        "error_log_path": ERROR_LOG_PATH,
        "workers": MAX_WORKERS,
        "start_time": START_TIME,
        "end_time": None,
        "exit_code": None,
        "validation": None,
    }


def _write_status(status: dict[str, Any]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS_PATH.with_name(STATUS_PATH.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, STATUS_PATH)


def _fetch_manifest() -> dict[str, int]:
    """Return the pinned FL2VA file -> size manifest from the HF API."""

    from huggingface_hub import HfApi

    entries: Iterable[Any] = HfApi().list_repo_tree(
        REPOSITORY,
        path_in_repo="FL2VA",
        recursive=True,
        expand=True,
        revision=REVISION,
    )
    manifest: dict[str, int] = {}
    for entry in entries:
        path = str(getattr(entry, "path", "")).replace("\\", "/")
        size = getattr(entry, "size", None)
        if not path.startswith("FL2VA/") or size is None:
            continue
        manifest[path] = int(size)
    if not manifest:
        raise ValidationError(
            "HF API returned no pinned FL2VA files; refusing to treat a local "
            "or offline partial checkout as successful"
        )
    return manifest


def _validate_download() -> dict[str, Any]:
    """Validate every API-listed file and the required H3 directory layout."""

    manifest = _fetch_manifest()
    missing: list[str] = []
    size_mismatch: list[dict[str, Any]] = []
    for relative, expected_size in sorted(manifest.items()):
        local_path = LOCAL_DIR.joinpath(*relative.split("/"))
        if not local_path.is_file():
            missing.append(relative)
            continue
        actual_size = local_path.stat().st_size
        if actual_size != expected_size:
            size_mismatch.append(
                {"path": relative, "expected": expected_size, "actual": actual_size}
            )

    required_files_missing = [path for path in REQUIRED_FILES if path not in manifest]
    required_layout_missing: list[str] = []
    for directory in REQUIRED_DIRECTORIES:
        local_directory = LOCAL_DIR.joinpath(*directory.split("/"))
        has_manifest_file = any(
            path.startswith(directory.rstrip("/") + "/") for path in manifest
        )
        if not local_directory.is_dir() or not has_manifest_file:
            required_layout_missing.append(directory)

    # Extra files are intentionally reported but never removed.
    expected_paths = set(manifest)
    extra_files: list[str] = []
    extra_safetensors: list[str] = []
    fl2va_root = LOCAL_DIR / "FL2VA"
    if fl2va_root.is_dir():
        for local_path in fl2va_root.rglob("*"):
            if local_path.is_file():
                relative = local_path.relative_to(LOCAL_DIR).as_posix()
                if relative not in expected_paths:
                    extra_files.append(relative)
                    if local_path.suffix.lower() == ".safetensors":
                        extra_safetensors.append(relative)

    validation: dict[str, Any] = {
        # Extra metadata is harmless and is retained, but an unlisted weight
        # shard could be picked up by a loader glob and silently mix revisions.
        # Fail closed without deleting anything so the operator can inspect it.
        "ok": not (
            missing
            or size_mismatch
            or required_files_missing
            or required_layout_missing
            or extra_safetensors
        ),
        "manifest_source": "huggingface_hub.HfApi.list_repo_tree",
        "manifest_revision": REVISION,
        "expected_file_count": len(manifest),
        "manifest_sizes": dict(sorted(manifest.items())),
        "missing": missing,
        "size_mismatch": size_mismatch,
        "required_files_missing": required_files_missing,
        "required_layout_missing": required_layout_missing,
        "extra_files_not_deleted": sorted(extra_files),
        "extra_safetensors_blocking": sorted(extra_safetensors),
    }
    if not validation["ok"]:
        raise ValidationError(
            json.dumps(validation, ensure_ascii=False, sort_keys=True), validation
        )
    return validation


def _download_snapshot() -> str:
    # Lazy import is required: cache/Xet/temp defaults above must be in place
    # before huggingface_hub reads its environment configuration.
    from huggingface_hub import snapshot_download

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result = snapshot_download(
        repo_id=REPOSITORY,
        revision=REVISION,
        local_dir=str(LOCAL_DIR),
        cache_dir=str(CACHE_DIR),
        allow_patterns=["FL2VA/**"],
        ignore_patterns=["Ref2VA/**"],
        max_workers=MAX_WORKERS,
        force_download=False,
    )
    return str(result)


def main() -> int:
    status = _base_status()
    _write_status(status)
    lock = SingleInstanceLock(LOCK_PATH)
    try:
        if not 1 <= MAX_WORKERS <= 64:
            raise ValueError(f"H3_DOWNLOAD_WORKERS must be between 1 and 64, got {MAX_WORKERS}")
        lock.acquire(
            {
                "pid": os.getpid(),
                "revision": REVISION,
                "local_dir": str(LOCAL_DIR),
                "start_time": START_TIME,
            }
        )
        status["state"] = "running"
        _write_status(status)
        result = _download_snapshot()
        print(f"snapshot_download_result={result}", flush=True)
        validation = _validate_download()
        status["validation"] = validation
        status["state"] = "completed"
        status["exit_code"] = 0
        status["end_time"] = _utc_now()
        _write_status(status)
        print(
            f"completed revision={REVISION} files={validation['expected_file_count']}",
            flush=True,
        )
        return 0
    except DuplicateInstanceError as exc:
        status["state"] = "lock_conflict"
        status["exit_code"] = 2
        status["end_time"] = _utc_now()
        status["validation"] = {"ok": False, "error": str(exc)}
        _write_status(status)
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    except Exception as exc:
        status["state"] = "failed"
        status["exit_code"] = 1
        status["end_time"] = _utc_now()
        details = getattr(exc, "details", None)
        status["validation"] = details or {"ok": False, "error": str(exc)}
        _write_status(status)
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
