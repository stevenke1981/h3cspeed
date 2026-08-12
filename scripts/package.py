#!/usr/bin/env python3
"""Create a deterministic source archive without rsync or generated build trees."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import zipfile


EXCLUDED_PARTS = {
    ".git", ".idea", ".vscode", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".cache", "_deps", "outputs",
    "third_party", "logs", "tmp", "temp",
}

# Build directories are deliberately matched by prefix.  The native Windows
# workflow uses names such as ``build-native-qwen-final`` and a literal
# ``build`` entry would accidentally package several gigabytes of products.
EXCLUDED_DIR_PREFIXES = ("build", "cache", "log")

EXCLUDED_SUFFIXES = (
    ".a", ".bin", ".d", ".dll", ".dylib", ".exe", ".exp", ".gch",
    ".ilk", ".lib", ".map", ".o", ".obj", ".pch", ".pdb", ".so",
    ".wasm", ".dmp", ".log", ".trace",
    ".mp4", ".mkv", ".mov", ".webm", ".zip", ".sha256", ".pyc",
    ".h3c", ".bf16",
    ".pyo", ".safetensors", ".gguf", ".ggml", ".pt", ".pth", ".ckpt",
    ".pem", ".secret", ".token", ".key", ".p12", ".pfx",
)

EXCLUDED_SECRET_NAMES = {
    ".npmrc", ".netrc", "credentials.json", "token.json",
    "id_rsa", "id_ed25519",
}


def excluded(relative: Path) -> bool:
    parts = tuple(part.lower() for part in relative.parts)
    if any(part in EXCLUDED_PARTS or part.startswith(EXCLUDED_DIR_PREFIXES)
           for part in parts):
        return True
    name = relative.name.lower()
    if (name in EXCLUDED_SECRET_NAMES or name.startswith((
            "hf_token", "huggingface_token"))):
        return True
    if name == ".env" or (name.startswith(".env.") and
                            name != ".env.example"):
        return True
    return name.endswith(EXCLUDED_SUFFIXES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        # Do not follow a repository symlink into an external model, cache or
        # credential tree while assembling a public archive.
        if path.is_symlink() or excluded(relative) or path.is_dir():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def manifest_files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace(os.sep, "/"): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    }


def write_zip(source: Path, archive: Path, epoch: int) -> None:
    timestamp = datetime.fromtimestamp(epoch, timezone.utc)
    # ZIP cannot encode dates before 1980.
    date_time = (max(timestamp.year, 1980), timestamp.month, timestamp.day,
                 timestamp.hour, timestamp.minute, timestamp.second)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=9) as bundle:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source.parent).as_posix()
            info = zipfile.ZipInfo(relative, date_time=date_time)
            info.create_system = 3  # Unix: make external_attr modes portable.
            mode = path.stat().st_mode
            # Windows source trees do not retain POSIX execute bits. Preserve
            # executable semantics for shipped scripts based on their shebang.
            executable = bool(mode & stat.S_IXUSR)
            if not executable and path.suffix.lower() in {".sh", ".py"}:
                executable = path.read_bytes().startswith(b"#!")
            info.external_attr = ((0o755 if executable else 0o644) & 0xFFFF) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, path.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--version")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    version = args.version or (root / "VERSION").read_text().strip()
    output = (args.output or root.parent / f"h3cspeed-v{version}.zip").resolve()
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "1786476000"))

    with tempfile.TemporaryDirectory(prefix="h3cspeed-package-") as temporary:
        stage = Path(temporary) / "h3cspeed"
        stage.mkdir()
        copy_tree(root, stage)
        provenance = {
            "name": "h3cspeed",
            "version": version,
            "upstream_h3_commit":
                "8974cc055ea9c02fcd14cc27dfda3e1027c05153",
            "llama_cpp_reference_commit":
                "f785fc9ea485e6cfdda129978310aa52939c3619",
            "model_repository": "MiniMaxAI/MiniMax-H3",
            "model_revision":
                "939557dc319dd91227e30195a763f272ba7f8765",
            "source_date_epoch": epoch,
            "generated_utc": datetime.fromtimestamp(
                epoch, timezone.utc).replace(microsecond=0).isoformat(),
        }
        (stage / "SOURCE_MANIFEST.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        files = manifest_files(stage)
        sums = "".join(f"{digest}  {name}\n" for name, digest in files.items())
        (stage / "SHA256SUMS").write_text(sums, encoding="utf-8")
        write_zip(stage, output, epoch)

    digest = sha256(output)
    checksum = output.with_suffix(output.suffix + ".sha256")
    checksum.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    print(output)
    print(checksum)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
