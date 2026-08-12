#!/usr/bin/env python3
"""Build deterministic, model-free h3cspeed runtime archives."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import zipfile


EPOCH = int(os.environ.get("SOURCE_DATE_EPOCH", "1786476000"))
FORBIDDEN_SUFFIXES = {
    ".bf16", ".bin", ".ckpt", ".ggml", ".gguf", ".h3c", ".mp4",
    ".npy", ".npz", ".onnx", ".pt", ".pth", ".safetensors",
}
COMMON_FILES = (
    "LICENSE", "THIRD_PARTY_NOTICES.md", "PROVENANCE.md", "VERSION",
    "README.md", "README.zh-TW.md",
)
COMMON_SCRIPTS = (
    "scripts/doctor.py",
    "scripts/download_h3_fl2va.py",
    "scripts/encode_h3_quantized_prompt.py",
    "scripts/prepare_h3_quantized_model.py",
    "scripts/run-h3-quantized.ps1",
    "scripts/run-3070ti-8gb.sh",
    "scripts/smoke-3070ti-8gb.sh",
    "scripts/fast-quality-3070ti-8gb.sh",
)
COMMON_PROFILES = (
    "profiles/rtx3070ti-8gb.env",
    "profiles/rtx3070ti-fast-quality.env",
)
COMMON_LICENSES = (
    "licenses/antirez-h3.c-LICENSE",
    "licenses/yyjson-LICENSE",
)
LINUX_PRIVATE_PREFIXES = (
    "libcudart.so", "libcublas.so", "libcublasLt.so",
    "libicuuc.so", "libicudata.so",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path, executable: bool = False) -> None:
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"required runtime input is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if executable:
        destination.chmod(destination.stat().st_mode | 0o755)


def copy_common(root: Path, bundle: Path) -> None:
    for relative in COMMON_FILES + COMMON_SCRIPTS + COMMON_PROFILES:
        source = root / relative
        copy_file(source, bundle / relative,
                  executable=source.suffix in {".py", ".sh"})
    for relative in COMMON_LICENSES:
        copy_file(root / relative, bundle / relative)


def one_match(directory: Path, pattern: str) -> Path:
    matches = sorted(path for path in directory.glob(pattern) if path.is_file())
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {pattern} below {directory}, found {matches}")
    return matches[0]


def find_icu_license(root: Path) -> Path:
    for candidate in (
        root / "third_party/icu/LICENSE",
        root / "third_party/icu/LICENSE.txt",
        Path("/usr/share/doc/libicu-dev/copyright"),
    ):
        if candidate.is_file():
            return candidate
    raise RuntimeError("ICU license was not found")


def find_cuda_eula(cuda: Path) -> Path:
    candidates = (cuda / "EULA.txt", cuda.resolve() / "EULA.txt")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"CUDA Toolkit EULA was not found below {cuda}")


def stage_windows(root: Path, install: Path, cuda: Path, bundle: Path) -> list[str]:
    installed_bin = install / "bin"
    output_bin = bundle / "bin"
    runtime_names = (
        "h3cspeed.exe", "h3cspeed-cuda-info.exe",
        "icuuc76.dll", "icudt76.dll",
    )
    for name in runtime_names:
        copy_file(installed_bin / name, output_bin / name)

    cuda_bin = cuda / "bin/x64"
    if not cuda_bin.is_dir():
        cuda_bin = cuda / "bin"
    cuda_libraries = (
        one_match(cuda_bin, "cublas64_*.dll"),
        one_match(cuda_bin, "cublasLt64_*.dll"),
    )
    for library in cuda_libraries:
        copy_file(library, output_bin / library.name)

    copy_file(find_icu_license(root), bundle / "licenses/ICU-LICENSE.txt")
    copy_file(find_cuda_eula(cuda), bundle / "licenses/NVIDIA-CUDA-EULA.txt")
    return list(runtime_names) + [item.name for item in cuda_libraries]


def ldd_paths(binary: Path, cuda: Path) -> dict[str, Path]:
    environment = dict(os.environ)
    cuda_lib = cuda / "lib64"
    environment["LD_LIBRARY_PATH"] = str(cuda_lib) + os.pathsep + environment.get(
        "LD_LIBRARY_PATH", "")
    pending = [binary]
    found: dict[str, Path] = {}
    while pending:
        target = pending.pop()
        result = subprocess.run(
            ["ldd", str(target)], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=environment,
        )
        if "not found" in result.stdout:
            raise RuntimeError(f"unresolved Linux runtime dependency:\n{result.stdout}")
        for line in result.stdout.splitlines():
            match = re.match(r"\s*(\S+)\s+=>\s+(\S+)\s+\(", line)
            if not match:
                continue
            soname, value = match.groups()
            if not soname.startswith(LINUX_PRIVATE_PREFIXES) or soname in found:
                continue
            path = Path(value)
            if not path.is_file():
                raise RuntimeError(f"ldd resolved {soname} to missing {path}")
            found[soname] = path
            pending.append(path)
    required = ("libcudart.so", "libcublas.so", "libcublasLt.so",
                "libicuuc.so", "libicudata.so")
    for prefix in required:
        if not any(name.startswith(prefix) for name in found):
            raise RuntimeError(f"Linux dependency closure lacks {prefix}")
    return found


def linux_launcher(name: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/.." && pwd)"
export LD_LIBRARY_PATH="$ROOT/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
exec "$ROOT/libexec/{name}" "$@"
"""


def stage_linux(root: Path, install: Path, cuda: Path, bundle: Path) -> list[str]:
    output_libexec = bundle / "libexec"
    output_lib = bundle / "lib"
    runtime: list[str] = []
    for name in ("h3cspeed", "h3cspeed-cuda-info"):
        source = install / "bin" / name
        copy_file(source, output_libexec / name, executable=True)
        launcher = bundle / "bin" / name
        launcher.parent.mkdir(parents=True, exist_ok=True)
        launcher.write_text(linux_launcher(name), encoding="utf-8", newline="\n")
        launcher.chmod(0o755)
        runtime.extend((f"bin/{name}", f"libexec/{name}"))

    dependencies = ldd_paths(output_libexec / "h3cspeed", cuda)
    for soname, source in sorted(dependencies.items()):
        copy_file(source.resolve(), output_lib / soname)
        runtime.append(f"lib/{soname}")

    copy_file(find_icu_license(root), bundle / "licenses/ICU-LICENSE.txt")
    copy_file(find_cuda_eula(cuda), bundle / "licenses/NVIDIA-CUDA-EULA.txt")
    return runtime


def runtime_readme(platform: str) -> str:
    common = """# h3cspeed binary runtime

This model-free engineering-preview archive targets NVIDIA CUDA compute
capability 8.6 (`sm_86`). Model weights, prompt sidecars, generated media, FFmpeg and
FFprobe are intentionally not included.

Required on the host:

- a compatible NVIDIA driver and an `sm_86` NVIDIA GPU;
- `ffmpeg` and `ffprobe` on PATH, or `H3_FFMPEG` / `H3_FFPROBE`;
- model weights stored outside this archive.
"""
    if platform == "windows-x86_64":
        return common + """
Windows additionally requires the Microsoft Visual C++ 2015-2022 x64
Redistributable. ICU 76 and the CUDA 13.2 cuBLAS runtime are bundled.
Run `bin\\h3cspeed.exe --help` to verify process startup.
"""
    return common + """
This build uses an Ubuntu 22.04 glibc baseline. CUDA 13.2 and ICU runtime
libraries are private to the archive; glibc, libstdc++ and the NVIDIA driver
remain system components. Run `bin/h3cspeed --help` to verify startup.
"""


def check_forbidden(bundle: Path) -> None:
    forbidden = []
    for path in bundle.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(bundle)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or any(
                part.lower() in {"models", "outputs", "third_party", "build"}
                for part in relative.parts):
            forbidden.append(relative.as_posix())
    if forbidden:
        raise RuntimeError(f"forbidden runtime payloads: {forbidden}")


def git_provenance(root: Path, allow_dirty: bool) -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain",
             "--untracked-files=all"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        raise RuntimeError("runtime packages require a readable Git checkout")
    dirty = bool(status)
    if dirty and not allow_dirty:
        raise RuntimeError(
            "runtime packages require a clean Git checkout; commit or stash changes")
    return commit, dirty


def write_metadata(root: Path, bundle: Path, platform: str, runtime: list[str],
                   allow_dirty: bool) -> None:
    commit, dirty = git_provenance(root, allow_dirty)
    files = {
        path.relative_to(bundle).as_posix(): sha256(path)
        for path in sorted(bundle.rglob("*")) if path.is_file()
    }
    metadata = {
        "architecture": "sm_86",
        "cuda": "13.2",
        "git_commit": commit,
        "git_dirty": dirty,
        "model_payloads_included": False,
        "platform": platform,
        "runtime_files": sorted(runtime),
        "version": (root / "VERSION").read_text(encoding="utf-8").strip(),
    }
    (bundle / "RUNTIME_MANIFEST.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files["RUNTIME_MANIFEST.json"] = sha256(bundle / "RUNTIME_MANIFEST.json")
    (bundle / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(files.items())),
        encoding="utf-8", newline="\n")


def add_zip_tree(bundle: Path, output: Path) -> None:
    timestamp = (2026, 8, 13, 0, 0, 0)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for path in sorted(bundle.rglob("*")):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(
                (Path(bundle.name) / path.relative_to(bundle)).as_posix(),
                date_time=timestamp,
            )
            info.create_system = 3
            mode = 0o755 if os.access(path, os.X_OK) else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes(), compresslevel=9)


def add_tar_tree(bundle: Path, output: Path) -> None:
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=EPOCH) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path in sorted((bundle.parent).rglob("*")):
                    if path == bundle.parent:
                        continue
                    relative = path.relative_to(bundle.parent)
                    if relative.parts[0] != bundle.name:
                        continue
                    if path.is_symlink():
                        raise RuntimeError(f"runtime archive cannot contain symlink: {path}")
                    info = archive.gettarinfo(str(path), arcname=relative.as_posix())
                    info.uid = info.gid = 0
                    info.uname = info.gname = "root"
                    info.mtime = EPOCH
                    if path.is_file():
                        with path.open("rb") as source:
                            archive.addfile(info, source)
                    else:
                        archive.addfile(info)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--install-dir", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--platform", choices=("windows-x86_64", "linux-x86_64"),
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true",
                        help="development only: package a dirty tree and record it")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.source_root.resolve()
    install = args.install_dir.resolve()
    cuda = args.cuda_root.resolve()
    output = args.output.resolve()
    version = (root / "VERSION").read_text(encoding="utf-8").strip()
    expected_suffix = ".zip" if args.platform == "windows-x86_64" else ".tar.gz"
    if not str(output).endswith(expected_suffix):
        raise RuntimeError(f"{args.platform} output must end in {expected_suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)

    name = f"h3cspeed-v{version}-{args.platform}-cuda13.2-sm86"
    with tempfile.TemporaryDirectory(prefix="h3cspeed-runtime-") as temporary:
        bundle = Path(temporary) / name
        bundle.mkdir()
        copy_common(root, bundle)
        runtime = (stage_windows(root, install, cuda, bundle)
                   if args.platform == "windows-x86_64"
                   else stage_linux(root, install, cuda, bundle))
        (bundle / "README-RUNTIME.md").write_text(
            runtime_readme(args.platform), encoding="utf-8", newline="\n")
        check_forbidden(bundle)
        write_metadata(root, bundle, args.platform, runtime, args.allow_dirty)
        if args.platform == "windows-x86_64":
            add_zip_tree(bundle, output)
        else:
            add_tar_tree(bundle, output)

    checksum = output.with_suffix(output.suffix + ".sha256")
    checksum.write_text(f"{sha256(output)}  {output.name}\n", encoding="ascii")
    print(output)
    print(checksum)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
