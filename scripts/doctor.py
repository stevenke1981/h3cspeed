#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import ctypes


def version(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=False, text=True,
                                encoding="utf-8", errors="replace",
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = result.stdout.splitlines() if result.stdout else []
        if result.returncode and not (
                os.path.basename(command[0]).lower() in {"cl", "cl.exe"}
                and result.returncode == 2 and output):
            return f"ERROR: exit {result.returncode}"
        return output[0] if output else "ok"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


def memory_gib(key: str) -> float:
    if sys.platform == "win32":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            value = (status.ullAvailPhys if key == "MemAvailable"
                     else status.ullTotalPhys)
            return value / (1024 ** 3)
        return 0.0
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            label, _, value = line.partition(":")
            if label == key:
                kib = int(value.strip().split()[0])
                return kib / (1024 ** 2)
    except (OSError, ValueError, IndexError):
        pass
    try:
        page_key = "SC_AVPHYS_PAGES" if key == "MemAvailable" else "SC_PHYS_PAGES"
        pages = os.sysconf(page_key)
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / (1024 ** 3)
    except (ValueError, OSError, AttributeError):
        return 0.0


def gpu_inventory() -> tuple[str, float | None, float | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,compute_cap,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}", None, None
    line = result.stdout.splitlines()[0].strip() if result.stdout else ""
    fields = [item.strip() for item in line.rsplit(",", 2)]
    if len(fields) != 3:
        return line or "unparseable nvidia-smi response", None, None
    try:
        compute_capability = float(fields[1])
        vram_gib = float(fields[2]) / 1024.0
    except ValueError:
        return line, None, None
    return line, vram_gib, compute_capability


def main() -> int:
    checks = {
        "cmake": ["cmake", "--version"],
        "ninja": ["ninja", "--version"],
        "C compiler": ["cl", "/Bv"] if sys.platform == "win32"
        else ["cc", "--version"],
        "nvcc": ["nvcc", "--version"],
        "pkg-config": ["pkg-config", "--version"],
        "ICU": ["pkg-config", "--modversion", "icu-uc"],
        "ffmpeg": ["ffmpeg", "-version"],
        "ffprobe": ["ffprobe", "-version"],
    }
    required = {"cmake", "C compiler", "nvcc", "ICU"}
    failed = False
    for label, command in checks.items():
        if label == "ICU" and sys.platform == "win32":
            root = Path(os.environ.get(
                "H3CSPEED_ICU_ROOT",
                Path(__file__).resolve().parents[1] / "third_party" / "icu",
            ))
            if ((root / "include" / "unicode" / "uchar.h").is_file() and
                    any((root / "lib64" / name).is_file()
                        for name in ("icuuc.lib", "icuuc76.lib")) and
                    (root / "bin64" / "icuuc76.dll").is_file() and
                    (root / "bin64" / "icudt76.dll").is_file()):
                print(f"[check]   ICU: bundled {root}")
                continue
        if shutil.which(command[0]) is None:
            print(f"[missing] {label}: {command[0]}")
            if label in required:
                failed = True
            continue
        value = version(command)
        print(f"[check]   {label}: {value}")
        if value.startswith("ERROR") and label in required:
            failed = True

    gpu_gib: float | None = None
    compute_capability: float | None = None
    if shutil.which("nvidia-smi") is None:
        print("[missing] NVIDIA driver: nvidia-smi")
        failed = True
    else:
        inventory, gpu_gib, compute_capability = gpu_inventory()
        print(f"[check]   NVIDIA driver: {inventory}")
        if inventory.startswith("ERROR") or gpu_gib is None:
            failed = True
        if compute_capability is not None and compute_capability < 8.0:
            print("[error]   BF16 backend requires CUDA compute capability 8.0+")
            failed = True

    total_ram_gib = memory_gib("MemTotal")
    available_ram_gib = memory_gib("MemAvailable")
    print(f"[check]   visible system RAM: {total_ram_gib:.1f} GiB")
    print(f"[check]   currently available RAM: {available_ram_gib:.1f} GiB")
    if gpu_gib is not None and gpu_gib <= 10.0:
        print("[profile] low-VRAM offload will be enabled automatically")
        if sys.platform == "win32":
            print("[profile] build with sm_86 for RTX 3070 Ti: "
                  r".\scripts\build-native.ps1 -CudaArchitectures 86")
        else:
            print("[profile] build with sm_86 for RTX 3070 Ti: "
                  "H3CSPEED_CUDA_ARCHITECTURES=86 ./scripts/build.sh")
        if available_ram_gib < 48.0:
            print("[warning] fewer than 48 GiB is currently available; generated "
                  "INT8 RAM backing may not fit. Close other workloads, increase "
                  "the WSL2 limit when applicable, or use a larger host.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
