#!/usr/bin/env python3
"""Run every validation possible without an NVIDIA GPU or MiniMax-H3 weights."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile


def run(command: list[str], root: Path) -> tuple[bool, str]:
    display = " ".join(command)
    completed = subprocess.run(
        command, cwd=root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    output = completed.stdout.rstrip()
    return completed.returncode == 0, f"$ {display}\n{output}".rstrip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    scripts = sorted(str(path.relative_to(root)) for path in
                     (root / "scripts").glob("*.py"))
    shell_scripts = sorted(str(path.relative_to(root)) for path in
                           (root / "scripts").glob("*.sh"))

    records: list[tuple[str, bool, str]] = []
    commands = [
        ("Python bytecode", [sys.executable, "-m", "py_compile", *scripts]),
        ("Backend API coverage", [sys.executable,
                                  "scripts/verify_backend_api.py"]),
        ("CUDA/C source syntax", [sys.executable,
                                  "scripts/source_syntax_lint.py"]),
        ("Python unit tests", [sys.executable, "-m", "unittest", "discover",
                               "-s", "tests", "-p", "test_*.py", "-v"]),
    ]
    if shutil.which("bash"):
        commands.append(("Shell syntax", ["bash", "-n", *shell_scripts]))

    with tempfile.TemporaryDirectory(prefix="h3cspeed-validation-") as temporary:
        build = Path(temporary) / "build"
        commands.extend([
            ("CMake portable configure", ["cmake", "-S", str(root),
                                           "-B", str(build),
                                           "-DH3CSPEED_OVERLAY_TESTS_ONLY=ON"]),
            ("CMake portable build", ["cmake", "--build", str(build),
                                      "--parallel"]),
            ("CTest portable suite", ["ctest", "--test-dir", str(build),
                                      "--output-on-failure"]),
        ])
        for name, command in commands:
            ok, output = run(command, root)
            records.append((name, ok, output))
            print(f"[{'pass' if ok else 'FAIL'}] {name}")
            if not ok:
                print(output, file=sys.stderr)

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    report_lines = [
        "# h3cspeed local validation results",
        "",
        f"- Generated: `{now}`",
        f"- Platform: `{platform.platform()}`",
        f"- Python: `{platform.python_version()}`",
        f"- nvcc available: `{'yes' if shutil.which('nvcc') else 'no'}`",
        f"- nvidia-smi available: `{'yes' if shutil.which('nvidia-smi') else 'no'}`",
        "",
        "## Results",
        "",
    ]
    for name, ok, output in records:
        report_lines.extend([
            f"### {'PASS' if ok else 'FAIL'} — {name}",
            "",
            "```text",
            output,
            "```",
            "",
        ])
    report_lines.extend([
        "## Not executed in this environment",
        "",
        "- nvcc compilation and link against a real CUDA Toolkit;",
        "- CUDA kernel execution on an NVIDIA GPU;",
        "- upstream BF16/int8 fixture parity;",
        "- MiniMax-H3 model loading and video/audio generation;",
        "- RTX 3070 Ti 8 GiB VRAM/RAM-offload and performance measurements.",
        "",
        "Those remain mandatory gates in `docs/VALIDATION.md`; passing the local",
        "checks does not convert this engineering preview into a production release.",
        "",
    ])
    report = "\n".join(report_lines)
    destination = args.report or (root / "VALIDATION_RESULTS.md")
    destination.write_text(report, encoding="utf-8")
    print(f"report: {destination}")
    return 0 if all(ok for _, ok, _ in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
