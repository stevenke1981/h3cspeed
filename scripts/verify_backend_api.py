#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

FUNCTION = re.compile(r'\b(?:extern\s+"C"\s+)?(?:[A-Za-z_][\w\s*]+?)\b(h3_gpu_[A-Za-z0-9_]+)\s*\(')


def names_in(paths: list[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        result.update(FUNCTION.findall(path.read_text(encoding="utf-8")))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--header", type=Path,
                        help="prepared upstream h3_gpu.h; otherwise use pinned symbol list")
    args = parser.parse_args()
    root = args.root.resolve()
    sources = sorted((root / "src").glob("h3_gpu_cuda*.cu"))
    implemented = names_in(sources)
    if args.header:
        expected = names_in([args.header])
    else:
        expected = {
            line.strip() for line in (root / "tests/backend_api_symbols.txt")
            .read_text(encoding="utf-8").splitlines() if line.strip()
        }
    missing = sorted(expected - implemented)
    unexpected = sorted(implemented - expected)
    print(f"expected: {len(expected)}")
    print(f"implemented: {len(implemented)}")
    if missing:
        print("missing:")
        for name in missing: print(f"  {name}")
    if unexpected:
        print("unexpected overlay symbols:")
        for name in unexpected: print(f"  {name}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
