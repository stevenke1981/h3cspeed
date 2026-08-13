#!/usr/bin/env python3
"""Validate immutable PERF-002 input manifests and per-engine results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_perf002_ab import (ContractError, canonical_bytes, load_input,
                            safe_regular, sha256_bytes, validate_result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    _, digest = load_input(args.manifest)
    if args.result:
        result_path = safe_regular(args.result, "result")
        payload = result_path.read_bytes()
        if payload != canonical_bytes(json.loads(payload)):
            raise ContractError("result is not canonical JSON")
        checksum_path = safe_regular(
            result_path.with_suffix(result_path.suffix + ".sha256"),
            "result checksum",
        )
        result_digest = sha256_bytes(payload)
        if checksum_path.read_text(encoding="ascii") != (
                f"{result_digest}  {result_path.name}\n"):
            raise ContractError("result checksum mismatch")
        validate_result(json.loads(payload), digest)
        print(f"PERF-002 result contract PASS: {args.result.name}")
    else:
        print(f"PERF-002 input contract PASS: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
