#!/usr/bin/env python3
"""Validate h3cspeed CUDA profile JSON reports without third-party packages."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
EVICTION_REASONS = ("capacity_lru", "phase_retire", "error_cleanup")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _nonnegative_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return converted


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def validate_report(report: Any) -> dict[str, Any]:
    root = _mapping(report, "report")
    if root.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    if root.get("kind") != "h3cspeed.cuda.profile":
        raise ValueError("unexpected report kind")

    context = _mapping(root.get("context"), "context")
    _nonnegative_integer(context.get("id"), "context.id")
    _nonnegative_integer(context.get("pid"), "context.pid")
    _nonnegative_integer(context.get("device"), "context.device")
    if not isinstance(context.get("label"), str) or not context["label"]:
        raise ValueError("context.label must be a non-empty string")
    if not isinstance(context.get("sm"), str) or not context["sm"].isdigit():
        raise ValueError("context.sm must be numeric text")
    if not isinstance(context.get("complete"), bool):
        raise ValueError("context.complete must be boolean")

    wall = _mapping(root.get("wall"), "wall")
    wall_seconds = _nonnegative_number(wall.get("seconds"), "wall.seconds")
    accounted = _nonnegative_number(
        wall.get("accounted_host_seconds"), "wall.accounted_host_seconds"
    )
    ratio = _nonnegative_number(wall.get("accounted_ratio"), "wall.accounted_ratio")
    if ratio > 1.0:
        raise ValueError("wall.accounted_ratio must not exceed one")
    for key in ("coverage_gate_valid", "coverage_gate_met"):
        if not isinstance(wall.get(key), bool):
            raise ValueError(f"wall.{key} must be boolean")
    if wall["coverage_gate_met"] and not wall["coverage_gate_valid"]:
        raise ValueError("coverage gate cannot pass without valid attribution")
    if wall_seconds == 0.0 and accounted != 0.0:
        raise ValueError("non-zero accounting requires non-zero wall time")
    expected_ratio = min(accounted / wall_seconds, 1.0) if wall_seconds else 0.0
    if not math.isclose(ratio, expected_ratio, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("wall.accounted_ratio is inconsistent")
    if wall["coverage_gate_valid"] and (
        wall["coverage_gate_met"] != (ratio >= 0.95)
    ):
        raise ValueError("wall.coverage_gate_met is inconsistent")

    timing = _mapping(root.get("timing"), "timing")
    for key in (
        "file_read_seconds",
        "pageable_copy_seconds",
        "h2d_enqueue_seconds",
        "compute_stream_wait_seconds",
        "upload_stream_wait_seconds",
        "event_wait_seconds",
        "allocation_seconds",
        "eviction_seconds",
        "compute_device_seconds",
    ):
        _nonnegative_number(timing.get(key), f"timing.{key}")

    counts = _mapping(root.get("counts"), "counts")
    for key in (
        "begin",
        "continue",
        "submit_sync",
        "last_use_fence",
        "file_read",
        "pageable_copy",
        "h2d_enqueue",
        "compute_stream_sync",
        "upload_stream_sync",
        "event_sync",
        "allocation",
        *EVICTION_REASONS,
    ):
        _nonnegative_integer(counts.get(key), f"counts.{key}")

    byte_counts = _mapping(root.get("bytes"), "bytes")
    for key in ("file_read", "pageable_copy", "h2d", *EVICTION_REASONS):
        _nonnegative_integer(byte_counts.get(key), f"bytes.{key}")

    for section_name in ("memory", "offload", "dispatches"):
        section = _mapping(root.get(section_name), section_name)
        if not section:
            raise ValueError(f"{section_name} must not be empty")
        for key, value in section.items():
            _nonnegative_integer(value, f"{section_name}.{key}")

    validity = _mapping(root.get("validity"), "validity")
    if validity.get("critical_path_scope") != "gpu_context_host_operations":
        raise ValueError("invalid critical_path_scope")
    for key in ("h2d_device_seconds", "compute_device_seconds"):
        if not isinstance(validity.get(key), bool):
            raise ValueError(f"validity.{key} must be boolean")

    non_additive = root.get("non_additive")
    if not isinstance(non_additive, list) or not all(
        isinstance(item, str) and item for item in non_additive
    ):
        raise ValueError("non_additive must be a list of strings")
    if "device_and_host_timings_may_overlap" not in non_additive:
        raise ValueError("overlap disclosure is required")
    return root


def validate_path(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return validate_report(json.load(stream))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    for report_path in args.reports:
        validate_path(report_path)
        print(f"profile report PASS: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
