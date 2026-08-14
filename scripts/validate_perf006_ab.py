#!/usr/bin/env python3
"""Validate a matched PERF-006 DiT-prefetch smoke pair.

The two input files are sanitized single-engine smoke results.  This validator
does not launch an engine and never accepts aggregate profile wait counters as
the PERF-006 gate: both trials must carry complete DiT upload-ready wait
evidence emitted by the same binary and normalized command contract.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_perf002_ab import (  # noqa: E402
    ContractError, publish_json, safe_regular, sha256_file,
)


SCHEMA_VERSION = 1
KIND = "h3cspeed.perf006.ab"
SMOKE_KIND = "h3cspeed.perf002.smoke"
SHA_KEYS = (
    "input_manifest_sha256", "bindings_sha256",
)
SHA256_DIGITS = frozenset("0123456789abcdef")
PREFETCH_COUNTERS = (
    "prefetch_reserve_count", "prefetch_upload_count",
    "prefetch_consume_count", "prefetch_cancel_count",
    "prefetch_error_count", "prefetch_block_count",
)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    return value


def _load(path: Path, label: str) -> tuple[dict[str, Any], str]:
    source = safe_regular(path, label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{label} is not valid UTF-8 JSON") from error
    return _mapping(value, label), sha256_file(source)


def _sha256(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in SHA256_DIGITS for character in value)):
        raise ContractError(f"{label} must be a lowercase SHA-256")
    return value


def _trial(result: dict[str, Any], variant: str) -> dict[str, Any]:
    if (result.get("schema_version") != 1 or
            result.get("kind") != SMOKE_KIND or
            result.get("engine") != "h3cspeed" or
            result.get("status") != "SMOKE_PASS"):
        raise ContractError(f"{variant} is not a passing h3cspeed smoke result")
    contract = _mapping(result.get("contract"), f"{variant}.contract")
    if contract != {"width": 864, "height": 480, "frames": 22, "fps": 24,
                    "steps": 2, "layers": 50, "seed": 42}:
        raise ContractError(f"{variant} does not use the PERF-006 smoke contract")
    evidence = _mapping(result.get("perf006_evidence"),
                        f"{variant}.perf006_evidence")
    expected_mode = "disabled" if variant == "baseline" else "one_ahead_convrot"
    if evidence.get("variant") != variant or evidence.get("dit_prefetch_mode") != expected_mode:
        raise ContractError(f"{variant} route marker is invalid")
    for key in ("async_refill_requested", "async_refill_active",
                "upload_wait_trace_requested", "upload_wait_trace_complete",
                "union_valid"):
        if evidence.get(key) is not True:
            raise ContractError(f"{variant} {key} must be true")
    if evidence.get("upload_wait_trace_overflow") is not False:
        raise ContractError(f"{variant} wait trace overflowed")
    if evidence.get("ssd_streaming") is not False or evidence.get("scope") != "dit_denoise":
        raise ContractError(f"{variant} did not use the non-SSD DiT scope")
    wait = evidence.get("exclusive_upload_ready_wait_seconds")
    count = evidence.get("upload_ready_wait_count")
    if (isinstance(wait, bool) or not isinstance(wait, (int, float)) or
            not math.isfinite(float(wait)) or float(wait) < 0 or
            isinstance(count, bool) or not isinstance(count, int) or count < 0):
        raise ContractError(f"{variant} wait evidence is invalid")
    binary_sha256 = _sha256(evidence.get("binary_sha256"),
                            f"{variant}.binary_sha256")
    matched_contract_sha256 = _sha256(
        evidence.get("matched_contract_sha256"),
        f"{variant}.matched_contract_sha256")
    profile_sha256 = _sha256(evidence.get("profile_sha256"),
                             f"{variant}.profile_sha256")
    counters: dict[str, int] = {}
    for key in PREFETCH_COUNTERS:
        value = evidence.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContractError(f"{variant}.{key} must be a nonnegative integer")
        counters[key] = value
    if variant == "baseline":
        if any(counters.values()):
            raise ContractError("baseline unexpectedly used DiT prefetch")
    elif (counters["prefetch_reserve_count"] <= 0 or
          counters["prefetch_upload_count"] <= 0 or
          counters["prefetch_consume_count"] <= 0 or
          counters["prefetch_block_count"] <= 0 or
          counters["prefetch_error_count"] != 0 or
          counters["prefetch_cancel_count"] > counters["prefetch_reserve_count"]):
        raise ContractError("candidate prefetch counters do not prove a successful route")
    media = _mapping(result.get("media"), f"{variant}.media")
    audio = _mapping(media.get("audio"), f"{variant}.media.audio")
    video = _mapping(media.get("video"), f"{variant}.media.video")
    if (media.get("full_decode") is not True or audio.get("non_silent") is not True or
            video.get("codec") != "h264" or video.get("width") != 864 or
            video.get("height") != 480 or video.get("frames") != 22):
        raise ContractError(f"{variant} media evidence is incomplete")
    wall_seconds = result.get("wall_seconds")
    if (isinstance(wall_seconds, bool) or
            not isinstance(wall_seconds, (int, float)) or
            not math.isfinite(float(wall_seconds)) or wall_seconds <= 0):
        raise ContractError(f"{variant}.wall_seconds must be positive and finite")
    return {
        "binary_sha256": binary_sha256,
        "matched_contract_sha256": matched_contract_sha256,
        "profile_sha256": profile_sha256,
        "wait_seconds": float(wait), "wait_count": count,
        "prefetch_counters": counters,
        "media_sha256": _sha256(media.get("sha256"),
                                 f"{variant}.media.sha256"),
        "scheduler_sha256": _sha256(_mapping(
            result.get("scheduler_evidence"),
            f"{variant}.scheduler_evidence").get("trace_sha256"),
            f"{variant}.scheduler_evidence.trace_sha256"),
        "attention_sha256": _sha256(_mapping(
            result.get("attention_evidence"),
            f"{variant}.attention_evidence").get("trace_sha256"),
            f"{variant}.attention_evidence.trace_sha256"),
        "wall_seconds": float(wall_seconds),
    }


def compare_results(baseline: dict[str, Any], candidate: dict[str, Any],
                    baseline_result_sha256: str,
                    candidate_result_sha256: str) -> dict[str, Any]:
    baseline_result_sha256 = _sha256(
        baseline_result_sha256, "baseline result SHA-256")
    candidate_result_sha256 = _sha256(
        candidate_result_sha256, "candidate result SHA-256")
    base = _trial(baseline, "baseline")
    cand = _trial(candidate, "candidate")
    for key in SHA_KEYS:
        base_sha = _sha256(baseline.get(key), f"baseline.{key}")
        candidate_sha = _sha256(candidate.get(key), f"candidate.{key}")
        if base_sha != candidate_sha:
            raise ContractError(f"matched trial mismatch: {key}")
    for key in ("binary_sha256", "matched_contract_sha256"):
        if base[key] != cand[key]:
            raise ContractError(f"matched trial mismatch: {key}")
    parity = all(base[key] == cand[key] for key in (
        "media_sha256", "scheduler_sha256", "attention_sha256"))
    baseline_wait = base["wait_seconds"]
    if baseline_wait <= 0:
        raise ContractError("baseline exclusive upload-ready wait must be positive")
    if base["wait_count"] <= 0 or cand["wait_count"] != base["wait_count"]:
        raise ContractError("matched trials must have the same positive wait count")
    reduction = 1.0 - cand["wait_seconds"] / baseline_wait
    gate_met = parity and reduction >= 0.50
    return {
        "schema_version": SCHEMA_VERSION, "kind": KIND,
        "input_manifest_sha256": baseline["input_manifest_sha256"],
        "bindings_sha256": baseline["bindings_sha256"],
        "binary_sha256": base["binary_sha256"],
        "matched_contract_sha256": base["matched_contract_sha256"],
        "baseline": {
            "result_sha256": baseline_result_sha256,
            "profile_sha256": base["profile_sha256"],
            "exclusive_upload_ready_wait_seconds": baseline_wait,
            "upload_ready_wait_count": base["wait_count"],
            "prefetch_counters": base["prefetch_counters"],
            "wall_seconds": base["wall_seconds"],
        },
        "candidate": {
            "result_sha256": candidate_result_sha256,
            "profile_sha256": cand["profile_sha256"],
            "exclusive_upload_ready_wait_seconds": cand["wait_seconds"],
            "upload_ready_wait_count": cand["wait_count"],
            "prefetch_counters": cand["prefetch_counters"],
            "wall_seconds": cand["wall_seconds"],
        },
        "media_parity": parity,
        "media_sha256": base["media_sha256"] if parity else None,
        "scheduler_trace_sha256": (
            base["scheduler_sha256"] if parity else None),
        "attention_trace_sha256": (
            base["attention_sha256"] if parity else None),
        "exclusive_upload_ready_wait_reduction": reduction,
        "required_reduction": 0.50,
        "status": "OBSERVED_WAIT_PASS" if gate_met else "NOT_PASS",
        "remaining_gates": [
            "counterbalanced cold repetitions",
            "generated INT8 repeated eviction/reload",
            "pageable staging failure injection",
            "124-frame/8-step full A/B",
            "newer NVIDIA architecture",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-result", required=True, type=Path)
    parser.add_argument("--candidate-result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    baseline, baseline_hash = _load(args.baseline_result, "baseline result")
    candidate, candidate_hash = _load(args.candidate_result, "candidate result")
    report = compare_results(baseline, candidate, baseline_hash, candidate_hash)
    publish_json(args.output, report)
    print(f"PERF-006 matched A/B status: {report['status']}")
    return 0 if report["status"] == "OBSERVED_WAIT_PASS" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as error:
        print(f"PERF-006 A/B contract error: {error}", file=sys.stderr)
        raise SystemExit(3)
