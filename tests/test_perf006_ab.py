#!/usr/bin/env python3
"""Contract tests for the PERF-006 paired smoke validator."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_validator():
    path = ROOT / "scripts" / "validate_perf006_ab.py"
    spec = importlib.util.spec_from_file_location("validate_perf006_ab", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(variant: str, wait: float, media: str = "a" * 64) -> dict:
    candidate = variant == "candidate"
    return {
        "schema_version": 1, "kind": "h3cspeed.perf002.smoke",
        "engine": "h3cspeed", "status": "SMOKE_PASS",
        "input_manifest_sha256": "1" * 64, "bindings_sha256": "2" * 64,
        "contract": {"width": 864, "height": 480, "frames": 22,
                     "fps": 24, "steps": 2, "layers": 50, "seed": 42},
        "wall_seconds": 10.0,
        "scheduler_evidence": {"trace_sha256": "3" * 64},
        "attention_evidence": {"trace_sha256": "4" * 64},
        "media": {
            "sha256": media, "full_decode": True,
            "video": {"codec": "h264", "width": 864, "height": 480,
                      "frames": 22},
            "audio": {"non_silent": True},
        },
        "perf006_evidence": {
            "variant": variant,
            "dit_prefetch_mode": "one_ahead_convrot" if candidate else "disabled",
            "async_refill_requested": True, "async_refill_active": True,
            "upload_wait_trace_requested": True,
            "upload_wait_trace_complete": True,
            "upload_wait_trace_overflow": False, "union_valid": True,
            "ssd_streaming": False, "scope": "dit_denoise",
            "exclusive_upload_ready_wait_seconds": wait,
            "upload_ready_wait_count": 5,
            "binary_sha256": "5" * 64,
            "matched_contract_sha256": "6" * 64,
            "profile_sha256": "7" * 64,
            "prefetch_reserve_count": 11 if candidate else 0,
            "prefetch_upload_count": 11 if candidate else 0,
            "prefetch_consume_count": 11 if candidate else 0,
            "prefetch_cancel_count": 0,
            "prefetch_error_count": 0,
            "prefetch_block_count": 1 if candidate else 0,
        },
    }


class Perf006AbTests(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = load_validator()

    def compare(self, baseline: dict, candidate: dict) -> dict:
        return self.validator.compare_results(
            baseline, candidate, "8" * 64, "9" * 64)

    def test_half_or_more_wait_reduction_passes_with_exact_media_parity(self) -> None:
        report = self.compare(result("baseline", 4.0), result("candidate", 2.0))
        self.assertEqual(report["status"], "OBSERVED_WAIT_PASS")
        self.assertEqual(report["exclusive_upload_ready_wait_reduction"], 0.5)

    def test_incomplete_or_aggregate_evidence_is_rejected(self) -> None:
        for key in ("upload_wait_trace_complete", "union_valid"):
            with self.subTest(key=key):
                candidate = result("candidate", 1.0)
                candidate["perf006_evidence"][key] = False
                with self.assertRaisesRegex(self.validator.ContractError, key):
                    self.compare(result("baseline", 4.0), candidate)
        overflow = result("candidate", 1.0)
        overflow["perf006_evidence"]["upload_wait_trace_overflow"] = True
        with self.assertRaisesRegex(self.validator.ContractError, "overflowed"):
            self.compare(result("baseline", 4.0), overflow)

    def test_media_difference_or_insufficient_reduction_is_not_pass(self) -> None:
        media_mismatch = self.compare(
            result("baseline", 4.0), result("candidate", 1.0, "b" * 64))
        self.assertEqual(media_mismatch["status"], "NOT_PASS")
        slow = self.compare(result("baseline", 4.0), result("candidate", 2.1))
        self.assertEqual(slow["status"], "NOT_PASS")

    def test_binary_contract_and_positive_baseline_are_fail_closed(self) -> None:
        candidate = result("candidate", 1.0)
        candidate["perf006_evidence"]["binary_sha256"] = "f" * 64
        with self.assertRaisesRegex(self.validator.ContractError, "binary_sha256"):
            self.compare(result("baseline", 4.0), candidate)
        with self.assertRaisesRegex(self.validator.ContractError, "must be positive"):
            self.compare(result("baseline", 0.0), result("candidate", 0.0))

    def test_missing_hash_or_mismatched_wait_count_is_fail_closed(self) -> None:
        candidate = result("candidate", 1.0)
        candidate["perf006_evidence"]["upload_ready_wait_count"] = 3
        with self.assertRaisesRegex(self.validator.ContractError,
                                    "same positive wait count"):
            self.compare(result("baseline", 4.0), candidate)
        candidate = result("candidate", 1.0)
        candidate["scheduler_evidence"]["trace_sha256"] = None
        with self.assertRaisesRegex(self.validator.ContractError,
                                    "lowercase SHA-256"):
            self.compare(result("baseline", 4.0), candidate)

    def test_candidate_requires_real_prefetch_counters(self) -> None:
        for key in ("prefetch_reserve_count", "prefetch_upload_count",
                    "prefetch_consume_count", "prefetch_block_count"):
            with self.subTest(key=key):
                candidate = result("candidate", 1.0)
                candidate["perf006_evidence"][key] = 0
                with self.assertRaisesRegex(self.validator.ContractError,
                                            "do not prove"):
                    self.compare(result("baseline", 4.0), candidate)
        candidate = result("candidate", 1.0)
        del candidate["perf006_evidence"]["prefetch_upload_count"]
        with self.assertRaisesRegex(self.validator.ContractError,
                                    "nonnegative integer"):
            self.compare(result("baseline", 4.0), candidate)


if __name__ == "__main__":
    unittest.main()
