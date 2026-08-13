from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "scripts" / "validate_profile_report.py"
SPEC = importlib.util.spec_from_file_location("validate_profile_report", VALIDATOR_PATH)
assert SPEC and SPEC.loader
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)
PROFILE_WRITER = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None


class ProfileReportTests(unittest.TestCase):
    def test_c_writer_emits_valid_json(self) -> None:
        if PROFILE_WRITER is None:
            self.skipTest("C profile writer executable was not provided")
        with tempfile.TemporaryDirectory(prefix="h3-profile-test-") as directory:
            result = subprocess.run(
                [str(PROFILE_WRITER), directory],
                check=True,
                text=True,
                capture_output=True,
            )
            report_path = Path(result.stdout.strip())
            self.assertNotIn("unit", report_path.name)
            self.assertIn("redacted", report_path.name)
            report = VALIDATOR.validate_path(report_path)
            self.assertEqual(report["counts"]["capacity_lru"], 1)
            self.assertEqual(report["counts"]["phase_retire"], 1)
            self.assertEqual(report["counts"]["error_cleanup"], 1)
            self.assertEqual(report["bytes"]["capacity_lru"], 100)
            self.assertNotIn("E:\\", json.dumps(report))
            self.assertEqual(report["context"]["label"], "redacted")
            self.assertFalse(report["wall"]["coverage_gate_valid"])

    def test_nan_and_negative_values_fail_closed(self) -> None:
        base = self._minimal_report()
        base["timing"]["file_read_seconds"] = math.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            VALIDATOR.validate_report(base)
        base = self._minimal_report()
        base["counts"]["capacity_lru"] = -1
        with self.assertRaisesRegex(ValueError, "non-negative"):
            VALIDATOR.validate_report(base)

    def test_missing_overlap_disclosure_fails_closed(self) -> None:
        report = self._minimal_report()
        report["non_additive"] = []
        with self.assertRaisesRegex(ValueError, "overlap"):
            VALIDATOR.validate_report(report)

    def test_inconsistent_accounting_fails_closed(self) -> None:
        report = self._minimal_report()
        report["wall"]["seconds"] = 4.0
        report["wall"]["accounted_host_seconds"] = 1.0
        report["wall"]["accounted_ratio"] = 1.0
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            VALIDATOR.validate_report(report)

    @staticmethod
    def _minimal_report() -> dict[str, object]:
        timing = {
            key: 0.0
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
            )
        }
        counts = {
            key: 0
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
                "capacity_lru",
                "phase_retire",
                "error_cleanup",
            )
        }
        return {
            "schema_version": 1,
            "kind": "h3cspeed.cuda.profile",
            "context": {
                "id": 1,
                "pid": 1,
                "label": "test",
                "device": 0,
                "sm": "86",
                "complete": True,
            },
            "wall": {
                "seconds": 0.0,
                "accounted_host_seconds": 0.0,
                "accounted_ratio": 0.0,
                "coverage_gate_valid": False,
                "coverage_gate_met": False,
            },
            "timing": timing,
            "counts": counts,
            "bytes": {
                "file_read": 0,
                "pageable_copy": 0,
                "h2d": 0,
                "capacity_lru": 0,
                "phase_retire": 0,
                "error_cleanup": 0,
            },
            "memory": {"device_peak_bytes": 0},
            "offload": {"uploads": 0},
            "dispatches": {"direct": 0},
            "validity": {
                "h2d_device_seconds": False,
                "compute_device_seconds": False,
                "critical_path_scope": "gpu_context_host_operations",
            },
            "non_additive": ["device_and_host_timings_may_overlap"],
        }


if __name__ == "__main__":
    sys.argv = [sys.argv[0]]
    unittest.main()
