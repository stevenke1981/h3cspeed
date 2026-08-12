"""Portable tests for the header-only H3 model pack preparer."""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import prepare_h3_quantized_model as prep


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    for name, (dtype, shape, value) in tensors.items():
        start = len(payload)
        payload.extend(value)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(payload)]}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class PrepareH3QuantizedModelTests(unittest.TestCase):
    def test_fl2va_validator_requires_curve_projection_weights(self) -> None:
        required = {
            "adaln_t_table": prep.TensorSpec("adaln_t_table", "F32", (1025, 8), (0, 32800)),
            "audio_patch_proj.weight": prep.TensorSpec("audio_patch_proj.weight", "F32", (5376, 32), (32800, 720928)),
            "video_patch_proj.weight": prep.TensorSpec("video_patch_proj.weight", "F32", (5376, 96), (720928, 2785312)),
            "condition_proj.weight": prep.TensorSpec("condition_proj.weight", "BF16", (5376, 5120), (2785312, 57835552)),
            "final_layer.audio_out.weight": prep.TensorSpec("final_layer.audio_out.weight", "F32", (32, 5376), (57835552, 58523680)),
            "final_layer.video_out.weight": prep.TensorSpec("final_layer.video_out.weight", "F32", (96, 5376), (58523680, 60588160)),
            "rope.inv_freq": prep.TensorSpec("rope.inv_freq", "F32", (16,), (60588160, 60588224)),
            "blocks.0.attn.q_norm.weight": prep.TensorSpec("blocks.0.attn.q_norm.weight", "BF16", (128,), (60588224, 60588480)),
            "blocks.0.attn.k_norm.weight": prep.TensorSpec("blocks.0.attn.k_norm.weight", "BF16", (128,), (60588480, 60588736)),
        }
        report = prep.HeaderReport(
            path=Path("missing-curve.safetensors"), file_size=1, header_size=1,
            data_start=1, tensors=required, metadata={}, header_sha256="test",
            dtype_counts={},
        )
        with self.assertRaisesRegex(
            prep.PreparationError,
            r"missing required tensor blocks\.0\.adaln_proj\.linear\.weight",
        ):
            prep._validate_fl2va(report)

    def test_header_reader_checks_offsets_and_dtype_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "valid.safetensors"
            _write_safetensors(path, {"x": ("F32", [2], b"\0" * 8)})
            report = prep.read_safetensors_header(path)
            self.assertEqual(report.tensors["x"].byte_size, 8)
            self.assertEqual(report.payload_size, 8)

            malformed = Path(directory) / "malformed.safetensors"
            _write_safetensors(malformed, {"x": ("F32", [2], b"\0" * 4)})
            with self.assertRaises(prep.PreparationError):
                prep.read_safetensors_header(malformed)

    def test_marker_is_read_only_when_small(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "marker.safetensors"
            marker = b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}'
            _write_safetensors(path, {"layer.comfy_quant": ("U8", [len(marker)], marker)})
            report = prep.read_safetensors_header(path)
            self.assertEqual(prep._marker_json(report, "layer.comfy_quant")["convrot_groupsize"], 256)

    def test_prepare_uses_hardlinks_copies_allowlisted_files_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "ComfyUI" / "models"
            base = root / "MiniMax-H3"
            config = base / "FL2VA" / "transformer" / "config.json"
            config.parent.mkdir(parents=True)
            config.write_text("{}\n", encoding="utf-8")

            reports: dict[str, prep.HeaderReport] = {}
            for role, (folder, filename) in prep.PACK_FILES.items():
                source = models / folder / filename
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(role.encode("ascii"))
                reports[role] = prep.HeaderReport(
                    path=source,
                    file_size=source.stat().st_size,
                    header_size=4,
                    data_start=12,
                    tensors={},
                    metadata={},
                    header_sha256="test",
                    dtype_counts={},
                )
            pack = prep.PackReport(
                models_root=models,
                base_root=base,
                files=reports,
                small_files=[config],
                details={role: {} for role in prep.PACK_FILES},
            )
            output = root / "prepared"
            with mock.patch.object(prep, "validate_pack", return_value=pack):
                self.assertEqual(prep.prepare_pack(output, models, base), output.resolve())
                self.assertTrue((output / "manifest.json").is_file())
                self.assertTrue(os.path.samefile(
                    models / "diffusion_models" / prep.PACK_FILES["fl2va"][1],
                    output / "base" / prep.PACK_COMPONENT_PATHS["fl2va"] / prep.PACK_FILES["fl2va"][1],
                ))
                self.assertEqual(
                    (output / "base" / "FL2VA" / "transformer" / "config.json").read_text(
                        encoding="utf-8"
                    ),
                    "{}\n",
                )
                manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
                self.assertFalse(manifest["large_payloads_copied"])
                self.assertEqual(Path(manifest["model_root"]), output / "base")
                self.assertEqual(manifest["model_root_relative"], "base")
                self.assertTrue((output / manifest["model_root_relative"] / "FL2VA").is_dir())
                self.assertEqual({entry["mode"] for entry in manifest["large_payloads"]}, {"hardlink"})
                self.assertEqual({entry["mode"] for entry in manifest["small_configs"]}, {"copy"})
                with self.assertRaises(prep.PreparationError):
                    prep.prepare_pack(output, models, base)


if __name__ == "__main__":
    unittest.main()
