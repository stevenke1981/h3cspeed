from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import h3_model_info as model_info  # noqa: E402


DTYPE_BYTES = model_info.DTYPE_BYTES


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    document: dict[str, object] = {"__metadata__": {"format": "test"}}
    cursor = 0
    for name, (dtype, shape) in tensors.items():
        size = 0
        if dtype in DTYPE_BYTES:
            elements = 1
            for dimension in shape:
                elements *= dimension
            size = elements * DTYPE_BYTES[dtype]
        document[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size
    header = json.dumps(document, separators=(",", ":")).encode("utf-8")
    header += b" " * ((-len(header)) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(len(header).to_bytes(8, "little"))
        stream.write(header)
        if cursor:
            stream.seek(cursor - 1, 1)
            stream.write(b"\0")


def qualified(prefix: str, suffix: str) -> str:
    return f"{prefix}.{suffix}" if prefix else suffix


def standard_tensors(
    *,
    prefix: str = "",
    hidden: int = 5376,
    layers: int = 50,
    skip_layer: int | None = None,
    refiner_layers: int = 2,
    adaln: bool = False,
) -> dict[str, tuple[str, list[int]]]:
    tensors: dict[str, tuple[str, list[int]]] = {
        qualified(prefix, "video_patch_proj.weight"): ("TEST", [hidden, 24 * 4]),
        qualified(prefix, "audio_patch_proj.weight"): ("TEST", [hidden, 32]),
        qualified(prefix, "blocks.0.attn.q_norm.weight"): ("TEST", [128]),
        qualified(prefix, "blocks.0.attn.qkv_proj.weight"): (
            "TEST",
            [3 * 56 * 128, hidden],
        ),
        qualified(prefix, "blocks.0.mlp.fc1.weight"): (
            "TEST",
            [2 * 14336, hidden],
        ),
        qualified(prefix, "condition_proj.weight"): ("TEST", [hidden, 5120]),
        qualified(prefix, "rope.inv_freq"): ("TEST", [16]),
    }
    if adaln:
        tensors[qualified(prefix, "adaln_t_table")] = ("TEST", [1025, 8])
    else:
        tensors[qualified(prefix, "time_embedder.proj_in.weight")] = (
            "TEST",
            [hidden, 256],
        )
        tensors[qualified(prefix, "time_embedder.proj_out.weight")] = (
            "TEST",
            [2688, hidden],
        )
    for block in range(1, layers):
        if block == skip_layer:
            continue
        tensors[qualified(prefix, f"blocks.{block}.norm1.weight")] = (
            "TEST",
            [hidden],
        )
    for block in range(refiner_layers):
        tensors[qualified(prefix, f"token_refiner.blocks.{block}.norm1.weight")] = (
            "TEST",
            [hidden],
        )
    return tensors


class ModelInfoTests(unittest.TestCase):
    def test_standard_multishard_checkpoint_is_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            component = Path(temporary) / "transformer"
            tensors = list(standard_tensors().items())
            write_safetensors(component / "model-00001.safetensors", dict(tensors[::2]))
            write_safetensors(component / "model-00002.safetensors", dict(tensors[1::2]))

            results = model_info.inspect_path(component)
            self.assertEqual(len(results), 1)
            result = results[0]
            self.assertTrue(result.compatible)
            self.assertEqual(result.shards, 2)
            self.assertEqual(result.config.num_layers, 50)
            self.assertEqual(result.config.num_attention_heads, 56)
            self.assertEqual(result.config.variant, "time-embedder")

    def test_prefixed_checkpoint_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            component = Path(temporary) / "transformer"
            prefix = "model.diffusion_model"
            write_safetensors(
                component / "model.safetensors",
                standard_tensors(prefix=prefix),
            )
            output = io.StringIO()
            with redirect_stdout(output):
                code = model_info.main([str(component), "--json", "--strict-h3cspeed"])
            self.assertEqual(code, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["components"][0]["prefix"], prefix)
            self.assertTrue(payload["components"][0]["h3cspeed"]["compatible"])

    def test_adaln_variant_is_reported_and_strict_mode_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            component = Path(temporary) / "transformer"
            write_safetensors(
                component / "model.safetensors",
                standard_tensors(adaln=True),
            )
            output = io.StringIO()
            with redirect_stdout(output):
                code = model_info.main([str(component), "--json", "--strict-h3cspeed"])
            self.assertEqual(code, 0)
            payload = json.loads(output.getvalue())
            detected = payload["components"][0]
            self.assertEqual(detected["config"]["variant"], "adaln-curves")
            self.assertTrue(detected["h3cspeed"]["compatible"])
            self.assertEqual(detected["h3cspeed"]["issues"], [])

    def test_invalid_adaln_curve_table_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            component = Path(temporary) / "transformer"
            tensors = standard_tensors(adaln=True)
            tensors["adaln_t_table"] = ("TEST", [1025, 7])
            write_safetensors(component / "model.safetensors", tensors)
            with self.assertRaisesRegex(model_info.InspectionError, "curve table"):
                model_info.inspect_path(component)

    def test_sparse_blocks_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            component = Path(temporary) / "transformer"
            write_safetensors(
                component / "model.safetensors",
                standard_tensors(layers=4, skip_layer=2),
            )
            with self.assertRaisesRegex(model_info.InspectionError, "not contiguous"):
                model_info.inspect_path(component)

    def test_duplicate_tensor_across_shards_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            component = Path(temporary) / "transformer"
            tensors = standard_tensors()
            write_safetensors(component / "model-1.safetensors", tensors)
            duplicate = {
                "video_patch_proj.weight": tensors["video_patch_proj.weight"]
            }
            write_safetensors(component / "model-2.safetensors", duplicate)
            with self.assertRaisesRegex(model_info.InspectionError, "duplicate tensor"):
                model_info.inspect_path(component)

    def test_model_root_discovers_multiple_h3_components(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "model"
            write_safetensors(
                root / "fl2va_transformer" / "model.safetensors",
                standard_tensors(),
            )
            write_safetensors(
                root / "ref2va_transformer" / "model.safetensors",
                standard_tensors(),
            )
            write_safetensors(
                root / "audio_vae" / "model.safetensors",
                {"decoder.weight": ("TEST", [8, 8])},
            )
            results = model_info.inspect_path(root)
            self.assertEqual(len(results), 2)
            self.assertTrue(all(result.compatible for result in results))

    def test_geometry_alignment_matches_h3_temporal_contract(self) -> None:
        report = model_info.geometry_report(641, 480, 6)
        self.assertEqual(report["width"]["aligned"], 672)
        self.assertEqual(report["height"]["aligned"], 480)
        self.assertEqual(report["frames"]["aligned"], 22)
        self.assertEqual(model_info.align_frame_count(5), 5)
        self.assertEqual(model_info.align_frame_count(22), 22)
        self.assertEqual(model_info.align_frame_count(56), 56)

    def test_known_dtype_byte_range_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tiny.safetensors"
            write_safetensors(path, {"tiny": ("BF16", [2, 3])})
            tensors = model_info.read_safetensors_header(path)
            self.assertEqual(tensors["tiny"].shape, (2, 3))

            raw = path.read_bytes()
            header_size = int.from_bytes(raw[:8], "little")
            document = json.loads(raw[8 : 8 + header_size].decode("utf-8"))
            document["tiny"]["data_offsets"] = [0, 10]
            header = json.dumps(document, separators=(",", ":")).encode("utf-8")
            header += b" " * ((-len(header)) % 8)
            path.write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * 10)
            with self.assertRaisesRegex(model_info.InspectionError, "byte range"):
                model_info.read_safetensors_header(path)

    def test_cli_error_is_structured_in_json_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = io.StringIO()
            errors = io.StringIO()
            with redirect_stdout(output), redirect_stderr(errors):
                code = model_info.main([temporary, "--json"])
            self.assertEqual(code, 2)
            payload = json.loads(output.getvalue())
            self.assertIn("error", payload)
            self.assertEqual(errors.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
