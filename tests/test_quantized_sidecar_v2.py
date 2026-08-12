"""CPU-only contract tests for the quantized FL2VA sidecar v2 bridge."""

from __future__ import annotations

import hashlib
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import encode_h3_quantized_prompt as sidecar


class _Values:
    ndim = 2
    shape = (2, sidecar.WIDTH)

    def tobytes(self, order: str = "C") -> bytes:
        self_order = order
        if self_order != "C":
            raise AssertionError(self_order)
        return b"\0" * (2 * sidecar.WIDTH * 2)


class _Image:
    shape = (1, 480, 864, 3)


class QuantizedSidecarV2Tests(unittest.TestCase):
    def test_canonical_keyframe_uses_isolated_random_temporary_directory(self) -> None:
        source = Path(sidecar.__file__).read_text(encoding="utf-8")
        self.assertIn('TemporaryDirectory(prefix="h3cspeed-keyframe-")', source)
        self.assertIn("Image.open(io.BytesIO(data))", source)
        self.assertIn("canonical_bytes = [path.read_bytes()", source)
        self.assertIn("destination.is_symlink()", source)
        self.assertNotIn('destination.name + ".tmp.png"', source)

    def test_image_token_expansion_matches_minimax_geometry(self) -> None:
        entries = [(151652, 1.0), ({"type": "image", "data": _Image()}, 1.0),
                   (151653, 1.0), (42, 1.0)]
        ids = sidecar._expanded_token_ids(entries)
        self.assertEqual(ids[0], sidecar.VISION_START)
        self.assertEqual(ids[-1], 42)
        grid_h, grid_w = sidecar._image_grid(480, 864)
        self.assertEqual(ids.count(sidecar.VISION_PAD), (grid_h * grid_w) // 4)
        self.assertEqual(ids.count(sidecar.VISION_START), 1)
        self.assertEqual(ids.count(sidecar.VISION_END), 1)

    def test_v2_header_binds_render_role_and_image_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conditioning.h3c"
            image_hash = hashlib.sha256(b"first-image").digest()
            model_hash = hashlib.sha256(b"model").digest()
            sidecar._write_sidecar(
                path, "fox", [1, 2], _Values(), model_hash, [1, 1],
                mode=sidecar.MODE_FL2VA_I2V,
                keyframe_role=sidecar.ROLE_FIRST,
                render_width=864,
                render_height=480,
                image_hashes=[image_hash],
            )
            data = path.read_bytes()
            self.assertEqual(data[:8], sidecar.MAGIC)
            self.assertEqual(struct.unpack_from("<I", data, 8)[0], 2)
            self.assertEqual(data[104:110], bytes([1, 1, 1, 1, 0, 1]))
            self.assertEqual(struct.unpack_from("<II", data, 112), (864, 480))
            self.assertEqual(struct.unpack_from("<I", data, 120)[0], 32)
            recipe_size = struct.unpack_from("<Q", data, 32)[0]
            prompt_size = struct.unpack_from("<Q", data, 16)[0]
            self.assertEqual(
                data[128 + prompt_size + recipe_size:128 + prompt_size + recipe_size + 32],
                image_hash,
            )

    def test_i2v_requires_keyframe_and_render_geometry(self) -> None:
        with self.assertRaises(sidecar.SidecarError):
            sidecar._write_sidecar(
                Path("unused.h3c"), "fox", [1, 2], _Values(), b"x" * 32, [1, 1],
                mode=sidecar.MODE_FL2VA_I2V, keyframe_role=sidecar.ROLE_FIRST,
                render_width=0, render_height=0, image_hashes=[b"x" * 32],
            )

    def test_i2v_geometry_requires_qwen_rounding_grid(self) -> None:
        with self.assertRaises(sidecar.SidecarError):
            sidecar._write_sidecar(
                Path("unused.h3c"), "fox", [1, 2], _Values(), b"x" * 32, [1, 1],
                mode=sidecar.MODE_FL2VA_I2V, keyframe_role=sidecar.ROLE_FIRST,
                render_width=850, render_height=480, image_hashes=[b"x" * 32],
            )

    def test_all_sidecar_modes_require_h3c_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(sidecar.SidecarError, "must end in .h3c"):
                sidecar._write_sidecar(
                    Path(directory) / "private_conditioning", "fox", [1, 2],
                    _Values(), b"x" * 32, [1, 1],
                    mode=sidecar.MODE_FL2VA_I2V,
                    keyframe_role=sidecar.ROLE_FIRST,
                    render_width=864, render_height=480,
                    image_hashes=[b"x" * 32])
            with self.assertRaisesRegex(sidecar.SidecarError, "must end in .h3c"):
                sidecar._write_sidecar(
                    Path(directory) / "private_conditioning.txt", "fox", [1, 2],
                    _Values(), b"x" * 32, [1, 1],
                    mode=sidecar.MODE_T2V,
                    keyframe_role=0,
                    render_width=0, render_height=0,
                    image_hashes=[])


if __name__ == "__main__":
    unittest.main()
