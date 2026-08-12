#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "h3cspeed_runtime_package", ROOT / "scripts/package_runtime.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RuntimePackageTest(unittest.TestCase):
    def test_common_payload_includes_quantized_60_second_runner(self) -> None:
        package = load_module()
        self.assertIn("scripts/run_h3_quantized_60s.py", package.COMMON_SCRIPTS)

    def test_forbidden_model_and_conditioning_payloads_fail_closed(self) -> None:
        package = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "runtime"
            bundle.mkdir()
            for name in ("prompt.h3c", "embedding.bf16", "model.safetensors",
                         "model.onnx", "tensor.npy", "tensor.npz"):
                path = bundle / name
                path.write_bytes(b"private")
                with self.subTest(name=name):
                    with self.assertRaisesRegex(RuntimeError, "forbidden"):
                        package.check_forbidden(bundle)
                path.unlink()

    def test_linux_launcher_uses_private_runtime_and_preserves_arguments(self) -> None:
        package = load_module()
        launcher = package.linux_launcher("h3cspeed")
        self.assertIn('LD_LIBRARY_PATH="$ROOT/lib', launcher)
        self.assertIn('exec "$ROOT/libexec/h3cspeed" "$@"', launcher)

    def test_copy_file_rejects_symlink_inputs(self) -> None:
        package = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.txt"
            source.write_text("public", encoding="utf-8")
            link = root / "link.txt"
            try:
                link.symlink_to(source)
            except OSError:
                self.skipTest("symlinks are unavailable on this host")
            with self.assertRaisesRegex(RuntimeError, "required runtime input"):
                package.copy_file(link, root / "copied.txt")

    def test_windows_zip_is_unix_mode_aware_and_rooted(self) -> None:
        package = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "h3cspeed-test"
            script = bundle / "scripts/probe.py"
            script.parent.mkdir(parents=True)
            script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            script.chmod(0o755)
            archive = Path(temporary) / "runtime.zip"
            package.add_zip_tree(bundle, archive)
            with zipfile.ZipFile(archive) as source:
                info = source.getinfo("h3cspeed-test/scripts/probe.py")
            self.assertEqual(info.create_system, 3)
            self.assertTrue((info.external_attr >> 16) & 0o111)

    def test_runtime_readmes_state_external_media_and_driver_requirements(self) -> None:
        package = load_module()
        windows = package.runtime_readme("windows-x86_64")
        linux = package.runtime_readme("linux-x86_64")
        for text in (windows, linux):
            self.assertIn("NVIDIA driver", text)
            self.assertIn("`sm_86`", text)
            self.assertNotIn("sm_80-or-newer", text)
            self.assertIn("ffmpeg", text)
            self.assertIn("Model weights", text)
        self.assertIn("Visual C++", windows)
        self.assertIn("Ubuntu 22.04", linux)

    def test_common_payload_includes_complete_linked_library_licenses(self) -> None:
        package = load_module()
        self.assertIn("licenses/yyjson-LICENSE", package.COMMON_LICENSES)
        yyjson = (ROOT / "licenses/yyjson-LICENSE").read_text(encoding="utf-8")
        self.assertIn("Copyright (c) 2020 YaoYuan", yyjson)
        self.assertIn("Permission is hereby granted", yyjson)

    def test_cuda_license_is_pinned_and_lists_bundled_components(self) -> None:
        package = load_module()
        license_path = package.cuda_license(ROOT)
        text = license_path.read_text(encoding="utf-8")
        self.assertIn("Attachment A", text)
        self.assertIn("libcudart.so", text)
        self.assertIn("libcublas.so", text)
        self.assertIn("libcublasLt.so", text)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "licenses/NVIDIA-CUDA-13.2-EULA.txt"
            target.parent.mkdir()
            target.write_text("wrong license", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "missing or modified"):
                package.cuda_license(root)

    def test_dirty_git_provenance_fails_closed_by_default(self) -> None:
        package = load_module()
        with mock.patch.object(
                package.subprocess, "check_output",
                side_effect=["deadbeef\n", " M README.md\n"]):
            with self.assertRaisesRegex(RuntimeError, "clean Git checkout"):
                package.git_provenance(ROOT, allow_dirty=False)
        with mock.patch.object(
                package.subprocess, "check_output",
                side_effect=["deadbeef\n", " M README.md\n"]):
            self.assertEqual(
                package.git_provenance(ROOT, allow_dirty=True),
                ("deadbeef", True),
            )


if __name__ == "__main__":
    unittest.main()
