#!/usr/bin/env python3
"""Portable contract tests for the resolution-matrix walking skeleton."""

from __future__ import annotations

import copy
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zlib


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_resolution_matrix.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_resolution_matrix", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_png(path: Path, width: int, height: int) -> None:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + kind + payload +
                struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))

    rows = b"".join(b"\0" + b"\0\0\0" * width for _ in range(height))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" +
        chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
        chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


def fixture_config(root: Path) -> tuple[Path, dict[str, object]]:
    inputs = root / "inputs"
    inputs.mkdir()
    files: dict[str, str] = {}
    names = {
        "comfy_main": "main.py",
        "t8_sampling": "sampling.py",
        "t8_nodes": "nodes.py",
        "comfy_attention": "attention.py",
        "model_file": "model.safetensors",
        "clip_file": "clip.safetensors",
        "video_vae_file": "video.safetensors",
        "audio_vae_file": "audio.safetensors",
        "prompt_file": "prompt.txt",
        "h3_text_encoder": "h3-text-encoder.safetensors",
        "h3_binary": "h3cspeed.exe",
    }
    for field, name in names.items():
        path = inputs / name
        path.write_bytes(b"private fixture\n")
        files[field] = str(path.resolve())
    profiles: dict[str, object] = {}
    for index, (profile, dimensions) in enumerate({
        "240": (448, 256), "480": (864, 480), "720": (1280, 704),
    }.items()):
        reference = inputs / f"reference-{profile}.png"
        write_png(reference, *dimensions)
        profiles[profile] = {
            "reference_png": str(reference.resolve()),
            "timeout_seconds": 3600 + index * 600,
        }
    value: dict[str, object] = {
        "schema_version": 1,
        "kind": "h3cspeed.resolution-matrix.config",
        "python_executable": str(Path(sys.executable).resolve()),
        "powershell_executable": str(Path(shutil.which("pwsh") or
                                          sys.executable).resolve()),
        "profiles": profiles,
        "h3_model_root": str(inputs.resolve()),
        "h3_comfy_root": str(inputs.resolve()),
        **files,
    }
    config = root / "private-config.json"
    config.write_text(json.dumps(value), encoding="utf-8")
    return config, value


class ResolutionMatrixTests(unittest.TestCase):
    def test_fixed_profiles_are_native_32_grid_without_stretch(self) -> None:
        runner = load_runner()
        expected = {
            "240": ({"width": 448, "height": 256},
                    {"width": 28, "height": 16}, 112),
            "480": ({"width": 864, "height": 480},
                    {"width": 54, "height": 30}, 405),
            "720": ({"width": 1280, "height": 704},
                    {"width": 80, "height": 44}, 880),
        }
        for profile, (dimensions, latent, patches) in expected.items():
            contract = runner.profile_contract(profile)
            self.assertEqual(contract["render"], dimensions)
            self.assertEqual(contract["output"], dimensions)
            self.assertEqual(contract["grid"], 32)
            self.assertEqual(contract["frames"], 124)
            self.assertEqual(contract["fps"], 24)
            self.assertEqual(contract["steps"], 2)
            self.assertEqual(contract["layers"], 50)
            self.assertEqual(contract["seed"], 42)
            self.assertEqual(contract["latent"], latent)
            self.assertEqual(contract["patch_tokens"], patches)
            self.assertEqual(contract["scaling"], "none")
            self.assertEqual(dimensions["width"] % 32, 0)
            self.assertEqual(dimensions["height"] % 32, 0)
            self.assertLessEqual(dimensions["width"] * dimensions["height"],
                                 768 * 1344)

    def test_profile_validation_rejects_render_mismatch_and_stretch(self) -> None:
        runner = load_runner()
        contract = runner.profile_contract("480")
        mismatched = copy.deepcopy(contract)
        mismatched["render"]["width"] = 448
        with self.assertRaisesRegex(runner.ContractError, "render dimensions"):
            runner.validate_profile_contract(mismatched)
        stretched = copy.deepcopy(contract)
        stretched["scaling"] = "stretch"
        with self.assertRaisesRegex(runner.ContractError, "stretching"):
            runner.validate_profile_contract(stretched)

    def test_malformed_plan_fails_with_contract_error(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, _ = fixture_config(root)
            config, digest = runner.load_config(config_path)
            output = root / "matrix"
            plan = runner.build_plan(config, digest, output, ("240",))
            malformed = copy.deepcopy(plan)
            del malformed["contracts"][0]["profile"]
            with self.assertRaises(runner.ContractError):
                runner.validate_plan(malformed, output)

    def test_schema_is_exact_and_paths_are_bound_regular_files(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, value = fixture_config(root)
            validated = runner.validate_config(value)
            self.assertEqual(validated["profiles"]["240"]["timeout_seconds"], 3600)
            extra = dict(value)
            extra["execute"] = True
            with self.assertRaisesRegex(runner.ContractError, "unexpected or missing"):
                runner.validate_config(extra)
            missing = dict(value)
            del missing["prompt_file"]
            with self.assertRaisesRegex(runner.ContractError, "unexpected or missing"):
                runner.validate_config(missing)
            relative = dict(value)
            relative["prompt_file"] = "prompt.txt"
            with self.assertRaisesRegex(runner.ContractError, "absolute regular"):
                runner.validate_config(relative)

    def test_each_profile_reference_must_have_exact_target_dimensions(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, value = fixture_config(root)
            wrong = root / "wrong.png"
            write_png(wrong, 864, 480)
            value["profiles"]["240"]["reference_png"] = str(wrong.resolve())
            with self.assertRaisesRegex(runner.ContractError, "exactly 448x256"):
                runner.validate_config(value)

    def test_commands_bind_each_contract_and_have_no_scaling_flags(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, _ = fixture_config(root)
            config, digest = runner.load_config(config_path)
            output = root / "matrix"
            plan = runner.build_plan(config, digest, output, ("240", "480", "720"))
            runner.validate_plan(plan, output)
            self.assertEqual([(item["profile"], item["engine"])
                              for item in plan["commands"]], [
                ("240", "h3cspeed"), ("240", "comfyui"),
                ("480", "h3cspeed"), ("480", "comfyui"),
                ("720", "h3cspeed"), ("720", "comfyui"),
            ])
            contracts = {item["profile"]: item for item in plan["contracts"]}
            expected_timeout = {"240": "3600", "480": "4200", "720": "4800"}
            for item in plan["commands"]:
                contract = contracts[item["profile"]]
                argv = item["argv"]
                if item["engine"] == "comfyui":
                    self.assertEqual(argv.count("--resolution-matrix"), 1)
                    width_flag, height_flag = "--width", "--height"
                    frames_flag, steps_flag = "--frames", "--steps"
                    reference_flag, prompt_flag = "--reference-png", "--prompt-file"
                    self.assertEqual(argv[argv.index("--timeout") + 1],
                                     expected_timeout[contract["profile"]])
                    self.assertNotIn("--render-width", argv)
                    self.assertNotIn("--render-height", argv)
                else:
                    self.assertEqual(argv.count("-ResolutionMatrix"), 1)
                    width_flag, height_flag = "-Width", "-Height"
                    frames_flag, steps_flag = "-Frames", "-Steps"
                    reference_flag, prompt_flag = "-FirstFrame", "-PromptFile"
                    self.assertNotIn("-RenderWidth", argv)
                    self.assertNotIn("-RenderHeight", argv)
                    self.assertEqual(argv[argv.index("-Layers") + 1], "50")
                    self.assertEqual(argv[argv.index("-Seed") + 1], "42")
                    for optimization in ("-LayerMajor", "-AsyncRefill", "-DitPrefetch"):
                        self.assertEqual(argv.count(optimization), 1)
                self.assertEqual(argv[argv.index(width_flag) + 1],
                                 str(contract["output"]["width"]))
                self.assertEqual(argv[argv.index(height_flag) + 1],
                                 str(contract["output"]["height"]))
                self.assertEqual(argv[argv.index(frames_flag) + 1], "124")
                self.assertEqual(argv[argv.index(steps_flag) + 1], "2")
                reference = Path(argv[argv.index(reference_flag) + 1])
                self.assertEqual(runner.png_dimensions(reference),
                                 (contract["output"]["width"],
                                  contract["output"]["height"]))
                self.assertEqual(argv[argv.index(prompt_flag) + 1],
                                 config["prompt_file"])
                self.assertFalse({"-RenderWidth", "-RenderHeight", "--render-width",
                                  "--render-height", "--resize", "--scale", "--stretch"}
                                 & set(argv))
            self.assertEqual(plan["status"], "NOT_RUN")
            self.assertEqual(plan["mode"], "dry-run")

    def test_h3_render_override_is_rejected_even_when_value_matches(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, _ = fixture_config(root)
            config, digest = runner.load_config(config_path)
            output = root / "matrix"
            plan = runner.build_plan(config, digest, output, ("240",))
            command = next(item for item in plan["commands"]
                           if item["engine"] == "h3cspeed")
            insert_at = command["argv"].index("-Frames")
            command["argv"][insert_at:insert_at] = [
                "-RenderWidth", "448", "-RenderHeight", "256"
            ]
            with self.assertRaisesRegex(runner.ContractError, "scaling or stretching"):
                runner.validate_plan(plan, output)

    def test_private_output_is_new_no_clobber_and_plan_stays_not_run(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, _ = fixture_config(root)
            output = root / "private-matrix"
            plan_path, digest = runner.create_dry_plan(
                config_path, output, ("240", "480", "720"))
            self.assertEqual(len(digest), 64)
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(plan["status"], "NOT_RUN")
            self.assertEqual([item["profile"] for item in plan["contracts"]],
                             ["240", "480", "720"])
            self.assertFalse(any((output / f"{profile}p" / "comfyui" / "runtime").exists()
                                 for profile in ("240", "480", "720")))
            if os.name != "nt":
                self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
                self.assertEqual(stat.S_IMODE((output / "240p").stat().st_mode), 0o700)
                self.assertEqual(
                    stat.S_IMODE((output / "240p" / "h3cspeed").stat().st_mode), 0o700)
            with self.assertRaisesRegex(runner.ContractError, "no-clobber"):
                runner.create_dry_plan(config_path, output, ("240",))

    def test_output_inside_source_tree_is_rejected_before_creation(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            config_path, _ = fixture_config(Path(temporary))
            output = ROOT / ".resolution-matrix-test-output"
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(runner.ContractError, "outside the source tree"):
                runner.create_dry_plan(config_path, output, ("480",))
            self.assertFalse(output.exists())

    def test_output_inside_comfy_or_model_root_is_rejected(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, value = fixture_config(root)
            protected = Path(value["comfy_main"]).parent / "matrix-output"
            with self.assertRaisesRegex(runner.ContractError, "ComfyUI and model roots"):
                runner.create_dry_plan(config_path, protected, ("240",))
            self.assertFalse(protected.exists())
            output = ROOT / ".resolution-matrix-test-output"
            with self.assertRaisesRegex(runner.ContractError, "outside the source tree"):
                runner.create_dry_plan(config_path, output, ("480",))
            self.assertFalse(output.exists())

    def test_cli_dry_run_does_not_create_runtime(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, _ = fixture_config(root)
            output = root / "cli-output"
            result = runner.main([
                "--config", str(config_path), "--output-dir", str(output),
                "--profiles", "240",
            ])
            self.assertEqual(result, 0)
            self.assertTrue((output / "resolution-matrix-plan.json").is_file())
            self.assertFalse((output / "240p" / "comfyui" / "runtime").exists())
            self.assertFalse((output / "240p" / "h3cspeed" / "profile").exists())

    @unittest.skipUnless(os.name == "nt", "Windows dry-run executable boundary")
    def test_dry_run_never_launches_config_selected_powershell(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, value = fixture_config(root)
            sentinel = root / "inputs" / "sentinel.exe"
            sentinel.write_bytes(b"must not execute")
            value["powershell_executable"] = str(sentinel.resolve())
            config_path.write_text(json.dumps(value), encoding="utf-8")
            output = root / "sentinel-output"
            original_run = runner.subprocess.run
            launched: list[str] = []

            def record_run(argv, *args, **kwargs):
                launched.append(str(argv[0]))
                return original_run(argv, *args, **kwargs)

            with mock.patch.object(runner.subprocess, "run", side_effect=record_run):
                runner.create_dry_plan(config_path, output, ("240",))
            self.assertNotIn(str(sentinel.resolve()), launched)

    def test_execute_is_explicit_and_uses_fixed_h3_then_comfy_order(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, _ = fixture_config(root)
            output = root / "execute-output"
            plan_path, _ = runner.create_dry_plan(config_path, output, ("240",))
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            config, digest = runner.load_config(config_path)
            with mock.patch.object(runner, "_run_child", return_value=0) as launched:
                runner.execute_plan(plan, output, config, digest)
            self.assertEqual(launched.call_count, 2)
            self.assertEqual(launched.call_args_list[0].args[0][0],
                             config["powershell_executable"])
            self.assertIn("run_perf007_h3.ps1", launched.call_args_list[0].args[0][4])
            self.assertIn("perf002_comfy_trace.py", launched.call_args_list[1].args[0][1])
            self.assertEqual(launched.call_args_list[0].args[2], 4500.0)
            self.assertTrue((output / "240p" / "h3cspeed" /
                             "producer-private.log").is_file())
            self.assertTrue((output / "240p" / "comfyui" /
                             "producer-private.log").is_file())
            summary = json.loads(
                (output / "resolution-matrix-execution.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["kind"], "h3cspeed.resolution-matrix.execution")
            self.assertEqual(summary["status"], "EXECUTED_UNVERIFIED")
            self.assertEqual(summary["acceptance"]["speed_alignment"], "OBSERVED_ONLY")
            self.assertEqual(len(summary["profiles"]), 1)
            self.assertEqual(set(summary["profiles"][0]["engines"]),
                             {"h3cspeed", "comfyui"})
            self.assertEqual(summary["execution_order"], "h3cspeed_then_comfyui")

    def test_execute_can_reverse_each_profile_pair_for_counterbalance(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, _ = fixture_config(root)
            output = root / "reverse-output"
            plan_path, _ = runner.create_dry_plan(config_path, output, ("240",))
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            config, digest = runner.load_config(config_path)
            with mock.patch.object(runner, "_run_child", return_value=0) as launched:
                runner.execute_plan(plan, output, config, digest, reverse_order=True)
            self.assertEqual(launched.call_count, 2)
            self.assertIn("perf002_comfy_trace.py", launched.call_args_list[0].args[0][1])
            self.assertEqual(launched.call_args_list[1].args[0][0],
                             config["powershell_executable"])
            summary = json.loads(
                (output / "resolution-matrix-execution.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["execution_order"], "comfyui_then_h3cspeed")

    def test_reverse_order_requires_explicit_execute(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, _ = fixture_config(root)
            output = root / "reverse-dry-output"
            with mock.patch.object(runner, "_run_child") as launched:
                result = runner.main([
                    "--config", str(config_path), "--output-dir", str(output),
                    "--profiles", "240", "--reverse-order",
                ])
            self.assertEqual(result, 2)
            self.assertFalse(output.exists())
            launched.assert_not_called()

    def test_execute_summary_is_no_clobber(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, _ = fixture_config(root)
            output = root / "summary-output"
            plan_path, _ = runner.create_dry_plan(config_path, output, ("240",))
            summary = output / "resolution-matrix-execution.json"
            summary.write_text("protected", encoding="utf-8")
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            config, digest = runner.load_config(config_path)
            with mock.patch.object(runner, "_run_child") as launched:
                with self.assertRaisesRegex(runner.ContractError, "summary already exists"):
                    runner.execute_plan(plan, output, config, digest)
            launched.assert_not_called()

    def test_execute_rejects_tampered_command_and_canonical_png_clobber(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, _ = fixture_config(root)
            output = root / "protected-output"
            plan_path, _ = runner.create_dry_plan(config_path, output, ("240",))
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            config, digest = runner.load_config(config_path)
            tampered = copy.deepcopy(plan)
            tampered["commands"][0]["argv"][0] = "attacker.exe"
            with self.assertRaises(runner.ContractError):
                runner.execute_plan(tampered, output, config, digest)
            canonical = output / "240p" / "h3cspeed" / "conditioning.h3c.first.png"
            canonical.write_bytes(b"protected")
            with mock.patch.object(runner, "_run_child") as launched:
                with self.assertRaisesRegex(runner.ContractError, "no-clobber"):
                    runner.execute_plan(plan, output, config, digest)
            launched.assert_not_called()

    def test_execute_rechecks_shared_inputs_between_engines(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, value = fixture_config(root)
            output = root / "toctou-output"
            plan_path, _ = runner.create_dry_plan(config_path, output, ("240",))
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            config, digest = runner.load_config(config_path)
            prompt = Path(value["prompt_file"])

            def first_child(*_args, **_kwargs):
                prompt.write_text("changed between engines", encoding="utf-8")
                return 0

            with mock.patch.object(runner, "_run_child",
                                   side_effect=first_child) as launched:
                with self.assertRaisesRegex(runner.ContractError, "prompt file"):
                    runner.execute_plan(plan, output, config, digest)
            self.assertEqual(launched.call_count, 1)

    def test_cli_execute_opt_in_rebuilds_plan_and_launches_both_engines(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, _ = fixture_config(root)
            output = root / "cli-execute-output"
            with mock.patch.object(runner, "_run_child", return_value=0) as launched:
                result = runner.main([
                    "--config", str(config_path), "--output-dir", str(output),
                    "--profiles", "240", "--execute",
                ])
            self.assertEqual(result, 0)
            self.assertEqual(launched.call_count, 2)

    def test_cli_dry_then_execute_reuses_existing_plan_without_overwrite(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, _ = fixture_config(root)
            output = root / "two-step-output"
            dry_result = runner.main([
                "--config", str(config_path), "--output-dir", str(output),
                "--profiles", "240",
            ])
            self.assertEqual(dry_result, 0)
            plan_path = output / "resolution-matrix-plan.json"
            original_plan = plan_path.read_bytes()
            with mock.patch.object(runner, "_run_child", return_value=0) as launched:
                execute_result = runner.main([
                    "--config", str(config_path), "--output-dir", str(output),
                    "--profiles", "240", "--execute",
                ])
            self.assertEqual(execute_result, 0)
            self.assertEqual(launched.call_count, 2)
            self.assertEqual(plan_path.read_bytes(), original_plan)

    def test_cli_execute_existing_directory_without_plan_fails_closed(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, _ = fixture_config(root)
            output = root / "missing-plan-output"
            output.mkdir()
            with mock.patch.object(runner, "_run_child") as launched:
                result = runner.main([
                    "--config", str(config_path), "--output-dir", str(output),
                    "--profiles", "240", "--execute",
                ])
            self.assertEqual(result, 2)
            launched.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "Windows DACL contract")
    def test_windows_private_output_has_restrictive_inheritable_dacl(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path, value = fixture_config(root)
            output = root / "acl-output"
            runner.create_dry_plan(config_path, output, ("240",))
            environment = dict(os.environ)
            environment["H3CSPEED_TEST_ACL_PATH"] = str(output)
            script = r"""
$acl = Get-Acl -LiteralPath $env:H3CSPEED_TEST_ACL_PATH
$current = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$unexpected = @($acl.Access | Where-Object {
    $_.AccessControlType -eq [Security.AccessControl.AccessControlType]::Allow -and
    $_.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value -ne $current
})
Write-Output "$($acl.AreAccessRulesProtected),$($unexpected.Count)"
"""
            completed = subprocess.run(
                [value["powershell_executable"], "-NoProfile", "-NonInteractive",
                 "-Command", script], capture_output=True, text=True, check=False,
                shell=False, timeout=30, env=environment)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout.strip(), "True,0")

    @unittest.skipUnless(os.name == "nt", "Windows process-tree contract")
    def test_windows_timeout_terminates_descendant_process(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child_pid_file = root / "child.pid"
            log = root / "private.log"
            parent_code = (
                "import subprocess,sys,time; from pathlib import Path; "
                "child=subprocess.Popen([sys.executable,'-c',"
                "'import time; time.sleep(30)']); "
                "Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(30)"
            )
            with log.open("xb") as stream:
                with self.assertRaisesRegex(runner.ContractError, "tree timed out"):
                    runner._run_child(
                        [sys.executable, "-c", parent_code, str(child_pid_file)],
                        stream, 1.0)
            child_pid = int(child_pid_file.read_text(encoding="utf-8"))
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int,
                                             ctypes.c_uint32]
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            kernel32.WaitForSingleObject.restype = ctypes.c_uint32
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            handle = kernel32.OpenProcess(0x00100000, False, child_pid)
            if handle:
                try:
                    self.assertEqual(kernel32.WaitForSingleObject(handle, 0), 0,
                                     "descendant process survived the timeout")
                finally:
                    kernel32.CloseHandle(handle)


if __name__ == "__main__":
    unittest.main()
