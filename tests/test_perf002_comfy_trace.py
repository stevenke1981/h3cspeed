#!/usr/bin/env python3
"""Portable contract tests for the real ComfyUI PERF-002 driver."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "perf002_comfy_trace.py"
    spec = importlib.util.spec_from_file_location("perf002_comfy_trace", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Perf002ComfyTraceTests(unittest.TestCase):
    def test_workflow_is_real_first_frame_fl2va_contract(self) -> None:
        producer = load_module()
        workflow = producer._build_workflow(
            "reference.png", "private prompt", "fl2va.safetensors",
            "qwen.safetensors", "video.safetensors", "audio.safetensors")
        self.assertEqual(workflow["6"]["class_type"], "MiniMaxH3AudioConditioningT8")
        self.assertEqual(workflow["6"]["inputs"]["task_type"], "I2VA")
        self.assertEqual(workflow["6"]["inputs"]["first_frame"], ["5", 0])
        self.assertEqual(workflow["7"]["inputs"]["sampler_name"], "dual_clock_euler")
        self.assertEqual(workflow["7"]["inputs"]["scheduler"], "native_flow")
        self.assertEqual(workflow["10"]["class_type"], "SamplerCustomAdvanced")
        self.assertEqual(workflow["11"]["class_type"], "MiniMaxH3AVDecodeT8")
        self.assertEqual(workflow["13"]["class_type"], "SaveVideo")
        self.assertEqual(workflow["8"]["inputs"]["noise_seed"], 42)
        self.assertNotIn("testsrc2", json.dumps(workflow))

    def test_workflow_supports_h3_aligned_480p_five_second_contract(self) -> None:
        producer = load_module()
        workflow = producer._build_workflow(
            "reference.png", "a presenter speaks", "model.safetensors",
            "clip.safetensors", "video.safetensors", "audio.safetensors",
            frames=124)
        self.assertEqual(workflow["6"]["inputs"]["length"], 124)
        self.assertEqual(workflow["12"]["inputs"]["fps"], 24.0)

    def test_publish_json_is_no_clobber(self) -> None:
        producer = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "trace.json"
            producer._publish_json(destination, {"z": 1, "a": [True]})
            self.assertEqual(destination.read_text(encoding="utf-8"),
                             '{"a":[true],"z":1}\n')
            with self.assertRaisesRegex(producer.TraceError, "overwrite"):
                producer._publish_json(destination, {"a": 2})

    def test_existing_media_target_is_rejected_before_runtime(self) -> None:
        producer = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "output.mp4"
            destination.write_bytes(b"existing")
            with self.assertRaisesRegex(producer.TraceError, "new absolute"):
                producer._new_destination(destination, "output media")

    def test_bound_file_rejects_symlink_ancestor(self) -> None:
        producer = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            real.mkdir()
            source = real / "source.py"
            source.write_text("# fixture\n", encoding="utf-8")
            linked = root / "linked"
            try:
                linked.symlink_to(real, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation is unavailable")
            with self.assertRaisesRegex(producer.TraceError, "links or reparse"):
                producer._regular(linked / source.name, "bound source")

    def test_comfy_cli_disables_unrelated_custom_nodes(self) -> None:
        producer = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            runtime.mkdir()
            model_root = root / "models"
            model_root.mkdir()
            argv = producer._comfy_argv(root, runtime, model_root, 18199)
        self.assertIn("--disable-all-custom-nodes", argv)
        allow = argv.index("--whitelist-custom-nodes")
        self.assertEqual(argv[allow + 1], producer.T8_NODE_DIRECTORY)

    def test_hooks_require_sampler_scope_for_sage_counts(self) -> None:
        producer = load_module()
        state = {"sampling_active": False, "setup_success": False,
                 "sample_success": False, "raw_audio_protocol": False,
                 "audio_steps": 0, "sigma_video": [], "sigma_audio": [],
                 "sage_attempts": 0, "sage_hits": 0, "fallbacks": 0,
                 "all_bf16": True, "in_sage": 0}

        class Sigma:
            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return [1.0, 0.5, 0.0]

            def __len__(self):
                return 3

        fake = types.ModuleType("_perf002_fake_sampling")
        fake_torch = types.ModuleType("_perf002_fake_torch")
        fake_torch.bfloat16 = object()
        fake.time_shift_sigma = lambda value, _video, _audio: value
        fake.sageattn = lambda *args, **kwargs: "sage"
        fake.attention_pytorch = lambda *args, **kwargs: "pytorch"

        def original_attention_sage(*args, **kwargs):
            return fake.sageattn(*args, **kwargs)

        fake.attention_sage = original_attention_sage
        # These are the aliases captured by Comfy's minimax model at import
        # time.  The producer must rebind them before queueing the graph.
        fake.optimized_attention = original_attention_sage
        fake.optimized_attention_masked = original_attention_sage
        fake.SAGE_ATTENTION_IS_AVAILABLE = True
        consumer = types.ModuleType("comfy.ldm.minimax.model")
        consumer.optimized_attention = original_attention_sage

        def original_setup(*args, **kwargs):
            return (None, None, Sigma())

        def original_sample(*args, **kwargs):
            q = types.SimpleNamespace(dtype=fake_torch.bfloat16)
            consumer.optimized_attention(q, q, q, 1)
            return "latent"

        fake.setup_dual_clock_sampling = original_setup
        fake.sample_minimax_h3_dual_clock_euler = original_sample
        previous_torch = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        sys.modules[fake.__name__] = fake
        sys.modules[consumer.__name__] = consumer
        try:
            producer._install_runtime_hooks(state)
            fake.setup_dual_clock_sampling(
                None, None, 2, 12.0, 3.0, "dual_clock_euler", "native_flow")
            fake.sample_minimax_h3_dual_clock_euler(
                None, None, Sigma(), audio_velocity_is_raw=True)
        finally:
            sys.modules.pop(fake.__name__, None)
            sys.modules.pop(consumer.__name__, None)
            if previous_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = previous_torch
        self.assertTrue(state["setup_success"])
        self.assertTrue(state["sample_success"])
        self.assertTrue(state["raw_audio_protocol"])
        self.assertEqual(state["audio_steps"], 2)
        self.assertEqual(state["sage_attempts"], 1)
        self.assertEqual(state["sage_hits"], 1)
        self.assertEqual(state["fallbacks"], 0)
        self.assertTrue(state["all_bf16"])
        self.assertIn("comfy.ldm.minimax.model",
                      state["attention_alias_rebound_modules"])

    def test_hooks_count_bf16_failure_and_sage_fallback(self) -> None:
        producer = load_module()
        state = {"sampling_active": False, "setup_success": False,
                 "sample_success": False, "raw_audio_protocol": False,
                 "audio_steps": 0, "sigma_video": [], "sigma_audio": [],
                 "sage_attempts": 0, "sage_hits": 0, "fallbacks": 0,
                 "all_bf16": True, "in_sage": 0}

        class Sigma:
            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return [1.0, 0.5, 0.0]

            def __len__(self):
                return 3

        fake = types.ModuleType("_perf002_fake_fallback")
        fake_torch = types.ModuleType("_perf002_fake_torch_fallback")
        fake_torch.bfloat16 = object()
        fake.time_shift_sigma = lambda value, _video, _audio: value

        def failing_sage(*args, **kwargs):
            raise RuntimeError("test Sage failure")

        fake.sageattn = failing_sage
        fake.attention_pytorch = lambda *args, **kwargs: "pytorch"

        def original_attention_sage(*args, **kwargs):
            try:
                return fake.sageattn(*args, **kwargs)
            except RuntimeError:
                return fake.attention_pytorch(*args, **kwargs)

        fake.attention_sage = original_attention_sage
        fake.optimized_attention = original_attention_sage
        fake.optimized_attention_masked = original_attention_sage
        fake.SAGE_ATTENTION_IS_AVAILABLE = True
        consumer = types.ModuleType("_perf002_fake_fallback_consumer")
        consumer.optimized_attention = original_attention_sage

        fake.setup_dual_clock_sampling = lambda *args, **kwargs: (None, None, Sigma())

        def original_sample(*args, **kwargs):
            q = types.SimpleNamespace(dtype=object())
            consumer.optimized_attention(q, q, q, 1)
            return "latent"

        fake.sample_minimax_h3_dual_clock_euler = original_sample
        previous_torch = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        sys.modules[fake.__name__] = fake
        sys.modules[consumer.__name__] = consumer
        try:
            producer._install_runtime_hooks(state)
            fake.sample_minimax_h3_dual_clock_euler(
                None, None, Sigma(), audio_velocity_is_raw=True)
        finally:
            sys.modules.pop(fake.__name__, None)
            sys.modules.pop(consumer.__name__, None)
            if previous_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = previous_torch
        self.assertEqual(state["sage_attempts"], 1)
        self.assertEqual(state["sage_hits"], 0)
        self.assertEqual(state["fallbacks"], 1)
        self.assertFalse(state["all_bf16"])


if __name__ == "__main__":
    unittest.main()
