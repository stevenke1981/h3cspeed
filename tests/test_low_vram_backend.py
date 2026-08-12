#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
import os
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LowVramBackendTest(unittest.TestCase):
    def test_real_cli_parses_square_dimensions_without_aspect_error(self) -> None:
        """Parse the 256x256 baseline without loading a model or generating."""
        configured_cli = os.environ.get("H3CSPEED_CLI")
        if configured_cli:
            executable = Path(configured_cli)
        else:
            executable = ROOT / "build-native" / (
                "h3cspeed.exe" if os.name == "nt" else "h3cspeed"
            )
        if not executable.is_file():
            self.skipTest(
                "native h3cspeed CLI is not built "
                "(CTest supplies H3CSPEED_CLI from its target)"
            )
        with tempfile.TemporaryDirectory(prefix="h3cspeed-cli-parse-") as temporary:
            output = Path(temporary) / "not-created.mp4"
            completed = subprocess.run(
                [
                    str(executable),
                    "-d", str(Path(temporary) / "missing-model"),
                    "-p", "parse-only",
                    "-o", str(output),
                    "--width", "256", "--height", "256",
                    "--frames", "22", "--steps", "4",
                    "--layers", "50", "--reuse", "1", "--core-reuse", "1",
                    "--ssd-streaming",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        combined = f"{completed.stdout}\n{completed.stderr}".lower()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("missing required model file", combined)
        self.assertNotIn("aspect", combined)

    def test_explicit_offload_not_managed_memory(self) -> None:
        source = (ROOT / "src/h3_gpu_cuda.cu").read_text(encoding="utf-8")
        common = (ROOT / "src/h3_cuda_common.cuh").read_text(encoding="utf-8")
        self.assertNotIn("cudaMallocManaged", source)
        self.assertIn("h3cspeed_tensor_prepare", source)
        self.assertIn("cudaStreamWaitEvent", source)
        self.assertIn("last_use", common)
        self.assertIn("operation_epoch", common)

    def test_generated_int8_has_authoritative_ram_copy(self) -> None:
        source = (ROOT / "src/h3_gpu_cuda.cu").read_text(encoding="utf-8")
        self.assertIn("make_generated_weight_offloadable", source)
        self.assertIn("copy generated weight to system RAM", source)
        self.assertIn("generated int8 weights have no disk source", source.lower())
        self.assertIn("host_cache_candidate_locked", source)
        self.assertIn("!candidate->source_path", source)

    def test_failed_event_record_keeps_current_epoch_pinned(self) -> None:
        source = (ROOT / "src/h3_gpu_cuda.cu").read_text(encoding="utf-8")
        operation = source[
            source.index("void h3cspeed_operation_complete"):
            source.index("static h3_gpu_tensor *tensor_new_internal")
        ]
        self.assertIn("int all_recorded = 1", operation)
        self.assertIn("if (all_recorded)", operation)
        self.assertIn("gpu->operation_epoch++", operation)
        self.assertLess(operation.index("if (all_recorded)"),
                        operation.index("gpu->operation_epoch++"))

    def test_failed_file_refill_does_not_cache_invalid_bytes(self) -> None:
        source = (ROOT / "src/h3_gpu_cuda.cu").read_text(encoding="utf-8")
        refill = source[
            source.index("static int tensor_read_file"):
            source.index('extern "C" h3_gpu_tensor *h3_gpu_tensor_load_bf16')
        ]
        self.assertIn("if (ok) host_lru_append_locked", refill)
        self.assertIn("else host_backing_release_locked", refill)

    def test_scratch_uses_shared_budget(self) -> None:
        source = (ROOT / "src/h3_gpu_cuda.cu").read_text(encoding="utf-8")
        scratch = source[source.index("void *h3cspeed_scratch_reserve"):]
        self.assertIn("device_allocate_locked", scratch[:5000])
        self.assertIn("release_scratch_on_submit", source)

    def test_bf16_bias_uses_single_rounding_gpu_path(self) -> None:
        source = (ROOT / "src/h3_gpu_cuda.cu").read_text(encoding="utf-8")
        supported = source[
            source.index("static int h3cspeed_cublas_linear_supported"):
            source.index("int h3cspeed_linear")
        ]
        self.assertIn("int has_bias", supported)
        self.assertIn("output_dtype == H3_GPU_BF16 &&", supported)
        self.assertIn("!has_bias", supported)
        linear = source[source.index("int h3cspeed_linear"):]
        self.assertIn(
            "input->dtype, weight->dtype, output->dtype, bias != nullptr",
            linear,
        )

    def test_layer_norm_uses_centered_variance_pass(self) -> None:
        source = (ROOT / "src/h3_gpu_cuda.cu").read_text(encoding="utf-8")
        norm = source[
            source.index("__global__ static void norm_kernel"):
            source.index("static int norm_dispatch")
        ]
        self.assertIn("float centered_square = 0.0f", norm)
        self.assertIn("float centered = h3cspeed_device_load", norm)
        self.assertIn("centered_square = fmaf(centered, centered, centered_square)",
                      norm)
        self.assertIn("variance = shared[blockDim.x] / (float)width", norm)
        self.assertNotIn(
            "shared[blockDim.x] / (float)width - mean * mean", norm)

    def test_scale_add_uses_per_column_scale(self) -> None:
        source = (ROOT / "src/h3_gpu_cuda.cu").read_text(encoding="utf-8")
        kernel = source[
            source.index("__global__ static void scale_add_kernel"):
            source.index("extern \"C\" int h3_gpu_qkv_rope_f32")
        ]
        self.assertIn("h3cspeed_device_load(scale, scale_dtype, column)", kernel)
        self.assertNotIn("scale_elements", kernel)
        self.assertIn("scale->elements < width", kernel)

    def test_video_qkv_rope_interleaved_rms_numeric_reference(self) -> None:
        """Exercise the Metal video-QKV contract with distinct synthetic rows."""
        source = (ROOT / "src/h3_gpu_cuda.cu").read_text(encoding="utf-8")
        video = source[
            source.index('extern "C" int h3_gpu_video_qkv_rope_f32'):
        ]
        self.assertIn("rope_half, epsilon, 1, 0, 1", video)
        self.assertIn("((size_t)head * 3) * head_dim + dimension", source)
        self.assertIn("((size_t)head * sequence + row) * head_dim", source)

        sequence, heads, head_dim, rope_half = 2, 2, 4, 2
        epsilon = 1e-5
        # Per-row/head interleaved [Q(dim), K(dim), V(dim)] storage.
        qkv = [
            3.0, 4.0, 0.0, 0.0,  6.0, 8.0, 0.0, 0.0,  9.0, 10.0, 11.0, 12.0,
            1.0, 0.0, 0.0, 0.0,  2.0, 0.0, 0.0, 0.0,  13.0, 14.0, 15.0, 16.0,
            5.0, 12.0, 0.0, 0.0,  8.0, 15.0, 0.0, 0.0,  17.0, 18.0, 19.0, 20.0,
            2.0, 0.0, 0.0, 0.0,  4.0, 0.0, 0.0, 0.0,  21.0, 22.0, 23.0, 24.0,
        ]
        cosine = [0.0, 0.0] * sequence
        sine = [1.0, 1.0] * sequence
        outputs = [[[] for _ in range(heads)] for _ in range(sequence)]
        for row in range(sequence):
            for head in range(heads):
                base = (row * heads + head) * head_dim * 3
                q = qkv[base:base + head_dim]
                k = qkv[base + head_dim:base + 2 * head_dim]
                v = qkv[base + 2 * head_dim:base + 3 * head_dim]
                q_inv = 1.0 / math.sqrt(sum(x * x for x in q) / head_dim + epsilon)
                k_inv = 1.0 / math.sqrt(sum(x * x for x in k) / head_dim + epsilon)
                q = [x * q_inv for x in q]
                k = [x * k_inv for x in k]
                q_rotated, k_rotated = list(q), list(k)
                for dimension in range(rope_half):
                    paired = dimension + rope_half
                    c = cosine[row * rope_half + dimension]
                    s = sine[row * rope_half + dimension]
                    q_rotated[dimension] = q[dimension] * c - q[paired] * s
                    q_rotated[paired] = q[paired] * c + q[dimension] * s
                    k_rotated[dimension] = k[dimension] * c - k[paired] * s
                    k_rotated[paired] = k[paired] * c + k[dimension] * s
                outputs[row][head] = (q_rotated, k_rotated, v)

        # CUDA SDPA consumes head-major [head, row, dim] outputs.
        query, key, value = [], [], []
        for head in range(heads):
            for row in range(sequence):
                q, k, v = outputs[row][head]
                query.extend(q)
                key.extend(k)
                value.extend(v)
        self.assertEqual(value, [9.0, 10.0, 11.0, 12.0,
                                 17.0, 18.0, 19.0, 20.0,
                                 13.0, 14.0, 15.0, 16.0,
                                 21.0, 22.0, 23.0, 24.0])
        expected_query = [0.0, 0.0, 1.2, 1.6,
                          0.0, 0.0, 10.0 / 13.0, 24.0 / 13.0,
                          0.0, 0.0, 2.0, 0.0,
                          0.0, 0.0, 2.0, 0.0]
        for actual, expected in zip(query, expected_query):
            self.assertAlmostEqual(actual, expected, places=3)
        expected_key = [0.0, 0.0, 1.2, 1.6,
                        0.0, 0.0, 16.0 / 17.0, 30.0 / 17.0,
                        0.0, 0.0, 2.0, 0.0,
                        0.0, 0.0, 2.0, 0.0]
        for actual, expected in zip(key, expected_key):
            self.assertAlmostEqual(actual, expected, places=3)

    def test_fused_final_head_inverse_is_per_row_f32(self) -> None:
        source = (ROOT / "src/h3_gpu_cuda.cu").read_text(encoding="utf-8")
        kernel = source[
            source.index("__global__ static void adaln_kernel"):
            source.index("static int adaln_dispatch")
        ]
        dispatch = source[
            source.index("static int adaln_dispatch"):
            source.index('extern "C" int h3_gpu_adaln_f32')
        ]
        fused = source[
            source.index('extern "C" int h3_gpu_adaln_linear_bf16'):
            source.index("__global__ static void scale_add_kernel")
        ]
        self.assertIn("float *inverse_output", kernel)
        self.assertIn("if (inverse_output && threadIdx.x == 0)", kernel)
        self.assertIn("inverse_output[row] = inverse_value", kernel)
        self.assertIn("h3_gpu_tensor *inverse", dispatch)
        self.assertIn("inverse->dtype != H3_GPU_F32 || inverse->elements < rows",
                      dispatch)
        self.assertIn("h3cspeed_tensor_wait(gpu, inverse)", dispatch)
        self.assertIn("inverse ? static_cast<float *>(inverse->data) : nullptr",
                      dispatch)
        self.assertIn("inverse->dtype != H3_GPU_F32 || inverse->elements < rows",
                      fused)
        self.assertIn("h3_gpu_tensor_new_bf16(gpu, normalized_elements)", fused)
        self.assertIn("adaln_dispatch(gpu, normalized, inverse", fused)
        self.assertIn("inverse RMS must be F32 with at least", fused)

    def test_3070ti_wrapper_enables_capacity_flags(self) -> None:
        wrapper = (ROOT / "scripts/run-3070ti-8gb.sh").read_text(encoding="utf-8")
        profile = (ROOT / "profiles/rtx3070ti-8gb.env").read_text(encoding="utf-8")
        for value in (
            "H3_CUDA_OFFLOAD", "H3_CUDA_VRAM_BUDGET_MIB",
            "H3_CUDA_WEIGHT_CACHE_MIB", "H3_CUDA_RELEASE_SCRATCH",
            "--ssd-streaming", "--frames 22",
            "--width 256", "--height 256",
        ):
            self.assertIn(value, wrapper)
        self.assertNotIn('args+=(--token-reduction', wrapper)
        self.assertNotIn('args+=(--layers', wrapper)
        self.assertNotIn('args+=(--core-reuse', wrapper)
        self.assertIn('args+=(--width 256 --height 256)', wrapper)
        self.assertIn("H3_CUDA_VRAM_BUDGET_MIB=5888", profile)
        self.assertIn("H3_CUDA_PINNED_HOST_MIB=128", profile)
        smoke = (ROOT / "scripts/smoke-3070ti-8gb.sh").read_text(
            encoding="utf-8")
        self.assertIn("h3cspeed-3070ti-8gb", smoke)
        self.assertIn("--seed 42", smoke)
        self.assertIn("--frames 22", smoke)
        self.assertIn("--steps 4", smoke)
        self.assertIn("--width 256 --height 256", smoke)
        self.assertIn("--layers 50", smoke)
        self.assertIn("--reuse 1", smoke)
        self.assertIn("--core-reuse 1", smoke)
        self.assertIn("--ssd-streaming", smoke)
        self.assertNotIn("--token-reduction", smoke)
        self.assertNotIn("--reuse 2", smoke)
        self.assertNotIn("--core-reuse 4", smoke)

    def test_text_sidecar_branch_supports_fl2va_and_is_fail_closed(self) -> None:
        bootstrap = (ROOT / "scripts/bootstrap.py").read_text(encoding="utf-8")
        prepared = ROOT / "third_party/h3/h3.c"
        if not prepared.is_file():
            # Public clones intentionally omit the prepared upstream tree. The
            # bootstrap patch is the authoritative source in overlay-only CI.
            for marker in (
                    "H3CSPEED_TEXT_EMBEDDING",
                    "H3CSPEED_TEXT_ENCODER_SHA256",
                    "h3_parse_sha256_hex",
                    "h3cspeed_text_embedding_load_file",
                    "skipping native Qwen text encoder",
                    "h3_cache_clear(ctx)",
                    "patch_frame_anchor_allocation",
                    "patch_sidecar_keyframe_hash_guard"):
                self.assertIn(marker, bootstrap)
            self.assertIn("patch_text_embedding_sidecar(root)", bootstrap)
            return
        source = prepared.read_text(encoding="utf-8")
        self.assertIn('getenv("H3CSPEED_TEXT_EMBEDDING")', source)
        self.assertIn('getenv("H3CSPEED_TEXT_ENCODER_SHA256")', source)
        self.assertIn("h3_parse_sha256_hex", source)
        self.assertIn("text_sidecar_sha256", source)
        self.assertIn("h3cspeed_keyframe_snapshot_create", source)
        self.assertIn("first_snapshot.sha256", source)
        self.assertIn("last_snapshot.sha256", source)
        self.assertIn("first_frame_path = first_snapshot.path", source)
        self.assertIn("last_frame_path = last_snapshot.path", source)
        self.assertIn("h3cspeed_keyframe_snapshot_discard", source)
        self.assertIn("64 hexadecimal", source)
        self.assertIn("does not support Ref2VA references", source)
        self.assertIn("using FL2VA I2V text sidecar", source)
        self.assertIn("h3cspeed_text_embedding_load_file_ex", source)
        self.assertIn("h3cspeed_text_embedding_load_file", source)
        self.assertIn("skipping native Qwen text encoder", source)
        self.assertIn("params->reference_count && !reference_visual_indices", source)
        self.assertIn('h3_key_file(&key, "text-sidecar", sidecar)', source)
        self.assertIn('"|text-sha=%s"', source)
        self.assertIn("if (text_sidecar_path) h3_cache_clear(ctx);", source)
        self.assertIn("patch_text_embedding_sidecar(root)", bootstrap)
        sidecar_branch = source[source.index("int text_ok;"):source.index(
            "conditioned = visual_count", source.index("int text_ok;"))]
        self.assertIn("if (text_sidecar_path)", sidecar_branch)
        self.assertIn("else {", sidecar_branch)
        self.assertIn("h3_text_encode_bf16", sidecar_branch)
        self.assertNotIn("h3_text_encode_bf16", sidecar_branch[:sidecar_branch.index("else {")])

    def test_sidecar_stat_key_is_linux_and_macos_portable(self) -> None:
        bootstrap = (ROOT / "scripts/bootstrap.py").read_text(encoding="utf-8")
        for marker in ("#elif defined(__APPLE__)", "status.st_mtim.tv_sec",
                       "status.st_mtimespec.tv_sec", "#include <string.h>"):
            self.assertIn(marker, bootstrap)
        prepared = ROOT / "third_party/h3/h3.c"
        if prepared.is_file():
            source = prepared.read_text(encoding="utf-8")
            self.assertLess(source.index("#include <string.h>"),
                            source.index("h3_parse_sha256_hex"))
            self.assertIn("#elif defined(__APPLE__)", source)
            self.assertIn("status.st_mtim.tv_sec", source)

    def test_ffmpeg_ssize_max_has_its_posix_header(self) -> None:
        bootstrap = (ROOT / "scripts/bootstrap.py").read_text(encoding="utf-8")
        self.assertIn("patch_ffmpeg_limits(root)", bootstrap)
        self.assertIn('marker + "#include <limits.h>\\n"', bootstrap)
        prepared = ROOT / "third_party/h3/h3_ffmpeg.c"
        if prepared.is_file():
            source = prepared.read_text(encoding="utf-8")
            self.assertIn("SSIZE_MAX", source)
            self.assertIn("#include <limits.h>", source)

    def test_linux_random_seed_does_not_require_new_glibc(self) -> None:
        bootstrap = (ROOT / "scripts/bootstrap.py").read_text(encoding="utf-8")
        self.assertIn("patch_linux_random_seed(root)", bootstrap)
        self.assertIn("#include <sys/random.h>", bootstrap)
        self.assertIn("getrandom(cursor, remaining, 0)", bootstrap)
        prepared = ROOT / "third_party/h3/h3_cli.c"
        if prepared.is_file():
            source = prepared.read_text(encoding="utf-8")
            self.assertIn("#if defined(__linux__)", source)
            self.assertIn("getrandom(cursor, remaining, 0)", source)


if __name__ == "__main__":
    unittest.main()
