# Changelog

## 0.2.0 — RTX 3070 Ti 8 GiB / RAM offload preview

- Added automatic low-VRAM mode for CUDA devices with 10 GiB VRAM or less.
- Added a hard tracked CUDA allocation budget covering weights, activations and
  scratch memory.
- Added an event-safe VRAM LRU for read-only weights with per-tensor upload-ready
  and last-compute-use events.
- Added system-RAM backing for file-loaded BF16/F32 weights and generated INT8
  weights/scales.
- Added a second RAM LRU that drops only file-reconstructible copies; generated
  INT8 weights remain authoritative in RAM.
- Added safetensors/file fallback through a bounded pinned staging buffer.
- Added configurable VRAM, GPU weight cache, RAM cache, pinned-memory and staging
  limits.
- Added submit-time scratch release for lower phase-boundary peaks.
- Added `h3cspeed-cuda-info` reporting for the effective offload policy.
- Added RTX 3070 Ti build/run/smoke helpers and a reusable environment profile.
- Added portable offload-policy tests, low-VRAM static regression checks and
  executable wrapper-profile tests.
- Enforced the upstream minimum trained 22-frame decoder chunk in the RTX
  3070 Ti wrapper and deterministic smoke test.
- Hardened failed CUDA event recording, device free, scratch release and
  file-refill paths so tensors are never made evictable prematurely.
- Updated documentation and validation gates for WSL2, 8 GiB VRAM and 64–96 GiB
  system-memory configurations.
- Completed native NVCC/CUDA 13.2 validation on an RTX 3070 Ti, including
  17/17 focused CTest checks and a real 20-step, 22-frame H.264/AAC generation
  matching the red-fox/snowy-pine prompt.
- Added an experimental 480p fast-quality preset using a 288×160 internal
  canvas and `--core-reuse 4`; its 124-frame / 5.167-second path completed with
  exit 0 and passed full decode plus visual prompt checks on the RTX 3070 Ti.
  The preset remains explicitly speed-oriented because internal-canvas
  upscaling introduces visible softness and mild horizontal ghosting.
- Added native support for the 39.55 GiB four-file ComfyUI T2V pack: direct
  FL2VA INT8 ConvRot weights, Qwen NVFP4/AWQ decoding with activation-side
  pre-scales, F16 Video VAE conversion and F32 Audio VAE loading.
- Added a header-only, fail-closed model preparer that validates the quantized
  schemas and creates a native model root with hardlinks instead of copying the
  39.55 GiB payload.
- Added ConvRot, NVFP4, converted-weight eviction/reload and AdaLN curve
  regressions while preserving the public CUDA API at 103/103.
- Added an explicit ComfyUI CUDA conditioning bridge for the quantized pack.
  `encode_h3_quantized_prompt.py` emits a prompt-bound BF16 sidecar with token
  IDs/tags and a whole-Qwen SHA-256; the native runtime accepts it only through
  `H3CSPEED_TEXT_EMBEDDING` plus `H3CSPEED_TEXT_ENCODER_SHA256`, and only for
  T2V and fail-closed FL2VA first/last-keyframe I2V. Sidecar v2 binds the
  canonical FFmpeg-processed keyframe SHA-256, render geometry and exact
  Comfy Picture/vision-pad token sequence; Ref2VA remains rejected.
- Added `scripts/run-h3-quantized.ps1`, a fail-closed Windows wrapper that
  discovers or accepts the ComfyUI venv Python, verifies the helper-reported
  model fingerprint against `Get-FileHash`, and launches the native INT8
  DiT/VAE path with 20-step, 256x256, 22-frame defaults. Direct native
  BF16-Qwen decoding remains experimental; the wrapper is the usable semantic
  route for this quantized pack.
- Added explicit internal-render geometry and deterministic seed forwarding to
  the quantized wrapper. Added a cross-platform, resumable 60-second runner
  that verifies twelve 124-frame T2V/FL2VA-I2V clips independently, binds
  resume state to the prompt/model/binary/settings, and normalizes the final
  output to exactly 1,440 frames at 24 fps.
- Added static CPU-only wrapper checks. Real 4-step and 20-step sidecar runs
  exited 0, passed full decode and showed a recognizable fox; the 20-step
  artifact and measured offload telemetry are recorded in
  `VALIDATION_RESULTS.md`.
- Completed a real twelve-segment, resumable 20-step FL2VA I2V acceptance on an
  RTX 3070 Ti 8 GiB. The final 864x480 artifact passed the exact 1,440-frame,
  60.000000-second, 32 kHz stereo sample-count and full-decode gates. The
  capacity/workflow is validated; the 288x160 internal render remains visibly
  soft and artifact-prone and is not labeled production-quality.
- Added opt-in `H3_CUDA_ATTENTION=sage`: per-token INT8 Q/K, DP4A QK, FP32
  online softmax and BF16 V/output for eligible attention shapes. Unsupported
  CUDA shapes/dtypes remain on the native GPU attention path; there is no CPU
  fallback. The RTX 3070 Ti focused B=1/H=56/N=800/D=128 benchmark measured a
  1.073x speedup with cosine 0.999999. The measured peak device allocation was
  55.03 MiB; full command/toolchain metadata is recorded in the README.
- Added model-free Windows and Linux CUDA 13.2 / `sm_86` runtime archives with
  private CUDA/ICU dependency closure, redistribution notices, deterministic
  file manifests and fail-closed exclusion of models, sidecars and media.
- Added a pinned Linux GitHub Actions binary build; Windows binaries are built
  and tested on the local RTX 3070 Ti acceptance host. Hosted Linux evidence is
  explicitly limited to compile/link/startup and embedded architecture checks;
  GPU inference remains a separate `sm_86` acceptance gate.

## 0.1.0 — engineering preview

- Added a Linux/WSL2 NVIDIA CUDA backend for the pinned `antirez/h3.c` public
  GPU API while retaining the upstream C model, safetensor, CLI and FFmpeg
  layers.
- Added BF16/FP32 cuBLAS GEMM, dynamic INT8 paths, norms, RoPE, AdaLN, gating,
  memory-bounded online-softmax attention, token reduction and Euler kernels.
- Added generic CUDA AudioVAE/VideoVAE primitives.
- Added the released fused 12-tap polyphase FIR + SnakeBeta alias-free audio
  activation and a staged-vs-fused CPU identity test.
- Replaced Apple Foundation tokenizer code with portable ICU + yyjson C and
  Apple Accelerate scaling with a portable Lanczos3 scaler.
- Added pinned-source bootstrap/hash verification, CMake, WSL2 helpers,
  interface-coverage checks, source syntax checks and portable tests.
