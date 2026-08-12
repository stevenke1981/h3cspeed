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
  12/12 focused CTest checks and a real 20-step, 22-frame H.264/AAC generation
  matching the red-fox/snowy-pine prompt.
- Added an experimental 480p fast-quality preset using a 288×160 internal
  canvas and `--core-reuse 4`; its 124-frame / 5.167-second path completed with
  exit 0 and passed full decode plus visual prompt checks on the RTX 3070 Ti.
  The preset remains explicitly speed-oriented because internal-canvas
  upscaling introduces visible softness and mild horizontal ghosting.

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
