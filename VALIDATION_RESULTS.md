# h3cspeed local validation results

- Generated: 2026-08-13 (Asia/Taipei)
- Platform: native Windows x64
- GPU: NVIDIA GeForce RTX 3070 Ti, CUDA `sm_86`, 8.00 GiB VRAM
- System RAM: 31.81 GiB
- Toolchain: CUDA Toolkit 13.2 / nvcc 13.2.78, Visual Studio Build Tools 2026,
  CMake 4.3.2, Ninja 1.13.2, ICU 76.1
- Build: `scripts/build-native.ps1 -BuildDirectory build-native-verify -CudaArchitectures 86`

## PASS

- Pinned upstream archive and prepared-tree hash verification, including a
  fresh forced bootstrap and post-patch verification.
- Python bytecode checks and all focused Python suites (POSIX-only launcher
  cases skip on native Windows).
- Public CUDA backend API coverage: 103/103.
- Windows Clang host syntax checks for all CUDA sources and portable C sources.
- Portable overlay configure/build and CTest: 9/9.
- Native CUDA/MSVC build and link: `h3cspeed.exe`,
  `h3cspeed-cuda-info.exe`, library, and focused tests.
- Native CTest: 16/16. This includes CUDA numeric-layout, ConvRot, NVFP4,
  converted-weight eviction/reload and scale-broadcast
  regressions, 64-bit `pread`/`stat` over a 5 GiB sparse
  file, POSIX spawn redirection, and real FFmpeg RGB+PCM encode, FFprobe, image
  decode, and audio decode. The FFmpeg test also reserves 48 CRT descriptors
  and runs two encoders concurrently to validate high-FD and handle-isolation
  behavior.
- Independent final C++ review: no remaining P0/P1. The review also verified
  that a missing `H3_FFMPEG` override fails with `ENOENT` instead of silently
  falling back to PATH.
- Clean PowerShell runtime with colocated ICU DLLs: CLI `--help` exit 0 and
  CUDA-info exit 0.
- The model-free Windows CUDA 13.2 / `sm_86` runtime ZIP was extracted under a
  path containing spaces and non-ASCII characters, then launched with PATH
  restricted to the bundle `bin` plus Windows system directories. CLI
  `--help` and CUDA-info both exited 0; CUDA-info identified the RTX 3070 Ti as
  `sm_86`. Every internal manifest hash matched, forbidden model/sidecar/media
  payload count was zero, and two independent packages were byte-for-byte
  deterministic. Each final archive carries its own adjacent `.sha256` file.
- CUDA 13.2 Compute Sanitizer memcheck on CUDA-info: 0 errors.
- Pinned MiniMax-H3 FL2VA snapshot revision `939557dc319dd91227e30195a763f272ba7f8765`
  downloaded outside the source archive; all 81 pinned manifest paths passed
  file-size and required-layout validation with no incomplete or Ref2VA files.
- A real 256×256, 22-frame, 20-step text-to-video run completed all 50 text
  layers, 20 denoise steps, 7 audio-VAE chunks, 36 video-VAE chunks and 22
  FFmpeg frames. The resulting H.264/AAC artifact independently passed full
  FFmpeg decode, frame-count and non-silent-audio checks; visual inspection
  matched the red-fox/snowy-pine prompt.
- The accepted run exercised automatic system-RAM plus SSD fallback on the
  8 GiB card: 5.93 GiB CUDA allocation budget, 1.50 GiB resident weight cache,
  11.53 GiB RAM cache, 1.90 GiB peak device allocation and 1.49 GiB peak
  resident working set.
- The experimental fast-quality preset also completed a real 864x480 output,
  124-frame (5.167-second), 20-step run with a 288x160 internal canvas,
  `--reuse 1`, `--core-reuse 4`, and SSD streaming. The CLI exited 0 after
  writing all 124 FFmpeg frames. The H.264 High/AAC-LC artifact passed full
  `ffmpeg -xerror` decoding; visual inspection of the first, middle and final
  frames showed a recognizable red fox in snow and pine woodland without the
  earlier block-noise failure. The low internal canvas causes visible softness
  and mild horizontal ghosting, so this is a speed/capacity preset rather than
  a native-480p quality claim. DiT streamed 216.050 GiB in 2035.254 seconds,
  peaked at 2070.95 MiB device memory, and the video VAE peaked at 9245.04 MiB
  host memory. End-to-end wall time was approximately 46 minutes on the test
  machine.
- The 39.55 GiB four-file ComfyUI T2V pack passed header/schema validation and
  native loading on the same RTX 3070 Ti. Focused CUDA tests cover the FL2VA
  INT8 ConvRot transform, contiguous Comfy Q/K/V layout, Qwen NVFP4 decoding,
  F16/I8 conversion and repeated converted-weight eviction/reload under a
  512 MiB VRAM budget. Compute Sanitizer memcheck reported 0 errors and
  racecheck reported 0 hazards for both ConvRot and NVFP4 focused tests.
- A real quantized 256x256, 22-frame, 4-step **Comfy-conditioned sidecar** smoke
  exited 0 and completed all native INT8 DiT, Audio VAE, Video VAE and FFmpeg
  stages. The H.264/AAC artifact passed full decode and visual inspection showed
  a recognizable red fox in snowy pine woodland. The sidecar was produced by
  the ComfyUI CUDA Qwen NVFP4/AWQ helper and validated by whole-model SHA-256
  before native launch. Direct native BF16-Qwen decoding remains experimental
  and is not this pack's semantic release gate.
- A real quantized **Comfy-conditioned sidecar** 256x256, 22-frame, 20-step run
  exited 0. The H.264/AAC MP4 is 82,341 bytes with SHA-256
  `BAB9018C73E6394039B99EC3F9F37E0C064B88657B6CC4A700D1E516617D7F4B`;
  full decode exited 0, audio decoded to 59,392 samples (mean -44.8 dB,
  max -25.5 dB), and first/middle/final frames showed a clear red fox in
  snowy pine woodland with no patch noise. DiT telemetry recorded 2,052.22 MiB
  peak device memory, 13,244.45 MiB peak host memory, 360.68 GiB uploads,
  357.75 GiB evictions and 100.56 GiB file fallback. The 11.241-second denoise
  profile is kernel/compute time and excludes offload wait; it is not an
  end-to-end wall-clock measurement.

Observed low-VRAM policy on the acceptance machine:

```text
offload mode: system-RAM + file fallback (automatic)
low-VRAM profile: yes
CUDA allocation budget: 5.94 GiB
resident GPU weight cache: 1.50 GiB
pinned host-memory cap: 128 MiB
RAM/file transfer staging: 64 MiB
```

## MANUAL_REQUIRED

- Resident-only versus system-RAM/file-offloaded numerical parity for the
  complete model, plus a matched performance comparison, is still pending.
- Complete-model resident-only versus offloaded parity remains separate from
  the passing focused converted-weight eviction/reload sanitizer test.
- Newer-than-Ampere NVIDIA architecture validation.
- Linux CUDA compilation, ELF startup and packaged dependency-closure evidence
  are produced by the pinned hosted workflow. Linux GPU inference remains
  unverified until the artifact is run on an `sm_86` Linux host; a hosted
  CPU-only build is not counted as GPU runtime PASS.

The source and native binaries are usable for CLI/model-directory workflows.
The remaining parity, racecheck and newer-architecture gates must pass before
calling this an architecture-wide production release.
