# h3cspeed local validation results

- Generated: 2026-08-12 (Asia/Taipei)
- Platform: native Windows x64
- GPU: NVIDIA GeForce RTX 3070 Ti, CUDA `sm_86`, 8.00 GiB VRAM
- System RAM: 31.81 GiB
- Toolchain: CUDA Toolkit 13.2 / nvcc 13.2.78, Visual Studio Build Tools 2026,
  CMake 4.3.2, Ninja 1.13.2, ICU 76.1
- Build: `scripts/build-native.ps1 -BuildDirectory build-native-verify -CudaArchitectures 86`

## PASS

- Pinned upstream archive and prepared-tree hash verification, including a
  fresh forced bootstrap and post-patch verification.
- Python bytecode checks and 24 Python tests (7 POSIX shell-launcher tests
  skipped on native Windows).
- Public CUDA backend API coverage: 103/103.
- Windows Clang host syntax checks for all CUDA sources and portable C sources.
- Portable overlay configure/build and CTest: 8/8.
- Native CUDA/MSVC build and link: `h3cspeed.exe`,
  `h3cspeed-cuda-info.exe`, library, and focused tests.
- Native CTest: 12/12. This includes CUDA numeric-layout and scale-broadcast
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
- Compute Sanitizer racecheck of repeated file-backed and generated-INT8
  eviction/reload cycles: requires the same model-level fixture.
- Newer-than-Ampere NVIDIA architecture validation.

The source and native binaries are usable for CLI/model-directory workflows.
The remaining parity, racecheck and newer-architecture gates must pass before
calling this an architecture-wide production release.
