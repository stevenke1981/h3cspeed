# Validation plan

Passing portable checks proves source consistency, not CUDA correctness. The
release remains an engineering preview until the real-GPU gates below pass.

## Gate 0 — portable overlay checks

```bash
python3 scripts/validate_local.py
```

Acceptance:

- 103/103 backend symbols are present;
- all CUDA translation units pass strict host-side CUDA syntax parsing;
- offload policy tests pass for 8 GiB auto mode, large-card resident mode,
  explicit budgets and invalid configuration;
- metadata, online-softmax, alias-free Snake and scaler tests pass;
- CMake portable build and CTest pass;
- shell scripts parse successfully.

## Gate 1 — NVCC build on RTX 3070 Ti

```bash
H3CSPEED_CUDA_ARCHITECTURES=86 ./scripts/build.sh
ctest --test-dir build --output-on-failure
./build/h3cspeed-cuda-info
```

Acceptance:

- NVCC compiles and links all four CUDA units;
- device is reported as `sm_86` with approximately 8 GiB total VRAM;
- low-VRAM offload is automatic;
- budget, weight cache, RAM cache and staging values match the selected profile.

## Gate 2 — memory-manager unit stress on GPU

`test_cuda_offload` creates multiple multi-chunk file-backed tensors under a
small artificial VRAM budget, forces eviction/reload, and validates canaries at
every staging-slot boundary and tensor tail. CTest runs both the legacy
synchronous path and `H3_CUDA_ASYNC_REFILL=1`; the two cases use unique fixture
names so they are safe under parallel CTest. Generated-INT8 cycling remains a
separate completion gate.

The async case also sets `H3_CUDA_REFILL_TRACE=1`. After a warm-up refill it
queues bounded compute kernels, refills a second 32 MiB file-backed tensor, and
requires all 16 exact 2 MiB DMA intervals to be queryable. A PASS requires at
least 0.05 ms of strict CUDA-event intersection between a DMA-only interval and
the compute interval, a system-visible GPU-start handshake before prepare, an
uncompleted compute end when prepare returns, ordered same-slot reuse, a
host-read span nested in the compute window, and an unchanged full fixture
size/hash. CPU read and CUDA event clocks are not subtracted from one another.
Acceptance:

- no use-after-free under Compute Sanitizer;
- a pinned current-operation tensor is never selected for eviction;
- upload-ready and last-use event ordering is correct;
- generated INT8 data survives multiple VRAM evict/reload cycles bit-for-bit;
- RAM pressure drops only file-reconstructible host copies;
- scratch growth and submit-time release update tracked live bytes correctly;
- shutdown leaves zero tracked device and host-cache bytes.

Run:

```bash
compute-sanitizer --tool memcheck ./build/test_cuda_offload
compute-sanitizer --tool racecheck ./build/test_cuda_offload
```

The focused file-backed test is necessary but not sufficient for PERF-005. It
proves non-zero H2D/compute overlap and host-read ordering for this synthetic
fixture, not an end-to-end model speedup. Generated-INT8 repeated eviction,
pageable-allocation failure injection, resident/offloaded model parity, and the
fixed 22-frame upload-wait reduction must still be measured before the phase is
marked complete.

PERF-006 adds `test_cuda_dit_prefetch` in two CTest variants. With
`H3_CUDA_DIT_PREFETCH=1`, the fixture reserves future device storage before
starting compute, then calls the private upload helper while a device-confirmed
kernel is active. It measures every DMA interval and requires at least one
strict CUDA-event intersection, checks that the future tensor is not pinned by
the current epoch, preserves the full source-file hash plus every staging-chunk
boundary, and evicts/reloads file-backed canaries. With
`H3_CUDA_DIT_PREFETCH=0`, both helpers are no-ops and the existing lazy upload
path remains responsible for the same canary. These are focused synthetic
primitive proofs only: the released DiT block schedule does not call the
helpers. The 22-frame exclusive upload-wait gate, generated-INT8/pageable
failure gates, and real model/media parity remain `NOT_RUN`.

## Gate 3 — released operation fixtures

Port the pinned upstream BF16, audio, DiT-block and VAE fixtures into CMake.
Initial tolerances:

```text
F32 elementwise/norm: max_abs <= 2e-5, relative_L2 <= 2e-5
BF16 boundaries:      exact BF16 where rounding is contractual
GEMM/attention:       max_abs <= 5e-3, relative_L2 <= 2e-3
INT8 path:            per-layer relative-L2 plus end-quality comparison
```

Compare resident mode and offload mode using the same inputs. Offload must not
change tensor values; only timing and memory placement may differ.

## Gate 4 — 8 GiB deterministic smoke

```bash
./scripts/smoke-3070ti-8gb.sh ./MiniMax-H3 \
  "一隻紅狐狸在雪地緩慢行走，固定鏡頭。" \
  ./outputs/3070ti-smoke.mp4
```

Fixed characteristics:

```text
seed: 42
canvas: 288x160
frames: 22
steps: 4
layers: 35
reuse: 2
core reuse: 4
SSD streaming: on
token reduction: on
preview: off
```

Acceptance:

- no CUDA OOM;
- tracked CUDA peak remains at or below the clamped budget;
- host RSS stays within configured RAM plus reasonable process overhead;
- all video/audio values are finite;
- MP4 metadata and frame count are correct;
- a second run does not show unbounded VRAM or host-memory growth;
- final counters show actual upload/eviction activity.

## Gate 5 — balanced 8 GiB generation

Use 576×320 output with a 288×160 internal render, 22 frames, 12 steps and 40
layers. Record:

- model preparation time;
- generated INT8 RAM size;
- peak VRAM and host RSS;
- VRAM and RAM eviction counts;
- bytes uploaded from RAM and read through file fallback;
- per-DiT-forward time;
- VAE encode/decode time;
- final wall time and output quality.

Repeat with host cache sizes 32, 48 and 56 GiB to identify the point where NVMe
fallback stops dominating.

## Gate 6 — cross-backend semantics

Run the same prompt/seed on the pinned Metal build and CUDA build. Pixel identity
is not expected, but compare subject identity, composition, motion continuity,
audio synchronization, latent relative L2 and embedding similarity.

## Gate 7 — release readiness

Before removing the engineering-preview label:

- pass all pinned upstream operation fixtures;
- complete at least ten repeated 8 GiB generations without leak/OOM/non-finite
  output;
- validate offload with Compute Sanitizer;
- replace or validate the reference attention kernel against a tiled path;
- benchmark cuDNN/autotuned VAE convolution alternatives;
- publish exact GPU, driver, CUDA, prompt, seed, dimensions, frames, steps,
  layers, cache sizes and cold/warm storage state.
