# h3cspeed architecture

## Stable model/backend boundary

The upstream project exposes tensors and accelerator operations through
`h3_gpu.h`; model parsing, safetensors, schedules, token layout, CLI behavior
and media handling remain ordinary C. `h3cspeed` keeps that C ABI and places
CUDA C++ behind it.

```text
H3 CLI / public C API
        |
upstream C model + safetensors + schedulers + FFmpeg
        |
h3_gpu.h C ABI
        |
CUDA tensors, offload scheduler, cuBLAS and custom kernels
```

This follows the backend-owned buffer/device/event principle used by
`llama.cpp`/GGML without translating H3 into the llama graph or taking a
`llama.cpp` build dependency.

## CUDA execution objects

Each `h3_gpu` context owns:

- one non-blocking compute stream;
- one non-blocking upload stream;
- cuBLAS and cuBLASLt handles;
- a readiness event and last-compute-use event for every tensor;
- one reusable transfer staging region;
- one reusable device scratch region;
- a device-resident weight LRU and a system-RAM LRU.

A context has a single-host-thread enqueue contract. The upstream SSD reader may
refill a distinct slot on another thread, but general concurrent operation
submission into one context is not supported.

## v0.2 three-tier weight storage

Low-VRAM mode is automatic on devices with at most 10 GiB of VRAM and can also
be selected with `H3_CUDA_OFFLOAD=ram+file`.

### Tier 1: bounded VRAM

Every tracked `cudaMalloc` participates in `vram_budget_bytes`, including
weights, activations and scratch. Read-only weights additionally participate in
`weight_cache_bytes`.

Before an allocation, the scheduler evicts least-recently-used weights until
both limits are satisfied. A weight used by the currently queued operation has
its `pin_epoch` set and is not an eviction candidate. After the operation is
enqueued, a `last_use` event is recorded on the compute stream. Eviction waits
for both upload-ready and last-use events before calling `cudaFree`.

This event model restores safe overlap that a global compute synchronization
would otherwise destroy.

### Tier 2: system RAM

File-loaded BF16/F32 weights may keep an ordinary or pinned host copy. Generated
INT8 weights and their scales have no safetensors source, so model preparation
performs a device-to-host copy before those tensors are marked offloadable.
They are therefore recoverable after VRAM eviction.

The RAM cache is also LRU-managed. When it is full, only host copies with a
valid file source may be discarded. Generated INT8 copies are non-droppable.
Pinned copies are capped separately; tensors beyond that cap use ordinary
pageable RAM and transfer through the staging window.

### Tier 3: checkpoint/file cache

A tensor whose RAM copy was dropped keeps its safetensors path, byte offset and
length. On its next use, the runtime reads the range through the reusable
staging window and uploads it to the CUDA upload stream. The operating system's
page cache can still satisfy a warm read from system RAM.

This tier is also the fallback when `H3_CUDA_HOST_CACHE_MIB` is too small to
hold all original weights.

## Why explicit offload instead of managed-memory oversubscription

The implementation deliberately does not use `cudaMallocManaged` as a substitute
for capacity planning. Managed-memory oversubscription and migration behavior
differ across native Linux, Windows and WSL2. Explicit host/file backing gives
the engine a deterministic VRAM ceiling, clear failure messages and control
over pinned-memory usage.

## Scratch lifetime

Scratch is allocated through the same budget manager as tensors. Resizing first
synchronizes compute, releases the old buffer, evicts weights if needed and
allocates the new size. With `H3_CUDA_RELEASE_SCRATCH=1`, `h3_gpu_submit()`
releases scratch after stream completion, reducing phase-boundary peaks on 8 GiB
cards.

## Matrix multiplication

For ordinary BF16/FP32 tensors, row-major H3 matrices are presented to cuBLAS
using the transpose identity:

```text
C(row-major MxN) = A(row-major MxK) * W(row-major NxK)^T
C^T = W * A^T
```

BF16 products accumulate in FP32. Dynamic INT8 uses one scale per activation
row and one per weight output row, accumulates to INT32, then dequantizes in a
CUDA epilogue. On low-VRAM systems, the resulting INT8 weight and scale tensors
are persisted in RAM and enter the same VRAM LRU as file-backed weights.

## Attention

The reference attention implementation is a stable one-pass online-softmax
kernel. It never materializes a `[heads, sequence, sequence]` score tensor, so
memory is bounded by the query/head output accumulator. It is a correctness and
capacity baseline; a future tiled FlashAttention-style implementation can
replace it without changing the public C API.

## Portability substitutions

- `h3_metal_probe()` remains as an ABI symbol but reports CUDA device and memory.
- Apple Accelerate/vImage resize is replaced by portable Lanczos3 C.
- Foundation tokenizer code is replaced by ICU + yyjson C.
- Objective-C/Metal sources remain only in the pinned audit tree and are not
  compiled.
