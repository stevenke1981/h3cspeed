# AGENTS.md — h3cspeed

## Goal

Maintain a source-compatible NVIDIA CUDA backend for the pinned `antirez/h3.c`
revision while keeping the public C API, model directory and CLI stable. The
v0.2 capacity target includes RTX 3070 Ti 8 GiB with explicit system-RAM/file
offload.

## Non-negotiable rules

1. Do not change H3 tensor layout merely to simplify a CUDA kernel.
2. Keep all exported functions `extern "C"` and synchronized with `h3_gpu.h`.
3. Preserve documented BF16 rounding boundaries.
4. Do not silently fall back to CPU compute in performance paths.
5. Never report benchmark results without full hardware/command metadata.
6. Keep upstream source pinned and hash-verified.
7. Never evict a tensor before its upload-ready and last-use events complete.
8. Never mark a generated tensor offloadable without an authoritative RAM or
   file backing copy.
9. RAM LRU eviction may discard only data that has a valid reconstruction
   source.
10. Every new device allocation, including scratch, must honor the shared VRAM
    budget in low-VRAM mode.

## Required portable checks

```bash
python3 -m py_compile scripts/*.py
python3 scripts/verify_backend_api.py
python3 scripts/source_syntax_lint.py
cmake -S . -B build-overlay -DH3CSPEED_OVERLAY_TESTS_ONLY=ON
cmake --build build-overlay --parallel
ctest --test-dir build-overlay --output-on-failure
```

## Required CUDA checks

```bash
H3CSPEED_CUDA_ARCHITECTURES=86 ./scripts/build.sh
python3 scripts/verify_backend_api.py --header third_party/h3/h3_gpu.h
compute-sanitizer --tool memcheck ./build/<focused-test>
compute-sanitizer --tool racecheck ./build/<offload-test>
```

For memory-manager changes, test a deliberately tiny VRAM budget and verify
repeated file-backed and generated-INT8 evict/reload cycles.

## Acceptance criteria

- API coverage remains 103/103;
- focused tests pass on Ampere and one newer NVIDIA architecture;
- Compute Sanitizer reports no out-of-bounds, use-after-free or race;
- resident and offloaded execution produce the same numerical result;
- peak device/RAM temporary memory is documented;
- performance is no slower than the reference path for the measured shape, or
  the change is explicitly labeled correctness/capacity-only.
