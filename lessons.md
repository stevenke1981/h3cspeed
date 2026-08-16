---
## Lesson #1 — 2026-08-16
**Trigger:** Edit tool / format-on-save rewrote `scripts/run_h3_quantized_60s.py` and broke `prior.get("sidecar_sha256")` substring tests.
**Rule:** For this repo's Python files that use exact-source substring assertions, apply surgical patches with a standalone script and verify `git diff --stat` stays at the intended line count before running CTest.
**Source:** host-cache promote / quantized runner defaults
---
## Lesson #2 — 2026-08-16
**Trigger:** 240p DiT wall was 97% eviction/file-read; model root is on ST4000DM004 HDD at 118 MiB/s.
**Rule:** Before another kernel or VRAM-cache knob, measure the model drive MediaType and a multi-GiB sequential read of the real DiT safetensors. Host-cache defaults without `H3_CUDA_DIT_PREFETCH=1` are 60% of *available* RAM and cannot hold the 19.53 GiB ConvRot pack on a 32 GiB host.
**Source:** h3cspeed speed investigation
---
