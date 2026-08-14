# PERF-007 480p digital-human talking benchmark

This benchmark uses the Codex-generated reference image in
`benchmarks/digital-human-480p/digital-human-864x480.png` and the adjacent
prompt fixture. The fixed output contract is 864x480, 24 fps, 124 frames
(5.166667 seconds), two denoise steps, 50 DiT layers, reuse 1, core-reuse 1,
seed 42, SageAttention and TF32 disabled. H3's 5-second frame count is aligned
to `5 + 17n`; 124 frames is the first aligned value at or above five seconds.

The bound-host measurements below used Windows 11, an RTX 3070 Ti (sm_86),
CUDA 13.2, MSVC/VS18 and the Release `build-perf007/h3cspeed.exe` binary
SHA-256 `E292221FABC521B59FFC2798B62F65EE47E82D3D0C6258EC4A468D4AF5738573`.
The 576x320 sidecar was SHA-256
`BB0D5CB45713A2430F505477182564CA39069B5BCCEDD31BEB462D958E5B9671`;
the tracked fixture and prompt hashes are recorded below. Model generation
times exclude sidecar creation, process startup, FFmpeg mux and QA.

## ComfyUI producer

Use `scripts/perf002_comfy_trace.py --frames 124` with the four bound model
files and private output/runtime directories. The producer runs the real
ComfyUI graph, not a synthetic media probe. The result must include the
dual-clock/native-flow scheduler trace, Sage hit/fallback trace, H.264/AAC
media, full FFmpeg decode and non-silent audio evidence.

## H3 producer

`scripts/run_perf007_h3.ps1` creates or reuses the bound FL2VA sidecar, runs
the native binary and emits a private JSON timing record. The default is a
native internal 864x480 render. For an explicitly labelled speed/quality
candidate, set `-RenderWidth` and `-RenderHeight` to same-aspect 32-pixel
multiples no larger than 864x480; the output remains 864x480 but the internal
canvas is smaller and must not be compared as native-quality parity.

The one-ahead ConvRot route is opt-in. On an 8 GiB card, a full future block
can exceed the shared weight-cache headroom, so the runner exposes
`-PrefetchMaxWeights` (1--12; default 8). It maps to the exact
`H3_CUDA_DIT_PREFETCH_MAX_WEIGHTS` environment value. The historical default
of 12 remains unchanged when the variable is absent. Any prefetch failure is
fail-closed; do not relabel a fallback run as a successful candidate.

## Acceptance and interpretation

Compare model-generation wall time separately from startup, sidecar creation,
FFmpeg mux and QA. Both outputs still require codec/dimension/frame-count,
full video/audio decode, non-silent audio and sampled-frame visual review.
Native H3 and reduced-internal-render H3 are different quality points. A
reduced render may be reported as a speed/capacity candidate, never as proof
that native H3 has matched ComfyUI quality or throughput.

## 2026-08-15 bound-host observation

The Codex image fixture SHA-256 is
`192e1191d2f597819c0bac96267e5827be2005d4c4d94141ff59099adbdf24ae`.
ComfyUI completed the real 124-frame graph in 333.50 seconds of prompt
execution. Its MP4 is 276,681 bytes with SHA-256
`2c88b11840b9e820c25081e0d42d4fccc59e3aac6cb3816b6f166c5b3a21aa0b`;
FFprobe reported H.264 864x480/124 at 24 fps and AAC 32 kHz stereo, full
FFmpeg decode exited 0, and audio was non-silent (mean -27.6 dB, max -10.0
dB). Sampled frames showed a clear, stable presenter with changing mouth
shapes.

The exact-native H3 run (864x480 internal, 50 layers, reuse 1, core-reuse 1)
was stopped after a 3,600-second bound-host timeout while still in denoise
step 2 of 2; no media was published, so it is `TIMEOUT`, not `PASS`.

The capacity-aware one-ahead candidate completed real media with the same
output contract. At 288x160 internal render it took 366.712963 seconds, but
visual QA failed because the face and mouth had severe deformation and
horizontal seams. At 576x320 internal render it took 1,391.846375 seconds
(core-reuse 1) and 1,381.070846 seconds (core-reuse 4); both produced the same
MP4 SHA-256
`802feb3da6a14b66d9c2432db098cb86055a05671bc65f991d5ef38e37d4e025`.
That output passed H.264/AAC, 864x480/124-frame, full decode and non-silent
audio checks; sampled frames were usable but softer than ComfyUI with a mild
horizontal seam. The best 576x320 candidate is therefore about 4.14x slower
than ComfyUI's generation time, while the 288x160 candidate is a speed-only
point and is not an acceptable digital-human quality baseline.

The new `H3_CUDA_DIT_PREFETCH_MAX_WEIGHTS` limit is the safe optimization
validated here: limiting the future ConvRot batch to eight weights avoids the
1.78 GiB single-allocation failure seen with a full 12-weight reservation on
the 8 GiB card. It improves capacity/completion, not enough throughput to
close the native-quality gap. A real native H3 speedup still requires a
matched DiT kernel/offload optimization and must be re-measured with the same
media QA gates.
