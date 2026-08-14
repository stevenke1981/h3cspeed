# h3cspeed local validation results

- Generated: 2026-08-14 (Asia/Taipei)
- Platform: native Windows x64
- GPU: NVIDIA GeForce RTX 3070 Ti, CUDA `sm_86`, 8.00 GiB VRAM
- System RAM: 31.81 GiB
- Toolchain: CUDA Toolkit 13.2 / nvcc 13.2.78, Visual Studio Build Tools 2026,
  CMake 4.3.2, Ninja 1.13.2, ICU 76.1
- Driver: NVIDIA 596.36
- Build: `scripts/build-native.ps1 -BuildDirectory build-quant -CudaArchitectures 86 -BuildType Release`

## PASS

- Pinned upstream archive and prepared-tree hash verification, including a
  fresh forced bootstrap and post-patch verification.
- Python bytecode checks and all focused Python suites (POSIX-only launcher
  cases skip on native Windows).
- Public CUDA backend API coverage: 103/103.
- Windows Clang host syntax checks for all CUDA sources and portable C sources.
- Portable overlay configure/build and CTest: 11/11.
- Native CUDA/MSVC build and link: `h3cspeed.exe`,
  `h3cspeed-cuda-info.exe`, library, and focused tests.
- Native CTest baseline before PERF-001 registration: 17/17. This includes CUDA numeric-layout, ConvRot, NVFP4,
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
- A real quantized FL2VA I2V + opt-in SageAttention 256x256, 22-frame, 4-step
  smoke exited 0 after native INT8 DiT, Audio VAE, Video VAE and FFmpeg. The
  70,295-byte H.264/AAC artifact has SHA-256
  `82CFCCB00E430D86E95B1042D4D9618A33E54990D24FC73652F02C107FFC89CA`;
  full decode exited 0, all 22 frames decoded, and first/middle/final visual
  inspection showed a coherent red fox walking through snow without patch
  noise. Audio decoded to 59,392 interleaved float samples with -38.75 dB RMS
  and -18.86 dB peak. DiT telemetry recorded 1,803.98 MiB peak device,
  1,522.56 MiB peak resident and 12,582.55 MiB peak host memory, 73.33 GiB
  uploads, 70.40 GiB evictions and 22.70 GiB file fallback. This four-step run
  is a pipeline/semantic smoke, not the 20-step I2V quality baseline.
- A second real FL2VA I2V + SageAttention wrapper smoke used the released
  864x480 output and 288x160 internal-render geometry, 22 frames, two steps,
  all 50 DiT layers and seed 4243 on the same RTX 3070 Ti / CUDA 13.2 host.
  The wrapper passed its canonical first-frame path into the native process and
  wrote a 484,093-byte H.264/AAC artifact with SHA-256
  `A1EC7B50FD09199C22B2001BA21FCF2BE6A6FA869076DDB98FC05220636EE464`.
  FFmpeg full decode exited 0; all 22 864x480 frames and 32 kHz stereo audio
  decoded, and first/middle/final inspection preserved the preceding snowy-lake
  fox subject without patch noise. Audio measured -54.69 dB RMS and -37.59 dB
  peak. This remains a two-step continuity/geometry smoke, not a 20-step I2V
  quality result or a completed sixty-second run.
- A complete resumable quantized FL2VA I2V + SageAttention acceptance used the
  packaged Windows `a372afb` runtime on the same RTX 3070 Ti / driver 596.36 /
  CUDA 13.2 host. It generated twelve independently verified 864x480 clips,
  each 124 frames, 20 steps, all 50 DiT layers, reuse 1, core-reuse 4 and a
  288x160 internal render. Segment 01 was T2V; segments 02-12 used the previous
  decoded final frame through the v2 prompt-bound FL2VA sidecar. The first
  invocation stopped after verified segment 02 and the second resumed from the
  content-bound state without regenerating those clips. Combined wall time for
  both invocations was about 3 h 20 min 53 s, including the initial and resume
  model/runtime/Comfy environment hashes, conditioning, offload wait, VAE,
  per-segment media checks, final encode and final deep verification. Observed
  per-segment DiT denoise was 45.369-49.085 s; representative DiT telemetry
  recorded up to 2,644.69 MiB peak device, 1,522.56 MiB peak resident and
  13,465.08 MiB peak host memory, with 109.25 GiB uploads and up to 34.05 GiB
  file fallback. Qwen image-aware conditioning used about 6.32 GiB process
  VRAM at the observed high point.
- The final 28,398,378-byte artifact has SHA-256
  `06ADEA7A78FF6E866F137DF0C8176C9181D09399457E20E429DD1B48AD3DDD06`.
  Independent root verification found H.264/yuv420p 864x480, 24/1 CFR, exactly
  1,440 decoded frames and 60.000000 s; AAC 32 kHz stereo decoded to exactly
  1,920,000 float samples per channel. FFmpeg `-xerror` full decode exited 0.
  Audio measured -45.54 dB RMS and -21.91 dB peak. First/middle/final and all
  eleven boundary pairs showed a recognizable fox and snowy environment, but
  also visible softness, ghosting, color blocking and generative boundary cuts.
  This is a capacity/resume/exact-media PASS, while visual quality at the
  288x160 internal render remains experimental rather than production-ready.
- The focused SageAttention benchmark used Windows 10 build 19045, RTX 3070 Ti
  8 GiB (`sm_86`), driver 596.36, CUDA 13.2 / nvcc 13.2.78, VS Build Tools
  18.6.0 (MSVC 14.51), Release architecture 86, B=1/H=56/N=800/D=128, two
  warmups and ten iterations. Command:
  `$env:H3_CUDA_ATTENTION='native'; .\build-quant\bench_cuda_attention.exe`.
  Native measured 104.806 ms and Sage 97.644 ms (1.073x), MAE 5.75368e-6,
  maximum absolute error 0.000488281 and cosine 0.999999. The measured device
  peak was 55.03 MiB, host cache was 0 MiB, and the five benchmark BF16 host
  arrays total 54.69 MiB. Compute Sanitizer on the final CUDA numeric test
  reported memcheck 0 errors and racecheck 0 hazards.
- The focused same-latent video-VAE parity gate (plan P1A / PERF-004) used the
  same RTX 3070 Ti / driver 596.36 / CUDA 13.2.78 / MSVC 14.51 host, a Release
  `sm_86` `build-perf` tree at `2b155b7` plus the new harness, and the FL2VA
  F16 video-VAE source under
  `E:\minimax-h3\ComfyUI\models\h3_t2v_quantized\base\FL2VA\video_vae\source`.
  `build-perf\test_cuda_vae_layer_major_parity.exe` decodes one deterministic
  xorshift64 latent twice in-process, tile-major first and then with
  `H3_VAE_LAYER_MAJOR=1`, with the test applying the standard offload profile
  (`H3_CUDA_LOW_VRAM=1`, `ram+file`, 5,888 MiB VRAM budget, 1,536 MiB weight
  cache, 128 MiB pinned, 64 MiB staging) and `H3_PROFILE=1` for path markers.
  At latent 7x16x32 (22 frames, 256x512, tiles 2x1 at 288 pixels, 2 states)
  the decode was bit-for-bit identical: 0/8,650,752 pixel mismatches, max-abs
  0, rel-L2 0; weight uploads fell from 18.06 GiB tile-major to 9.03 GiB
  layer-major (-50.0%, the two-state minimum) and device peak rose from
  1,908.34 MiB to 1,939.91 MiB (+2 x 15.79 MiB hidden pool). At latent
  7x30x54 (22 frames, 480x864, tiles 4x2 at 272 pixels, 8 states) the decode
  was again bit-for-bit identical: 0/27,371,520 pixel mismatches, max-abs 0,
  rel-L2 0; weight uploads fell from 72.23 GiB to 9.03 GiB (-87.5%), passing
  the P1A 22-frame smoke gate of at least 80% fewer upload bytes. Device peak
  rose from 1,909.68 MiB to 2,036.43 MiB (+8 x 15.84 MiB hidden pool), and
  both traversals dispatched identical work (1,176 linear, 288 SDPA). Total
  harness wall time was 55.4 s and 167.5 s respectively, each covering both
  decodes plus two cold weight loads, so per-path wall attribution and the
  124-frame <40 GiB / >=3x wall gates remain pending PERF-001
  instrumentation; the tile-major device peak matches the accepted 864x480
  production baseline exactly.
- The real 864x480/22-frame/2-step h3cspeed layer-major candidate preserved the
  exact baseline media SHA-256
  `54077780f5d45cfcd9d5b44b0fea91cea9c4fc15dceecf31cd60d376c9795f5b`.
  Video VAE traffic fell from 3,528 uploads / 72.23 GiB and 79.78 GiB
  evictions to 441 uploads / 9.03 GiB and 16.58 GiB evictions; VAE device peak
  rose from 1,909.68 MiB to 2,036.43 MiB. This is a PASS for the 22-frame
  layer-major traffic and media-parity candidate, not a full PERF-004 PASS.
  The candidate VAE profile wall was 107.4532 s with only 86.145% accounted
  coverage (`coverage=false`), so the >=3x VAE-wall gate is `NOT_RUN` pending a
  matched baseline profile. The 124-frame <40 GiB target and PERF-001 95%
  wall-accounting/disabled-overhead gates remain `NOT_RUN`/`NOT_MET`.
- The PERF-001 machine-readable profiler smoke used Windows 10 build 19045,
  RTX 3070 Ti 8 GiB (`sm_86`), driver 596.36, CUDA 13.2.78, MSVC
  19.44.35227, a Release architecture-86 `build-quant`, and
  `H3_PROFILE=1` plus a unique `H3_PROFILE_JSON_DIR`. Running
  `build-quant\test_cuda_scale_add.exe` exited 0 and emitted a complete
  schema-v1 per-context JSON report; `scripts/validate_profile_report.py`
  parsed it successfully. The report separated three H2D enqueues (80 bytes),
  four allocations, compute/upload stream waits, device event time, and four
  intentional phase-retire frees (112 bytes), while capacity-LRU and error
  cleanup stayed zero. The tiny smoke intentionally reports only 0.04% host
  accounting coverage because CUDA context startup dominates it; it is a
  producer/schema/runtime PASS, not the plan's 95% long-phase coverage or
  profiling-disabled <2% overhead PASS. The full Windows CUDA CTest run passed
  18 tests with the model-dependent layer-major VAE test explicitly skipped;
  the portable overlay passed 13/13 and API coverage remained 103/103.
  Compute Sanitizer on the same focused scale-add test reported memcheck 0
  errors and racecheck 0 hazards, errors or warnings.

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

# PERF-002A/B matched benchmark walking skeleton (2026-08-14)

- Immutable input manifest: PASS in portable contract tests. The creator
  hashes the actual bound regular files, reference PNG and private prompt;
  canonical output contains labels and SHA-256 values only and refuses
  overwrite, symlink/reparse inputs and unmatched algorithm-parity schedules.
- Synthetic 864x480/124-frame media harness: PASS with real ffprobe, full
  ffmpeg video/audio decode, 32 kHz stereo non-silent PCM and five PNG hashes.
- ComfyUI/h3cspeed 22-frame runtime adapters: PASS on the bound host; see the
  PERF-002C smoke evidence below.
- Matched 864x480/124-frame/8-step cold plus three-warm A/B: NOT RUN.
- Scheduler sigma/raw-audio evidence and actual Sage hit/fallback traces:
  PASS for the two isolated 22-frame smokes. No matched performance,
  quality-parity or speedup claim is made.

## PERF-002C isolated 22-frame adapter (2026-08-14)

- Synthetic child-process adapter: PASS. It generated 864x480 H.264/AAC media
  with 22 frames at 24 fps, emitted exact dual-clock 12/3 scheduler evidence
  and a Sage trace with hits and zero fallbacks, then passed full media/audio
  decode and five-frame hashing.
- Pre/post model, conditioning, runtime, prompt and reference bindings: PASS in
  the synthetic process test; no absolute path, raw prompt, argv or environment
  is present in the result.
- Output/source/ComfyUI/model isolation and Sage-fallback rejection: PASS in
  focused tests.
- Real h3cspeed 22-frame/2-step smoke: SMOKE_PASS on the bound host.
- Real ComfyUI 22-frame/2-step smoke: SMOKE_PASS on the bound host.
- Matched A/B timing and quality conclusion: NOT RUN.

## PERF-002C bound-host smoke evidence (2026-08-14)

- Immutable input manifest SHA-256:
  `49bb685df675dcf265fe87c5b95fc6587c7a42e6737dad427325be8de3cdf264`.
- h3cspeed: `SMOKE_PASS`, wall `688.2388647 s`, media SHA-256
  `54077780f5d45cfcd9d5b44b0fea91cea9c4fc15dceecf31cd60d376c9795f5b`,
  816,597 bytes, duration 0.925 s, `864x480/22f/24fps`, H.264/AAC stereo,
  non-silent audio and full decode. The runtime trace recorded Sage hits 102,
  unexpected fallbacks 0, DiT denoise 116.592 s, and Video VAE upload traffic
  72.23 GiB.
- ComfyUI: `SMOKE_PASS`, wall `397.98855 s`, media SHA-256
  `bb8dc8697c68a3f3a55d47038f138541bea467ffd54b7e828dacece5c9bbc6b8`,
  386,003 bytes, duration 0.917 s, `864x480/22f/24fps`, H.264/AAC stereo,
  non-silent audio and full decode. The runtime trace recorded Sage hits 100,
  unexpected fallbacks 0, and prompt execution 307.74 s.
- Five extracted frames from each output were inspected and were clear/full
  width; this is visual smoke evidence only, not a 124-frame quality result.
- These wall times are isolated smoke timings with different engine startup,
  conditioning and cache behavior. They are not a fair speed ranking. The
  matched 124-frame/8-step one-cold-plus-three-warm A/B remains `NOT RUN`, and
  PERF-001 95% wall-accounting/disabled-overhead gates remain `NOT RUN`.

## PERF-004 layer-major bound-host candidate (2026-08-14)

- The candidate used the same immutable manifest, prompt, first frame, seed and
  864x480/22-frame/2-step h3cspeed contract as the tile-major baseline, with
  `H3_VAE_LAYER_MAJOR=1`. Its final media SHA-256 was exactly the baseline
  `54077780f5d45cfcd9d5b44b0fea91cea9c4fc15dceecf31cd60d376c9795f5b`.
- Video VAE traffic changed from 3,528 uploads / 72.23 GiB and 79.78 GiB
  evictions to 441 uploads / 9.03 GiB and 16.58 GiB evictions. Device peak
  changed from 1,909.68 MiB to 2,036.43 MiB. The 22-frame traffic reduction
  and exact media parity are `PASS` for this candidate.
- Candidate VAE profile wall was 107.4532 s, but accounted coverage was only
  86.145% with `coverage=false`; no >=3x VAE-wall claim is accepted. Full
  process wall moved from 688.2389 s to 677.5203 s (-1.56%) in this single
  baseline/candidate comparison; it is not the required cold-plus-three-warm
  matched A/B.
- PERF-004 >=3x VAE wall is `NOT_RUN`; the 124-frame <40 GiB target is
  `NOT_RUN`; PERF-001 95% wall accounting and profiling-disabled overhead are
  `NOT_RUN`/`NOT_MET`. Compute-wait, file-read and eviction optimization must
  not be inferred from the traffic win alone.

## PERF-002C h3 runtime evidence producer (2026-08-14)

- Private opt-in h3cspeed trace producer: PASS in Windows MSVC `/W4 /WX`
  portable tests. It serializes the actual `h3_serving_schedule_build` arrays,
  12/3 shifts, generation geometry and seed only after every requested audio
  Euler step and the complete denoise succeed.
- CUDA attention evidence now distinguishes real Sage hits in `dit_bf16`,
  explicit native-control calls and unexpected Sage-ineligible BF16 fallbacks.
  F32 VAE attention is outside this scope.
- Trace publication rejects one-sided/relative/concurrent/existing targets,
  uses exclusive temporary files and no-clobber publication, and aborts state
  on generation cleanup. Overlay CTest passed 17/17; API remains 103/103.
- The real h3cspeed and ComfyUI 22-frame/2-step bound-host smokes now pass;
  this producer change still does not establish matched throughput, quality
  parity or a speedup.

## PERF-002C ComfyUI producer/startup preflight (2026-08-14)

- The repo-owned one-shot ComfyUI producer and adapter contract passed 31
  focused Python tests with 2 environment-dependent skips; the portable
  overlay passed 18/18. Python compilation, C/CUDA source syntax, pinned
  upstream verification and API coverage 103/103 also passed.
- A real no-inference startup preflight used the bound
  `E:\minimax-h3\.venv` environment, RTX 3070 Ti, PyTorch 2.11.0+cu130 and
  the external ComfyUI checkout. The private loopback server became ready in
  108.6 seconds with isolated input/output/temp/user/database paths, Sage
  selected, and only `comfyui-minimax-h3-audio-T8` whitelisted; the other
  installed H3 custom nodes were explicitly skipped. The process exited 0.
- The startup emitted an asyncio pending-accept cleanup warning while the
  one-shot process was closing. The producer therefore remains process-bound;
  reusable in-process lifecycle support is not claimed.
- No model was loaded and no graph was queued in this historical startup
  preflight. The subsequent bound-host smoke loaded the model and produced
  real media/traces; full source and Python-environment closure, native
  control, and matched A/B timing remain `NOT RUN`.

## PERF-002C h3 direct-binary binding preflight (2026-08-14)

- The h3 adapter now fail-closes unless the direct binary consumes the bound
  FL2VA model root, exact prompt bytes, immutable first frame, v2 conditioning
  sidecar, Qwen SHA-256, output path and both runtime trace paths. It rejects
  duplicate or unknown options, last-frame/Ref2VA input, internal render-size
  overrides and any geometry other than 864x480, 22 frames and 2 steps for the
  bounded smoke. It also requires Sage with TF32 disabled, rejects unbound
  positional arguments, validates the sidecar's v2 metadata and closes the
  native loader inventory over the exact four weights and four required
  configs. The sidecar is the single h3 conditioning artifact; native
  decode/mux FFmpeg and QA FFmpeg/FFprobe are separately path/hash-bound. The
  bound synthetic executable E2E and mismatch cases passed.
- `prepare_h3_quantized_model.py --validate-only` passed for four safetensors
  payloads and 26 config/tokenizer files. A schema-2
  `minimax-h3-comfy-fl2va-quantized-pack` root was prepared with hard links;
  `h3cspeed --info` identified FL2VA INT8 ConvRot, Qwen NVFP4/AWQ, video VAE
  and audio VAE from that root without loading the model for inference.
- A native 864x480 Codex ImageGen fixture was generated and mechanically
  normalized to exact RGB24 864x480. Its SHA-256 is
  `210A7F41E2B0030388030AFF170296FE8BC9F5D464FD43F0BA8104404A0D66D4`;
  it has full-width content rather than the former blurred side-fill layout.
- This preflight was recorded before the user process was released. After a
  fresh GPU gate, the current sidecar-backed h3 and ComfyUI smokes both
  reached `SMOKE_PASS`; an individual `SMOKE_PASS` is still not matched A/B
  or a speed/quality result. The full hashes and timings are recorded in the
  bound-host smoke evidence above.
