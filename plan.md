# h3cspeed MiniMax H3 performance plan

Status: **implementation in progress**

Last updated: 2026-08-14

## 1. Objective

Make the quantized MiniMax H3 FL2VA path on an RTX 3070 Ti 8 GiB approach
ComfyUI's matched 5-second 480p throughput, then exceed it where a fixed native
execution graph can remove framework and dispatch overhead. The work must keep
the current public C API, model directory, tensor layouts, BF16 rounding
boundaries, low-VRAM safety, and RAM/file reconstruction guarantees.

The first target is not another attention micro-optimization. It is to stop the
large amount of synchronous weight traffic that dominates the measured wall
time.

## 2. User outcome and scope

### In scope

- Quantized FL2VA T2V and first/last-keyframe I2V on NVIDIA CUDA.
- RTX 3070 Ti 8 GiB as the capacity/performance development target.
- A matched ComfyUI A/B at 864x480, 124 frames, 8 steps, 50 layers, the same
  model files, prompt, reference image, seed and SageAttention policy.
- Video VAE tile/chunk scheduling, DiT weight streaming, RAM/device caches,
  asynchronous prefetch, kernel selection/fusion, fixed allocations and CUDA
  Graph capture.
- Instrumentation that separates compute, upload, file read, allocation,
  synchronization, decode and mux wall time.

### Out of scope for the initial optimization series

- Changing H3 tensor layouts or numerical contracts to make a kernel easier.
- Silently using CPU compute.
- Ref2VA support, model conversion changes or quality/scheduler redesign.
- Claiming a ComfyUI win from unmatched workflows or old logs.
- Requiring more than 8 GiB VRAM for the supported low-VRAM profile.

## 3. Protected baseline

The optimization series must preserve:

- API coverage `103/103`.
- ConvRot INT8 QKV/MLP semantics and Comfy-contiguous QKV decoding.
- NVFP4/AWQ BF16 rounding parity.
- FL2VA sidecar prompt/model/keyframe binding and fail-closed validation.
- Every eviction waiting for upload-ready and last-use events.
- Every offloadable tensor retaining an authoritative RAM or file source.
- Resident and RAM/file execution producing the same numerical result within
  the existing focused-test tolerances.
- Full video/audio decoding and visible-frame acceptance.

## 4. Measured starting point

The current native baseline is the completed first-frame I2V run at
`E:\h3cspeed-8step-5s-864x480`.

| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3070 Ti, `sm_86`, 8 GiB |
| Driver | 596.36 |
| Host memory | 31.81 GiB |
| Runtime | CUDA 13.2 portable binary; PyTorch sidecar uses 2.11.0+cu130 |
| Model | FL2VA ConvRot INT8 + Qwen NVFP4/AWQ + FP16-packed/F32-runtime video VAE + FP32 audio VAE |
| Shape | internal/output 864x480, 124 frames, 24 fps |
| Sampling | 8 steps, 50 layers, reuse 1, core reuse 1, seed 42 |
| Attention | Sage; TF32 disabled |
| Offload | RAM + file, 5,888 MiB device budget, 1,536 MiB weight cache |
| Total native wall time | 17,511.210 s |
| DiT CUDA Euler time | 2,058.947 s |
| DiT traffic | 145.16 GiB upload, 143.24 GiB eviction, 45.39 GiB file fallback |
| Video VAE traffic | 505.59 GiB upload, 513.14 GiB eviction |
| DiT peaks | 5,860.81 MiB device, 1,522.56 MiB resident weights, 12,687.60 MiB host |
| Video VAE peaks | 1,909.68 MiB device, 9,245.04 MiB host |
| Output acceptance | exit 0; H.264 864x480/124 frames; AAC stereo; full decode exit 0 |

This proves native 480p capacity, not competitive throughput. The old ComfyUI
logs contain faster jobs, but they are not a matched A/B and therefore are not
the performance baseline for this plan.

## 5. Current bottleneck model

### 5.1 Video VAE: P0 traffic amplification

`third_party/h3/h3_video_vae.c` builds a resident decoder, loads all 36
transformer blocks once as tensor objects, then executes the complete decoder
for every temporal-chunk/spatial-tile pair. At 124 frames and the measured 4x2
tile plan, the traversal is:

```text
7 temporal chunks x 8 spatial tiles x 36 blocks = 2,016 block-attention calls
```

That exactly matches the recorded `sdpa=2016`. The VAE weights are resident as
host/file-backed tensor objects, but approximately 9 GiB of decoder weights do
not fit in a 1.5 GiB device cache. The tile-major loop consequently reloads
weights across tile/chunk executions and records about 505 GiB of uploads.

### 5.2 DiT: one model scan per step plus synchronous refill stalls

The 8-step DiT records `sdpa=402`, consistent with 50 blocks x 8 steps plus
refiner work, and 145.16 GiB of uploads, close to repeated full-model scans.
This traffic is expected under an 8 GiB device limit; the long idle component is
not. `upload_weight_locked()` currently serializes refill through one staging
buffer and synchronizes the upload stream after each staging chunk. File read,
host copy, H2D transfer and compute therefore have little opportunity to
overlap.

### 5.3 Measurement gap

CUDA event timing reports compute-stream time, while total wall time also
contains file reads, staged copies, stream waits, allocations, eviction waits,
VAE unpack/stitch and FFmpeg. Per-cause counters/timers are required before an
optimization is credited.

## 6. Execution phases

Each phase is a separate reviewable change set. A phase advances only after its
focused numerical, memory-safety and performance gates pass.

### Phase P0 - reproducible profiler and matched A/B harness

Implementation status:

- `PERF-001` private CUDA/offload profile schema and per-context JSON writer:
  implemented in the current change set; real-GPU coverage and disabled-overhead
  gates remain to be measured.
- The matched ComfyUI A/B driver and end-to-end stage timers remain pending.

Deliverables:

1. Extend `h3_gpu_stats` without breaking existing callers, or add a versioned
   profiling report, with cumulative wall seconds for:
   - file-backed reads;
   - pageable-to-staging copies;
   - H2D enqueue and H2D wait;
   - device allocation/eviction waits;
   - compute-stream waits and kernel time;
   - VAE unpack/stitch and FFmpeg.
2. Record bytes and operation counts by phase and, for profiling builds, by
   tensor/block class. Split capacity-LRU eviction from intentional
   phase-retire/free and error cleanup; current aggregate eviction counters mix
   those reasons. Do not log private prompts or model paths by default.
3. Add a machine-readable JSON report next to the human summary.
4. Create a matched benchmark driver for ComfyUI and h3cspeed using one
   Codex ImageGen-generated, native 16:9 864x480 reference PNG and an immutable
   run manifest. Normalize its metadata once, record its SHA-256 and never count
   image-generation time in either engine. Do not use the old centered
   480x480 image with blurred side fill as the quality fixture.

Verification:

- Critical-path spans or exclusive CPU-wait buckets account for at least 95%
  of measured phase wall time. Overlapping file-read, H2D and compute durations
  are reported separately and are never added as if mutually exclusive.
- Profiling disabled changes 22-frame wall time by less than 2%.
- The manifest records binary/source commit, model hashes, GPU UUID, driver,
  toolkit, scheduler, seed, dimensions, frames, steps, layers, attention, TF32,
  cache budgets and output hashes.
- ComfyUI and h3cspeed media both pass ffprobe, full `ffmpeg -xerror` decode,
  audio decode and five-frame visual QA before timing is compared.

### Phase P1 - collapse Video VAE weight traffic

Implementation status:

- The opt-in F32 layer-major walking skeleton and same-latent parity harness are
  implemented. The real 22-frame 864x480 candidate reduced Video VAE traffic
  from 3,528 uploads / 72.23 GiB and 79.78 GiB evictions to 441 uploads /
  9.03 GiB and 16.58 GiB evictions, with device peak changing from 1,909.68
  MiB to 2,036.43 MiB. Its final media SHA was exactly unchanged from the
  tile-major baseline, so the 22-frame traffic/media-parity gate is PASS.
- Candidate VAE profile wall was 107.4532 s with only 86.145% accounted
  coverage (`coverage=false`); the `>=3x` VAE-wall gate is therefore NOT_RUN,
  not PASS. The 124-frame `<40 GiB` traffic gate and PERF-001 telemetry gates
  remain pending.

The model stores the Video VAE weights as F16, but the current loader expands
them to approximately 9.2 GiB of F32 device-ready data. This makes a nominally
"resident" decoder churn through the 1.5 GiB weight cache for every tile/chunk
state. The first implementation preserves the existing F32 numerical path and
changes traversal only; compact F16 compute is a separate experiment.

#### P1A - F32 layer-major hidden-state pool (preferred walking skeleton)

For the measured 124-frame run there are 56 tile/chunk states. Each persistent
state needs only the block-to-block F32 `hidden` tensor (about 15.84 MiB), not a
complete approximately 388 MiB transient activation set. A 56-state hidden pool
is therefore about 0.866 GiB. Reuse one block-local transient workspace while
executing states sequentially:

1. Execute:

   ```text
   prepare the 56 hidden-only states
   load decoder block N once
     process each hidden state using one shared transient workspace
   retire block N after its last state
   apply output projection and unpack/stitch each indexed state
   ```

2. Keep input/output projection and frame stitching outside the 36-block loop
   only where the current per-state operation order and overlap blending remain
   numerically equivalent. Do not batch SDPA in the walking skeleton.
3. Give every hidden state explicit ownership and deterministic tile/chunk
   indexing; cleanup must neither leak nor double-free the pool.
4. Keep all hidden states on device when the measured shared budget permits.
   If grouping is required, `ceil(56/B) * 9 GiB` is the approximate weight
   traffic: use at least `B=8` for the 100 GiB gate and target `B>=16` for the
   40 GiB gate.
5. Spill only persistent hidden tensors, never full transient activations. Any
   asynchronous D2H spill needs its own completion event, generation and
   authoritative backing before the tensor becomes offloadable.
6. Keep the tile-major F32 path behind an oracle/rollback flag and add a
   reproducible same-latent VAE parity harness; final-video inspection alone is
   not a numerical test.

P1A gates:

- 864x480/22-frame VAE smoke: at least 80% fewer weight-upload bytes than the
  current traversal.
- 864x480/124-frame target: VAE upload bytes below 40 GiB; approximately
  9-12 GiB is expected if all 56 hidden states fit.
- VAE wall time improves at least 3x on the 3070 Ti baseline before the phase is
  accepted.
- Same-latent tile-major/layer-major reports cover 22- and 124-frame shapes,
  finite counts, per-state/frame max absolute and relative L2, tile stitching
  and temporal overlap; full media decode also passes.
- Report hidden-pool peak device bytes, transient workspace, host/file spill,
  activation H2D/D2H, weight uploads, capacity-only evictions and intentional
  retire/free separately.

Fresh-build requirement:

`third_party/h3` is a bootstrap product. Any VAE source change must be carried
through the hash-verified `scripts/upstream_overlay`/bootstrap mechanism and its
fresh-tree tests; editing only the ignored prepared source is invalid.

#### P1B - compact F16 VAE weights (independent follow-up)

Explore the approximately 4.5 GiB packed F16 weights only after P1A establishes
the F32 oracle and traffic win. This requires an explicit private-storage versus
public-API dtype decision: adding F16 to the public `h3_gpu_dtype` enum is an API
change and is not done implicitly. It also requires a real cuBLASLt mixed-type
or end-to-end F16/BF16 activation path; F16 weights with F32 activations must not
silently fall into a slower scalar mixed-matmul kernel.

P1B gates independently record per-op kernel time, max absolute/relative L2,
non-finites, upload traffic, scratch/live-device peak and visible media quality.
Keep it only if it beats accepted P1A wall time without changing layout,
rounding contracts, API stability or the shared VRAM budget.

### Phase P2 - asynchronous DiT refill pipeline

Deliverables:

1. Separate refill into read, host staging, H2D upload and ready-event stages.
2. Add a fail-closed per-tensor state machine such as
   `UNLOADED -> READING -> UPLOADING -> READY -> IN_USE`, plus `ERROR`, with an
   upload ticket and explicit staging-slot ownership.
3. Size two or more staging slots only when the shared pinned-host/RAM budgets
   permit them. Two 64 MiB slots already consume the current 128 MiB pinned
   budget. A pageable fallback is allowed for correctness but must report that
   asynchronous overlap was not established.
4. Enforce this event chain:

   ```text
   read -> slot owned -> H2D -> upload_done recorded
        -> compute waits upload_done -> last_use recorded
        -> staging slot/tensor storage may be reused or evicted
   ```

5. Add explicit next-block prefetch from the known 50-block schedule without
   marking all prefetched tensors as current-epoch pinned. Future reservations
   remain bounded, cancellable and scheduler-protected until consumption.
6. Remove per-chunk `cudaStreamSynchronize(upload_stream)` only after each slot
   has a completion fence; a source/staging buffer cannot be overwritten while
   its DMA is pending.
7. Preserve both upload-ready and last-use checks before any eviction, source
   overwrite or host free. Any read/upload failure transitions to `ERROR` and
   aborts rather than consuming partial data.
8. Coalesce source ranges only when they are validated, contiguous ranges from
   the same file and destination allocation. Each tensor retains independent
   ready/error state and an authoritative reconstruction source; generated INT8
   backing is never folded into file-backed ranges.

Gates:

- Focused two-block test includes a real file-backed read plus pinned H2D and
  compute dependency, and proves non-zero overlap on an Nsight Systems timeline
  or equivalent event trace. Pageable fallback is reported separately.
- No upload slot is overwritten before its ready/last-use event completes.
- Host source bytes remain unchanged and live while DMA is pending.
- Deliberately tiny VRAM tests repeatedly evict/reload file-backed and generated
  INT8 tensors without corruption.
- Resident/offloaded numeric parity and focused memcheck/racecheck pass.
- Exclusive DiT upload-wait wall time decreases at least 50% on a fixed
  cold-cache 22-frame/2-step 480p fixture; compute waits are not relabelled as
  upload waits.
- Performance target after P0 calibration: full 124-frame/8-step DiT wall time
  approaches 1.5x reported compute time; stretch target is 1.2x. Missing this
  target does not waive correctness/event gates and is recorded rather than
  hidden.

### Phase P3 - phase lifecycle and cache autotuning

Deliverables:

- Release Qwen completely after sidecar creation; release DiT before Video VAE;
  verify no phase holds stale GPU/RAM cache entries.
- Add a bounded cache sweep tool for VRAM budget, resident-weight budget, host
  cache, staging slots and scratch release policy.
- Store an opt-in `sm_86/8GB/32GB-host` measured profile and a separate
  `sm_86/8GB/64GB+-host` profile. Never infer the latter from the former.
- Prefer sequential/mapped model reads and retain reusable source bytes in RAM
  when host headroom permits; SSD is a reconstruction source, not the steady
  hot path.

Gates:

- No OOM or system-memory exhaustion across ten 22-frame runs.
- Auto-selected settings keep at least the documented OS/RAM headroom.
- A larger host-RAM result is labelled with its actual hardware and is not used
  as the 31.81 GiB baseline.

### Phase P4 - tuned CUDA kernels and fusion

Only begin after P1/P2 show that transfer no longer dominates.

Candidate changes, each independently benchmarked:

- cuBLASLt algorithm autotuning/cache for fixed ConvRot INT8 and BF16 shapes.
- SageAttention shape specializations for H3 video/audio sequences.
- Fused AdaLN + ConvRot activation transform/quantization.
- Fused QKV projection preparation + RoPE where rounding/layout contracts allow.
- Fused SwiGLU and down-projection/residual epilogues.
- cuDNN or tuned convolution paths for Video VAE operations not dominated by
  transformer blocks.

Gates:

- Correctness first: all focused numeric tests and BF16 boundaries pass.
- New scratch allocations obey the shared low-VRAM budget.
- A kernel is kept only if it is no slower for its recorded shape, or is clearly
  labelled correctness/capacity-only.
- Compute Sanitizer memcheck and racecheck are clean on focused tests.

### Phase P5 - fixed allocation plan and CUDA Graph capture

Deliverables:

- Preallocate/reuse activation and scratch arenas for fixed shapes.
- Remove hot-path `cudaMalloc`/`cudaFree` and avoid release-scratch-on-every-
  submit when a safe reusable arena fits.
- Capture stable per-block or per-step graphs only after asynchronous prefetch
  addresses/events are stable.
- Keep a non-graph fallback for diagnostics and unsupported shapes.

Gates:

- Graph and non-graph paths are numerically equivalent.
- Dispatch/submit CPU time decreases measurably.
- No graph captures a buffer that can be evicted or overwritten before replay.

### Phase P6 - matched ComfyUI parity and release gate

Run both engines from a clean state with the same:

- generated native 864x480 reference PNG;
- prompt, seed 42, FL2VA/Qwen/VAE hashes;
- 864x480 internal/output geometry, 124 frames, 8 steps and 50 layers;
- SageAttention and TF32 policy;
- cold-cache and warm-cache trials, each reported separately.

The conditioning contract is specifically first-frame FL2VA I2V: ComfyUI uses
`MiniMaxH3AudioConditioningT8(task_type=i2va, audio_mode=native)` with one
first frame, no Ref2VA reference and no LoRA. h3cspeed uses the same Comfy CUDA
Qwen conditioning through the sidecar plus its native keyframe VAE encode. The
manifest binds prompt bytes, token IDs/tags, Qwen and VAE model hashes,
geometry/frame count and the canonical first-frame SHA-256.

Benchmark tracks:

1. **Algorithm parity (primary):** both sides use
   `dual_clock_euler` + `native_flow`, video shift 12 and audio shift 3. Dump
   and compare the video/audio sigma arrays (`max_abs_diff <= 1e-6`) and verify
   the raw-audio velocity/update protocol before this track can support an
   engine speed or output-parity claim.
2. **Engine recommended (secondary):** ComfyUI may use
   `res_multistep/simple` while h3cspeed uses its serving Euler path. Report
   wall throughput and independent media quality, but label it
   apples-to-oranges and do not claim trajectory parity.
3. **Future scheduler parity:** if `res_multistep` is implemented natively, it
   gets its own numeric/quality ticket and matched A/B. It is not implicitly
   part of this optimization series.

Sage is also split into a native-attention parity track and an optimized Sage
track. The report must prove that ComfyUI actually loaded its SageAttention
package and that h3cspeed selected `H3_CUDA_ATTENTION=sage`; the two
implementations are not assumed bit-equivalent. Backend-hit and fallback
counters are mandatory; any Comfy `attention_pytorch` or native fallback marks
the Sage track `NOT PASS`.

For each track, separately report cold end-to-end time (Qwen/sidecar, model
load, keyframe encode, DiT, VAE decode and mux) and warmed steady-state compute.
A pre-generated sidecar or warm model cache on only one side invalidates the
comparison.

Preserve the raw timing shape at 124 video frames / 24 fps (approximately
5.166667 seconds). H3's 40 Hz audio grid may produce approximately 5.175
seconds; both engines keep the same raw overhang and use the same FFmpeg mux
policy rather than trimming only one side.

Release thresholds:

- **Approach ComfyUI:** median h3cspeed wall time <= 1.20x matched ComfyUI.
- **Match ComfyUI:** median h3cspeed wall time <= 1.05x matched ComfyUI.
- **Exceed ComfyUI:** median h3cspeed wall time < 0.95x matched ComfyUI.
- Use at least three warm trials after one cold trial; report median and range.
- Both outputs must pass media/audio decode and human review for subject
  fidelity, blur, ghosting, color blocks and temporal stability.
- Repeat focused CUDA checks on Ampere and one newer NVIDIA architecture before
  a general performance claim.
- Produce and inspect a Windows `sm_86` runtime archive and a Linux CUDA runtime
  archive. Cross-compilation is build evidence only; missing Linux/newer-GPU
  execution is recorded as `NOT RUN`/`MANUAL_REQUIRED`, never `PASS`.

## 7. Benchmark ladder

Use the smallest shape that exercises the target behavior, then move upward:

1. Synthetic tensor/upload tests: seconds, no model.
2. One/two-block DiT and one-block VAE tests: event and numeric validation.
3. 256x256, 22 frames, 2 steps: fast regression.
4. 864x480, 22 frames, 2 steps: spatial tile/offload gate.
5. 864x480, 124 frames, 8 steps: user-visible matched A/B.
6. Ten repeated short/medium runs: leak, cache and recovery gate.

Do not run the five-hour baseline after every small change. Full A/B is a phase
gate after focused evidence is green.

## 8. Required verification per implementation change

Portable checks:

```powershell
Get-ChildItem scripts -Filter *.py | ForEach-Object {
  python -m py_compile $_.FullName
}
python scripts/verify_backend_api.py
python scripts/source_syntax_lint.py
cmake -S . -B build-overlay -DH3CSPEED_OVERLAY_TESTS_ONLY=ON
cmake --build build-overlay --parallel
ctest --test-dir build-overlay --output-on-failure
```

CUDA changes additionally require:

```text
H3CSPEED_CUDA_ARCHITECTURES=86 build
focused numeric test
tiny-budget repeated RAM/file eviction test
Compute Sanitizer memcheck
Compute Sanitizer racecheck for offload/event changes
resident versus RAM/file numerical comparison
```

Every benchmark record must include the complete hardware, driver, toolkit,
build, command, environment, model hashes, peak device/RAM temporary memory,
traffic counters and media QA evidence.

## 9. Implementation tickets and order

| ID | Change | Depends on | Acceptance evidence |
| --- | --- | --- | --- |
| PERF-001 | Wall/I/O/H2D/wait profiler + JSON report | none | >=95% wall accounting, <2% disabled overhead |
| PERF-002 | Reproducible ComfyUI/h3cspeed A/B manifest/driver | PERF-001 | 002A/B manifest/media + 002C isolated adapter and h3 runtime trace producer implemented; bound-host 22-frame smokes `SMOKE_PASS`; matched A/B `NOT_RUN` |
| PERF-003 | F32 hidden-pool state-size spike + tile/layer parity harness | PERF-001 | ownership/memory model + same-latent 22-frame proof |
| PERF-004 | Layer-major F32 VAE execution via verified overlay | PERF-003 | 22f traffic/media parity PASS; 124f <40 GiB and >=3x VAE wall `NOT_RUN` |
| PERF-005 | Double-buffered async refill primitives | PERF-001 | opt-in two-slot/file-backed stress + strict H2D/compute timeline + sanitizer PASS; generated INT8 and 22f wall gates pending |
| PERF-006 | DiT next-block prefetch schedule | PERF-005 | private primitive + released ConvRot integration + real 22f media parity PASS; observed same-binary wait reduction 99.18%; fixed cold-cache gate `NOT_RUN`; wall speedup NOT_MET |
| PERF-007 | Phase cache lifecycle + 8GB cache sweep | PERF-004/006 | no stale phase entries, ten-run stability |
| PERF-008 | Shape-specific kernel autotuning/fusion | PERF-004/006 | numeric/sanitizer/per-shape speed gate |
| PERF-009 | Fixed arenas + CUDA Graph | PERF-007/008 | graph parity and dispatch reduction |
| PERF-010 | Full matched A/B on Ampere + newer GPU | all | thresholds in P6 |
| PERF-011 | Optional compact F16 VAE compute experiment | PERF-004 | faster than P1A plus independent numeric/API/budget gates |

## 10. Rollback and change discipline

- Keep every optimization behind a temporary opt-in environment flag until its
  correctness and performance gates pass.
- Preserve the current path as the oracle until the replacement becomes the
  accepted baseline.
- One performance concern per commit; do not mix scheduler, model format and
  memory-manager changes.
- If a phase misses its traffic or wall-time gate, record the measurement and
  revert/disable it rather than stacking more speculative changes.
- No benchmark claim is published from configuration alone. `PASS` requires a
  real run; unavailable newer-architecture evidence remains `MANUAL_REQUIRED`.

## 11. Immediate next change

Continue **PERF-002/PERF-004** from the real bound-host smoke and layer-major
candidate evidence. The immutable manifest SHA is
`49bb685df675dcf265fe87c5b95fc6587c7a42e6737dad427325be8de3cdf264`.
The tile-major h3cspeed baseline was 688.2388647 s; the layer-major candidate
was 677.5203 s (-1.56%) and produced the exact same media SHA. Video VAE
traffic fell from 3,528 uploads / 72.23 GiB and 79.78 GiB evictions to 441
uploads / 9.03 GiB and 16.58 GiB evictions; device peak rose from 1,909.68 MiB
to 2,036.43 MiB. This proves the 22-frame traffic/media-parity win, but not a
VAE wall-speedup claim: candidate VAE profile wall was 107.4532 s with only
86.145% accounted coverage (`coverage=false`).

The PERF-005 primitive now has an opt-in, two-slot pinned refill path and a
bounded private timeline. The focused 512 MiB test measured 16 exact 2 MiB DMA
chunks with 0.332352 ms strict H2D/compute overlap; host reads were nested in
the active compute window, the full fixture hash stayed unchanged, parallel
disabled/enabled CTest passed, and memcheck/racecheck were clean. This proves
the primitive overlaps DMA with independent compute for the focused fixture,
not a model speedup or a cross-clock CPU-read duration. The PERF-006
reserve/upload primitive is now independently proven. The released non-SSD
ConvRot INT8 schedule now reserves a protected one-ahead batch before current
compute and uploads it afterward, leaves future weights unpinned until
consumption, and always
drains compute and upload streams on submit/failure. BF16 SSD streaming is
unchanged and deferred. A direct real 864x480/22-frame candidate exercised all
98 one-ahead transitions and produced a byte-identical oracle MP4, closing the
released-route correctness and media-parity gate. A new DiT-scoped,
fail-closed upload-ready trace then measured a same-binary 22-frame pair:
baseline 0.917230380 s versus candidate 0.007519424 s, a 99.1802% reduction,
with identical media SHA. This pair emitted `OBSERVED_WAIT_PASS`; it does not
close the formal fixed cold-cache gate because Windows filesystem cache state
was possibly warm and the order was baseline then candidate. Candidate wall
was 686.814827 s versus 678.499696 s (+1.2255%), so an end-to-end speedup is
`NOT_MET` and the formal >=50% gate remains `NOT_RUN`.

The reverse-order counterbalanced pair is now complete for the 240p-class
profile. H3-then-ComfyUI measured 855.971099 s versus 435.675537 s, while
ComfyUI-then-H3 measured 853.185943 s versus 452.711302 s; the means are
854.578521 s and 444.193419 s (1.9239x). Both orders produced identical
media and passed the same decode/audio/Sage checks, but the result remains
`OBSERVED_ONLY`: the Windows filesystem cache was not flushed and the producer
clocks have different scopes. The runner's `--reverse-order` changes process
order only; it is not a formal cold-cache gate.

Two H3-only follow-ups did not justify another capacity knob: a 1,962 MiB
weight-cache probe measured 669.961096 s engine wall, 502.521261 s DiT wall,
364.713215 s eviction time and 1,146 LRU evictions; a four-weight prefetch
probe at the 1,536 MiB default measured 668.469394 s, 501.029463 s,
363.038060 s and 1,160 evictions. Both retained the same media hash and
40.1667 GB upload traffic. No default cache or prefetch policy was changed.
The next source slice is therefore eviction/file-read serialization in the DiT
working set, with generated-INT8 cycling and pageable-allocation failure
injection still focused test debt. Only after the wall regression is explained
or removed should the complete matched 864x480/124-frame/8-step matrix run
with one cold plus three warm trials per engine, including the 124-frame
<40 GiB and PERF-001 95% gates.

The h3cspeed producer records actual serving sigma arrays and `dit_bf16` Sage
hit/expected-native/unexpected-fallback counters through opt-in, no-clobber
trace files. The Comfy producer starts the bound checkout in an isolated
in-process child, queues the fixed first-frame I2VA graph, captures actual T8
sampler/raw-audio/Sage calls, and publishes only the real `SaveVideo` output.
The h3 direct-binary contract binds the v2 FL2VA sidecar, canonical first
frame, Qwen hash, exact native FL2VA inventory, FFmpeg tools and trace paths.
The Comfy producer still binds a bounded entry-point file set rather than the
full runtime import closure or Python-environment lock; those closure hashes,
native-control comparison, matched A/B timing, and quality review remain
pending. `expected_native_calls: 0` in the Comfy attention trace is an
explicit no-control-run marker, not native-control evidence.

The matched 124-frame/8-step cold-plus-three-warm A/B and profiling-disabled
overhead gate remain `NOT_RUN`. The candidate's PERF-001 critical-path
accounting is `NOT_MET` at 86.145% (`coverage=false`), so none of these gates
may be inferred from the isolated smoke or candidate media parity.

## 12. PERF-007 480p digital-human benchmark observation

The current user-facing comparison uses the Codex-generated 864x480 reference
image, the fixed prompt fixture, native FL2VA I2V sidecars and 124 frames at
24 fps (approximately 5.166667 seconds). ComfyUI completed the real graph in
333.50 seconds of prompt execution and passed media/audio/full-decode QA. The
exact-native h3cspeed run was stopped at a 3,600-second bound-host timeout in
denoise step 2/2, so no native-quality H3 speed claim is made.

An opt-in H3 candidate adds a bounded
`H3_CUDA_DIT_PREFETCH_MAX_WEIGHTS` policy and an explicit lower internal render
size while preserving the 864x480 output container. The 576x320 candidate
completed in 1,381.070846 seconds with usable but softer sampled frames; the
288x160 candidate completed in 366.712963 seconds but failed visual quality
because of facial deformation and horizontal seams. These are capacity/speed
observations, not native-quality parity.

The first DiT-side optimization is now implemented and measured. The
one-ahead scheduler reserves the future ConvRot block only after the current
block has enqueued its work, avoiding premature LRU eviction of current
weights. The opt-in prefetch route also defaults its automatic host cache to
85% of available RAM (with the existing 64 GiB cap, 2 GiB headroom and
explicit override). On the same 288x160-internal contract, the new run took
306.180956 s versus 366.712963 s previously, with identical media SHA and
full media/audio decode. This is a 16.51% speed improvement for the
speed-only preset and remains below the native-quality evidence bar.

The next performance slice must preserve the 576x320-or-native visual gate:
run a matched 576x320 source-optimized pair first, then revisit exact-native
864x480. Do not infer native H3/ComfyUI quality parity from the low-resolution
candidate.

The first DiT offload-churn slice is now implemented in the CUDA allocator.
After both upload-ready and last-use fences complete, an evicted offload
weight may enter a bounded same-size allocation reuse pool (eight entries, at
most one third of the configured weight cache, and no more than 512 MiB).
Pooled bytes stay inside the shared device-live budget; logical LRU eviction
and weight-upload counters are unchanged, so this targets cudaFree/cudaMalloc
churn rather than hiding reload traffic.

A final short sequential probe on the RTX 3070 Ti (driver 596.36, CUDA 13.2,
sm86) used the 448x256/22-frame/24-fps/2-step/50-layer/seed42 matrix contract,
ram+file offload, 5.75 GiB VRAM budget, 1.5 GiB weight cache, async refill,
one-ahead ConvRot prefetch8, layer-major VAE, Sage, and TF32 off. The final
binary SHA-256 was DFBBDA51F7F0FD65277B8176DDDDE0C0AE75B1AD6FDB788E9CA3414ED7E9BA7B;
the FL2VA pack manifest SHA-256 was 724650671BAB418E77C994AB2D6BE3BDA4189B760C9FA44B9B9631917FECE6FE;
the exact 240p reference, prompt, and sidecar SHA-256 values were
B12993DF2EB7B153A0BA9498D6E26A797D9EC5722CB1C187273691FC18AB86DF,
75E7B21F4D7B7EDC602C29F1B03C5C5E633DB3BD8A95E4D176BE54B794EB0AA4, and
794D0EDB4E928A86A06CE0F98A9C61A80A283204F774C1226912084B2747D7BA.
Compared with the pre-pool binary on the same short contract, DiT eviction
time fell from 36.440367 s to 3.236380 s and allocation time from 0.109522 s
to 0.076803 s; uploads stayed 1,235 / 37.41 GiB and logical evictions stayed
1,160. The final run reported reuse 290/1,337 with a 257.41 MiB pool.
End-to-end wall time was 253.881752 s versus 250.256931 s, so this is a
measured churn/capacity improvement, not a formal speedup claim; fixed
cold-cache counterbalanced A/B, 480p/720p, and 124-frame gates remain
NOT_RUN.

## 13. Native resolution matrix (240p/480p/720p)

The resolution matrix is a separate 2-step throughput track. It uses native
H3-grid profiles 448x256, 864x480, and 1280x704, with exact per-profile PNG
references, `render == output`, 124 frames, 24 fps, 50 layers, and seed 42.
The paired dry-run/execute contract is implemented in
`scripts/run_resolution_matrix.py`; H3 commands omit render-dimension
overrides so native execution defaults to `render == output`, and the runner
is fail-closed against resize flags,
protected output roots, stale plans, child-process timeout descendants, and
unverified Windows private-output ACLs.
An explicit `--execute` also publishes per-profile H3/Comfy wall-time ratios
as `OBSERVED_ONLY`; media, quality, and counterbalanced speed gates remain
separate acceptance steps.

The first real 240p/22-frame smoke passed media/audio/visual/Sage checks. The
whole first-run command measured 498.1 s for h3cspeed (including sidecar
generation) and 447.9 s for ComfyUI; this is an observation, not a matched
cold-cache speed claim. H3's engine-reported work was 248.93 s and ComfyUI's
prompt execution was 345.38 s, but those clocks have different scope. The H3
MP4 was larger because its encoder uses CRF18/AAC192k while ComfyUI preserves
its own SaveVideo defaults.

A formal 240p-class five-second run then reused the validated sidecar in a
fresh private output tree. Both engines produced valid 448x256/124-frame/
24-fps H.264/AAC media with full decode, non-silent audio, Sage/no-fallback,
and manual frame-62 visual checks passing. The H3 engine reported 664.741099 s
and the ComfyUI prompt reported 345.26 s; this sequential, non-counterbalanced
pair is an observed ~1.93x H3 slowdown, not the fixed cold-cache A/B gate.
The H3 profile shows the cause: 496.836 s DiT, 359.261 s eviction, 124.051 s
file reads, 1,160 evictions, and 40.26 GiB uploaded. A fresh 1,962 MiB
weight-cache candidate was effectively unchanged (664.960 s total; 497.997 s
DiT, 359.869 s eviction, 123.235 s file reads, 1,146 evictions), so the
default cache was not changed based on one run.

The cached-sidecar profile was then repeated in both process orders on
2026-08-16. H3-then-ComfyUI measured 855.971099 s versus 435.675537 s;
ComfyUI-then-H3 measured 853.185943 s versus 452.711302 s. The means were
854.578521 s and 444.193419 s (1.9239x). Both orders produced identical
H3/ComfyUI media hashes and passed full media/audio/Sage checks, but this is
`OBSERVED_ONLY`: Windows filesystem cache state was not flushed and the
producer clocks have different scope. A four-weight prefetch probe also
preserved 40.1667 GiB upload traffic and did not materially reduce DiT
eviction time. The next source change must target eviction/file-read
serialization rather than another unvalidated cache-size knob. The 480p and
720p real profiles remain `NOT_RUN`.
