# CUDA tuning and low-VRAM controls

## RTX 3070 Ti 8 GiB baseline

Compile Ampere `sm_86`:

```bash
H3CSPEED_CUDA_ARCHITECTURES=86 ./scripts/build.sh
```

Use the packaged profile through:

```bash
./scripts/run-3070ti-8gb.sh [normal h3cspeed arguments]
```

The wrapper defaults are intentionally conservative:

```text
CUDA allocation budget       5888 MiB
resident weight LRU           1536 MiB
pinned host-copy cap           128 MiB
transfer staging                64 MiB
scratch release at submit       on
SSD streaming                   on
middle-block token reduction    off (unless explicitly requested)
DiT layers                      50 (upstream default)
requested frames                 22
default output              256x256 (when all dimensions are omitted)
reuse / core reuse              1 / 1 (upstream defaults)
```

The runtime clamps the requested CUDA budget to memory actually free when the
context is created, so desktop/display use cannot turn the nominal setting into
an immediate allocation failure.

## Environment variables

- `H3_CUDA_OFFLOAD=auto|ram+file|off`
- `H3_CUDA_LOW_VRAM=1` compatibility switch that selects RAM offload
- `H3_CUDA_VRAM_BUDGET_MIB=N`
- `H3_CUDA_WEIGHT_CACHE_MIB=N`
- `H3_CUDA_HOST_CACHE_MIB=N`
- `H3_CUDA_PINNED_HOST_MIB=N`
- `H3_CUDA_STAGING_MIB=N`
- `H3_CUDA_ASYNC_REFILL=1` opt in to two event-fenced refill slots inside the
  existing pinned staging allocation
- `H3_CUDA_REFILL_TRACE=1` diagnostic-only, bounded private per-chunk host-read
  and H2D timing for the async refill path
- `H3_CUDA_DIT_PREFETCH=1` enables one-ahead reserve/upload for the released
  non-SSD ConvRot INT8 DiT schedule. The next active block is reserved while
  the current block remains scheduler-owned; only after current compute is
  enqueued is that reserved block uploaded on the private upload stream.
  Future weights remain unpinned but scheduler-protected from eviction until
  the next block consumes them or the reservation is cancelled. BF16 SSD
  streaming deliberately keeps its existing slot/thread schedule and is
  outside this first integration slice.
- `H3_CUDA_UPLOAD_WAIT_TRACE=1` requires profiling and records DiT-scoped
  upload-ready waits plus PERF-006 route/counter metadata. The exact-value
  opt-in lazily allocates a 4,096-entry timing-event ring only for the
  `dit_denoise` context. Initialization failure or overflow invalidates the
  performance gate; reports must show `upload_wait_trace_complete=true`,
  `upload_wait_trace_overflow=false` and `upload_wait_trace_union_valid=true`.
  This diagnostic mode is not intended for normal serving.
- `H3_CUDA_RELEASE_SCRATCH=0|1`
- `H3_CUDA_OFFLOAD_VERBOSE=1`
- `H3_CUDA_DEVICE=N`
- `H3_CUDA_TF32=1` opt-in numerical/performance trade-off
- `H3_PROFILE=1` print memory traffic and dispatch counters
- `H3_PROFILE_JSON_DIR=PATH` enable profiling and atomically write one
  machine-readable JSON report per CUDA context to an existing directory

`H3_CUDA_ASYNC_REFILL` is deliberately disabled by default. When it is exactly
`1` and the staging allocation is pinned, the runtime splits the configured
staging window into two equal slots. A slot is reused only after its recorded
H2D completion event finishes; the tensor upload-ready event is still recorded
after all chunks have been enqueued. If pinned staging or either slot event is
unavailable, the runtime keeps the original single-buffer synchronous path.
This switch does not increase the configured staging allocation and does not by
itself prove end-to-end DiT overlap or a video wall-time improvement.

When RAM/file offload is active, an evicted weight allocation enters a bounded
same-size device-allocation reuse pool only after its upload-ready and last-use
events have completed. The pool is capped at eight entries and at one third of
the configured weight cache, with a 512 MiB hard ceiling; pooled bytes remain
in device_live_bytes and are released before context teardown, so the shared
VRAM budget remains authoritative. A reuse hit avoids the driver cudaFree and
cudaMalloc pair but does not remove logical LRU eviction or weight upload.
The stderr memory summary reports weight-reuse hits/stores as a churn
diagnostic; it is not an end-to-end speedup claim.

The PERF-006 upload-wait trace measures the device interval associated with
the compute stream waiting for a weight's upload-ready event. It does not
relabel compute, file-read, eviction or final-drain time as upload-ready wait.
Use `scripts/validate_perf006_ab.py` on a same-binary baseline/candidate pair.
It emits `OBSERVED_WAIT_PASS` when the numerical threshold and parity contract
hold; this does not become the formal fixed cold-cache gate until cache/order
metadata and counterbalanced trials satisfy the benchmark plan. It is also not
an end-to-end speedup if process wall rises.

`H3_CUDA_REFILL_TRACE` is also exact-value opt-in and only becomes active when
the two-slot path is active. It keeps at most 64 private entries, records no
paths or tensor labels, and uses timing-enabled CUDA events immediately around
each DMA enqueue. The host-read timestamps use a separate monotonic CPU clock;
they establish ordering and nesting, not a cross-clock duration comparison.
Because event creation and collection add diagnostic overhead, leave this flag
unset outside focused validation. Trace allocation failure is fail-closed when
the flag was explicitly requested.

## Machine-readable profile reports

Set both variables when collecting performance evidence:

```bash
mkdir -p /tmp/h3-profile
H3_PROFILE=1 H3_PROFILE_JSON_DIR=/tmp/h3-profile ./build/h3cspeed ...
python3 scripts/validate_profile_report.py /tmp/h3-profile/*.json
```

The version-1 report keeps the public `h3_gpu_stats` ABI unchanged. It separates
file reads, pageable staging copies, H2D enqueue time, compute/upload stream
host waits, event waits, allocations, and capacity-LRU versus phase-retire and
error-cleanup frees. Device event durations and host waits may overlap and are
explicitly marked non-additive. The current context report exposes an
instrumented-host-to-context-wall diagnostic ratio, but deliberately emits
`coverage_gate_valid=false` and `coverage_gate_met=false`: overlapping worker
activity, VAE stitching, FFmpeg, and whole-process cold-start phases require the
later benchmark driver before the plan's 95% critical-path gate can be evaluated.

Only built-in phase labels are admitted to filenames and stderr; arbitrary
caller labels are redacted. JSON label content is always redacted, so reports
do not contain prompts or model paths even if a caller misuses the label API.
The output directory must already exist and be a stable, private directory
controlled by the invoking user. Directory path stability is the caller's
responsibility; the writer redacts report content and atomically publishes each
final filename without replacing an existing report.

## Sizing the RAM cache

The automatic host cache is 60% of system RAM currently available at context
creation, capped at 64 GiB and further clamped to leave at least 2 GiB free.
When the opt-in `H3_CUDA_DIT_PREFETCH=1` path is enabled, the default rises to
85% of currently available RAM because one-ahead DiT prefetch benefits from
retaining more file-backed weights; the same 2 GiB headroom clamp still wins.
An explicit `H3_CUDA_HOST_CACHE_MIB` value always overrides the percentage.
Under WSL2, available memory is measured inside the WSL VM. Leave enough RAM for
the Linux kernel, page cache, FFmpeg, CPU-side model data and Windows itself.

Suggested starting points:

| Physical RAM | WSL assignment | `H3_CUDA_HOST_CACHE_MIB` |
|---:|---:|---:|
| 64 GiB | 48–52 GiB | 28672–32768 |
| 96 GiB | 72–80 GiB | 49152–57344 |
| 128 GiB | 96 GiB | 65536 cap |

When the host cache fills, file-backed BF16/F32 copies are discarded in LRU
order and later reread from safetensors. Generated INT8 weights are retained,
so insufficient RAM will produce a configuration error during preparation.

`--ssd-streaming` uses two reusable BF16 host slots (about 1.49 GiB for this
model) and overwrites them while walking the DiT layers. Raising
`H3_CUDA_HOST_CACHE_MIB` does not turn those slots into a persistent per-layer
cache and therefore cannot eliminate the direct SSD rereads.

## Choosing render size and frames

H3 aligns requested frames to `5 + 17n`, but the released generation path
requires at least one trained 22-frame decoder chunk. For 8 GiB validation:

1. start with 256×256, 22 frames, 4 steps and all 50 layers;
2. for the fast-quality preset, retain a 288×160 internal render while scaling
   output to 864×480;
3. request five seconds only after the 22-frame run stays below the configured budget;
4. keep `--show` disabled until generation succeeds reliably;
5. increase render size before increasing frame count only when still-image
detail matters more than motion duration.

The fast-quality preset requests five seconds at 24 fps. H3 aligns the request
to its trained `5 + 17n` frame sequence, so the current runtime emits 124 frames
(about 5.17 seconds):

```bash
./scripts/fast-quality-3070ti-8gb.sh /path/to/MiniMax-H3
```

It keeps 20 denoising steps and all 50 layers, disables token reduction, and
uses `--core-reuse 4` to reduce repeated core evaluation. This is a deliberate
quality/performance trade-off and is not numerically identical to
`--core-reuse 1`.

## Diagnosing failures

`CUDA offload allocation` means non-offloadable activations/scratch or the
currently pinned working set cannot fit. Reduce internal render dimensions,
frames or the number of active layers before raising the budget.

`generated weight RAM offload` means the RAM cache cannot retain a generated
INT8 tensor. Assign more memory to WSL2 or raise `H3_CUDA_HOST_CACHE_MIB`.

Frequent `file-fallback` traffic in the final profile line means the RAM cache
is too small for the hot working set or the run is intentionally SSD-bound.

## Performance expectations

Offload is a capacity path, not a speed path. PCIe uploads and possible NVMe
reads occur when an evicted layer becomes hot. Increasing the VRAM weight cache
can reduce transfers, but never set the total budget so high that display,
CUDA context or temporary libraries have no headroom.
