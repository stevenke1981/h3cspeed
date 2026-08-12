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
- `H3_CUDA_RELEASE_SCRATCH=0|1`
- `H3_CUDA_OFFLOAD_VERBOSE=1`
- `H3_CUDA_DEVICE=N`
- `H3_CUDA_TF32=1` opt-in numerical/performance trade-off
- `H3_PROFILE=1` print memory traffic and dispatch counters

## Sizing the RAM cache

The automatic host cache is 60% of system RAM currently available at context
creation, capped at 64 GiB and further clamped to leave at least 2 GiB free.
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
