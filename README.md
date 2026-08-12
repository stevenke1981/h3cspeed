# h3cspeed

Traditional Chinese quick start: [`README.zh-TW.md`](README.zh-TW.md)

`h3cspeed` is a native Windows/Linux/WSL2 NVIDIA CUDA port overlay for
[`antirez/h3.c`](https://github.com/antirez/h3.c). It preserves the released
MiniMax-H3 model directory, safetensor parser, multimodal orchestration, FFmpeg
pipeline, public C API and CLI while replacing Apple Metal/MPSGraph with CUDA.
The backend boundary follows the useful device/buffer/event separation used by
`llama.cpp`/GGML; `llama.cpp` is a design reference, not a dependency.

## v0.2.0: RTX 3070 Ti 8 GiB and system-RAM offload

The v0.2 backend adds an explicit three-tier weight memory manager:

```text
hot tensors                 NVIDIA VRAM LRU cache
      ↕ asynchronous upload / readiness + last-use events
cold tensors                system-RAM cache
      ↕ evict only reproducible file-backed copies
checkpoint fallback         safetensors on SSD / OS page cache
```

Important properties:

- cards with 10 GiB VRAM or less automatically enable low-VRAM mode;
- a hard CUDA allocation budget covers weights, activations and scratch memory;
- file-loaded BF16/F32 weights are evictable from VRAM;
- generated INT8 weights and scales are copied to system RAM before becoming
  evictable, so they are never lost when their VRAM entry is recycled;
- the VRAM weight cache is LRU-managed and protected by per-tensor last-use
  events, preventing an in-flight kernel from reading freed memory;
- the system-RAM cache has its own LRU. When RAM pressure rises, only tensors
  that can be reconstructed from safetensors are discarded to the file tier;
- pinned host memory is deliberately capped and large transfers use a reusable
  staging buffer, which is safer on WSL2 than relying on managed-memory
  oversubscription;
- scratch allocations participate in the same VRAM budget and can be released
  after every submission.

This is explicit offload, not `cudaMallocManaged`. That choice is intentional:
Linux Unified Memory oversubscription is not equally available on Windows/WSL2,
where full managed-memory support and pinned-memory behavior remain more
restricted.

## Current status

This archive is an engineering preview. The pinned `h3_gpu.h` surface remains
fully covered at 103/103 functions. Portable tests, policy tests, source syntax,
metadata checks and deterministic packaging are included. Native Windows CUDA
13.2 compilation, focused kernel execution and the complete text-to-video
pipeline have run on an RTX 3070 Ti 8 GiB with the pinned FL2VA model. The
BF16-Qwen baseline produced a semantically matching red fox in a snowy pine
forest at 256x256, 22 frames and 20 steps, and its H.264/AAC output passed full
decode, frame-count, non-silent-audio and visual inspection. The quantized
four-file pack has a separate acceptance path described below: direct native
BF16-Qwen decoding remains experimental, while the hybrid Comfy-conditioned /
native INT8 DiT route is the usable T2V path.

## Requirements

Recommended RTX 3070 Ti host:

- Windows 11 native, Ubuntu 22.04/24.04, or Windows 11 with WSL2 Ubuntu;
- RTX 3070 Ti 8 GiB, CUDA compute capability 8.6;
- CUDA Toolkit and driver visible through `nvcc` and `nvidia-smi`;
- 64 GiB system RAM minimum for a constrained run, 96 GiB recommended;
- fast NVMe storage for the checkpoint and file fallback;
- CMake 3.25+, Ninja, GCC/G++, Python 3, ICU and FFmpeg.

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake ninja-build pkg-config python3 git \
  libicu-dev ffmpeg
```

For WSL2, allocate enough VM memory in `%UserProfile%/.wslconfig`, for example:

```ini
[wsl2]
memory=80GB
swap=16GB
```

Restart WSL after changing it:

```powershell
wsl --shutdown
```

## Build for RTX 3070 Ti

Native Windows PowerShell:

```powershell
.\scripts\build-native.ps1 -BuildDirectory build-native -CudaArchitectures 86
```

The script discovers Visual Studio Build Tools and CUDA, verifies the pinned
upstream, provisions the pinned ICU runtime when needed, and places the runtime
DLLs beside the executables.

```bash
cd h3cspeed
H3CSPEED_CUDA_ARCHITECTURES=86 ./scripts/build.sh
```

The bootstrap step downloads and verifies this exact upstream revision:

```text
antirez/h3.c
8974cc055ea9c02fcd14cc27dfda3e1027c05153
```

The optional MiniMax-H3 FL2VA model downloader is pinned to this immutable
Hugging Face revision and never stores the model in this source repository:

```text
MiniMaxAI/MiniMax-H3
939557dc319dd91227e30195a763f272ba7f8765
```

This project does not assert a license for the model snapshot; review the
upstream distribution terms before downloading or using it.

## Prepare the local ComfyUI quantized T2V pack

When the local ComfyUI checkout already contains the four H3 files, the
header-only preparer validates their schema before staging a portable model
root. It checks FL2VA ConvRot INT8 markers (group size 256), Qwen3-VL NVFP4
scales and `pre_quant_scale` aliases, the F16 video VAE, the F32 audio VAE,
and every safetensors offset/size without reading the large payloads.

```powershell
python scripts/prepare_h3_quantized_model.py --validate-only
python scripts/prepare_h3_quantized_model.py
```

The defaults target `E:\minimax-h3\ComfyUI\models`, use the small
configuration/tokenizer files under `E:\models\MiniMax-H3`, and create
`E:\minimax-h3\ComfyUI\models\h3_t2v_quantized`. Override the locations when
needed:

```powershell
python scripts/prepare_h3_quantized_model.py `
  --models-root E:\minimax-h3\ComfyUI\models `
  --base-root E:\models\MiniMax-H3 `
  --output-root E:\minimax-h3\ComfyUI\models\h3_t2v_quantized
```

The four large safetensors files are hardlinks, never copies. Only an
allow-listed set of small config/tokenizer files is copied below `base/`, and
`manifest.json` records source paths, sizes, header hashes, dtypes and schema
coverage. The command fails closed if validation fails or the output root
already exists; remove or choose a new output root only after reviewing the
manifest. The native model root is `h3_t2v_quantized/base` and retains the
`FL2VA/transformer`, `FL2VA/text_encoder`, `FL2VA/video_vae/source` and
`FL2VA/audio_vae` component directories.

The four linked payloads total 42,470,585,471 bytes (39.55 GiB). The native
CUDA path uses the FL2VA DiT's INT8 ConvRot weights directly, applies the
required online Hadamard rotation to activations, decodes the Qwen NVFP4/AWQ
weights with their blocked scales and activation-side `pre_quant_scale`, and
converts the F16 video VAE at load time. On Ampere (`sm_86`), Qwen NVFP4 is a
correctness/capacity path that materializes BF16 weights; it is not native
NVFP4 Tensor Core execution. This four-file root is T2V/FL2VA only and does not
include the separate Ref2VA transformer.

### ComfyUI CUDA conditioning bridge (recommended quantized path)

The native text encoder currently executes Qwen in BF16. Its direct NVFP4/AWQ
decoder is therefore marked experimental: it loads and runs, but is not the
semantic quality gate for this pack. For a usable quantized run, let the local
ComfyUI CUDA runtime encode the exact prompt, then pass its prompt-bound BF16
conditioning sidecar to the native INT8 DiT/VAE runtime. The bridge is explicit,
GPU-only and pure T2V; it never silently falls back to CPU or accepts Ref2VA
references.

The helper writes an atomic sidecar containing the prompt bytes, Comfy token
IDs/tags, BF16 conditioning and the whole Qwen model SHA-256. Invoke it with
the Python executable from the supplied ComfyUI virtual environment (the
wrapper performs this discovery automatically):

```powershell
<ComfyUI-root>\.venv\Scripts\python.exe scripts/encode_h3_quantized_prompt.py `
  --comfyui <ComfyUI-root> `
  --text-encoder <Qwen-NVFP4-or-AWQ-safetensors> `
  --output <cache-sidecar.h3c> `
  --prompt "A red fox walks through fresh snow in a pine forest." `
  --device cuda:0
```

For one command, the Windows wrapper discovers a `.venv`/`venv` Python below
the supplied ComfyUI root (or its parent) and the repository-relative native
build, or accepts `-ComfyPython` and `-BinaryPath` explicitly. It creates the cache sidecar first, validates the
helper's `model_sha256=` against `Get-FileHash`, sets
`H3CSPEED_TEXT_EMBEDDING` and `H3CSPEED_TEXT_ENCODER_SHA256` only for the child
process, then restores the caller's environment:

```powershell
.\scripts\run-h3-quantized.ps1 `
  -ModelRoot <prepared-root> `
  -ComfyUIRoot <ComfyUI-root> `
  -TextEncoder <Qwen-NVFP4-or-AWQ-safetensors> `
  -Prompt "A red fox walks through fresh snow in a pine forest." `
  -Output <output.mp4> `
  -Steps 20 -Width 256 -Height 256 -Frames 22
```

The wrapper defaults are `20` denoising steps, `256x256` and `22` frames. It
fails closed on missing roots, a non-CUDA device, helper failure, sidecar
absence, SHA mismatch, invalid dimensions or incompatible reuse settings. The
sidecar route has passed real 4-step and 20-step exit-0/full-decode smokes with
a recognizable fox. The 20-step 256x256/22-frame H.264/AAC artifact is recorded
in `VALIDATION_RESULTS.md`; direct native Qwen remains experimental.

After preparing the root, a short direct-native diagnostic run (experimental
Qwen path, without the sidecar) is:

```powershell
.\build\h3cspeed.exe `
  -d E:\minimax-h3\ComfyUI\models\h3_t2v_quantized\base `
  -p "A red fox walks through fresh snow in a pine forest." `
  --width 256 --height 256 --frames 22 --steps 4 --layers 50 `
  --reuse 1 --core-reuse 1 --ssd-streaming `
  -o outputs\quantized-smoke.mp4
```

Use 20 steps with the same layers/reuse settings for the quality baseline;
the 4-step command is only a pipeline and semantic diagnostic.

Inspect the selected memory policy before loading the model:

```bash
./build/h3cspeed-cuda-info
```

On an 8 GiB card it should report automatic low-VRAM mode, an approximately
5.75–6.2 GiB CUDA allocation budget, a 1.5 GiB resident weight cache and a
system-RAM cache derived from available RAM.

## Safest first run

The provided smoke test is a diagnostic run: it uses a square 256×256
internal/output canvas, the minimum trained 22-frame decoder chunk, four
denoising steps, all 50 layers, `--reuse 1` and `--core-reuse 1`. It is useful
for checking that a model can produce a recognizable animal on an 8 GiB card;
it is not a visual-quality PASS because four-step output can still be noisy.

```bash
./scripts/smoke-3070ti-8gb.sh ./MiniMax-H3
```

Equivalent direct command:

```bash
./scripts/run-3070ti-8gb.sh \
  -d ./MiniMax-H3 \
  -p "A red fox slowly walking through fresh snow, locked camera, natural light." \
  -o outputs/3070ti-smoke.mp4 \
  --width 256 --height 256 \
  --frames 22 --steps 4 \
  --layers 50 --reuse 1 --core-reuse 1
```

The wrapper adds only the low-VRAM plumbing that is needed for a safe default:
`--ssd-streaming`, a 22-frame minimum and a 256×256 output canvas when no
dimensions are supplied. It sets output width and height together so a square
diagnostic canvas is not paired with the upstream 864×480 output aspect ratio.
It does not add token reduction, layer thinning, or either reuse mode, so the
upstream quality defaults remain `--steps 20`, `--layers 50`, `--reuse 1` and
`--core-reuse 1`. If you explicitly pass `--reuse N`, the wrapper leaves
`--core-reuse` unset; explicitly passing both modes with values greater than
one is rejected before the binary starts.

For the formal quality baseline, use the same 256×256 shape with 20 denoising
steps and no token reduction:

```bash
./scripts/run-3070ti-8gb.sh \
  -d ./MiniMax-H3 \
  -p "A red fox slowly walking through fresh snow, locked camera, natural light." \
  -o outputs/3070ti-quality-baseline.mp4 \
  --width 256 --height 256 \
  --frames 22 --steps 20 \
  --layers 50 --reuse 1 --core-reuse 1
```

The four-step diagnostic path passes `--reuse 1 --core-reuse 1`, so every
requested pass is evaluated freshly. Keep its result separate from the
20-step quality baseline when assessing visual quality.

For a faster quality-oriented 8 GiB run, use the pinned preset. It requests a
480p (864×480) five-second clip while rendering internally at 288×160. H3
aligns the request to 124 frames (about 5.17 seconds). The preset keeps
20 denoising steps and all 50 DiT layers, and refreshes the persistent core
every four steps (`--core-reuse 4`). Denoiser reuse is pinned to 1; the preset
never enables token reduction and never combines reuse modes above 1:

```bash
./scripts/fast-quality-3070ti-8gb.sh \
  ./MiniMax-H3 \
  "A red fox walking through fresh snow in a pine forest." \
  outputs/fox-fast-quality.mp4
```

The model directory, prompt and output can also be supplied with
`H3_FAST_QUALITY_MODEL_DIR`, `H3_FAST_QUALITY_PROMPT` and
`H3_FAST_QUALITY_OUTPUT`. The preset delegates low-VRAM plumbing to the common
runner, including `ram+file` offload and SSD streaming. Its 480p dimensions are
output dimensions; the smaller internal canvas is the 8 GiB memory trade-off.
Increasing `H3_CUDA_HOST_CACHE_MIB` may reduce unrelated file-backed weight
evictions, but
it cannot prevent SSD stream-slot rereads: the stream intentionally retains a
bounded active layer window and rereads the next slot on each denoising pass.

After the quality baseline succeeds, try a larger output while retaining the
validated 256×256 internal render canvas:

```bash
./scripts/run-3070ti-8gb.sh \
  -d ./MiniMax-H3 \
  -p "A red fox walking through a snow-covered forest, stable tracking shot." \
  -o outputs/3070ti-balanced.mp4 \
  --width 576 --height 320 \
  --render-width 288 --render-height 160 \
  --frames 22 --steps 20 \
  --layers 50 --reuse 1 --core-reuse 1
```

Do not enable `--show` during initial 8 GiB validation; preview decoding creates
additional temporary VAE residency.

## Memory controls

All values use MiB:

| Variable | RTX 3070 Ti wrapper default | Purpose |
|---|---:|---|
| `H3_CUDA_OFFLOAD` | `ram+file` | `auto`, `ram+file`, or `off` |
| `H3_CUDA_VRAM_BUDGET_MIB` | `5888` | hard budget for all tracked CUDA allocations |
| `H3_CUDA_WEIGHT_CACHE_MIB` | `1536` | maximum resident offloadable weights |
| `H3_CUDA_HOST_CACHE_MIB` | automatic | system-RAM cache; default 60% of currently available RAM, capped at 64 GiB |
| `H3_CUDA_PINNED_HOST_MIB` | `128` | pinned host-copy cap, excluding staging |
| `H3_CUDA_STAGING_MIB` | `64` | reusable RAM/file-to-GPU transfer window |
| `H3_CUDA_RELEASE_SCRATCH` | `1` | free reusable GPU scratch after submit |
| `H3_CUDA_OFFLOAD_VERBOSE` | unset | print policy even when offload is disabled |

Example for a 96 GiB machine that reserves about 56 GiB for H3 weights:

```bash
export H3_CUDA_HOST_CACHE_MIB=57344
./scripts/run-3070ti-8gb.sh -d ./MiniMax-H3 -p "..." -o output.mp4
```

If the host cache is full, file-backed weights fall back to safetensors. INT8
weights generated at model preparation have no file source and therefore must
remain in the configured RAM cache; an insufficient cache produces a clear
error rather than silently losing the tensor.

## Validation

Without a GPU:

```bash
python3 scripts/validate_local.py
```

On the RTX 3070 Ti host:

```bash
H3CSPEED_CUDA_ARCHITECTURES=86 ./scripts/build.sh
ctest --test-dir build --output-on-failure
compute-sanitizer ./build/h3cspeed-cuda-info
./scripts/smoke-3070ti-8gb.sh ./MiniMax-H3
```

Monitor both VRAM and host RSS during the first run:

```bash
watch -n 0.5 nvidia-smi
/usr/bin/time -v ./scripts/smoke-3070ti-8gb.sh ./MiniMax-H3
```

## Known limits

- CUDA host-only syntax validation is not a substitute for an NVCC build.
- The reference attention and VAE convolution kernels are not yet
  FlashAttention/cuDNN optimized.
- Offload trades VRAM for PCIe traffic and host RAM; it will be substantially
  slower than full residency.
- Multi-GPU placement and CPU compute fallback are not yet implemented.
- A single `h3_gpu` context has a single-host-thread enqueue contract.

Release archive name: `h3cspeed-v0.2.0.zip`.
