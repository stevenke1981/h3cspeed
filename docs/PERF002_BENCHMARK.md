# PERF-002 matched benchmark contract

PERF-002 compares ComfyUI and h3cspeed only after both engines consume the
same immutable FL2VA first-frame I2V inputs. The committed code is the first
portable walking skeleton: it creates and validates input manifests and runs
real ffprobe/ffmpeg media QA. It does **not** yet launch either engine and does
not establish a speed result.

## Fixed primary contract

- Native reference PNG: 864x480, generated once with Codex ImageGen. Do not use
  the old centered 480x480 image with blurred side fill.
- FL2VA first-frame I2V, no Ref2VA, no LoRA, native audio.
- 864x480 internal and output geometry, 124 frames at 24 fps, 8 steps, 50
  layers, reuse 1, core-reuse 1, seed 42 and TF32 disabled.
- `algorithm_parity`: both engines use `dual_clock_euler`, `native_flow`, video
  shift 12 and audio shift 3. Sigma arrays must agree within `1e-6` and raw
  audio update semantics require separate runtime evidence.
- SageAttention is accepted only when the runtime trace is scoped to
  `dit_bf16`, records at least one real backend hit and reports zero
`unexpected_fallbacks`. `expected_native_calls` records the explicit
native/unset backend control run when that separate run is performed; the
bounded Comfy producer writes `0` because it does not execute that control,
which is not native-path evidence. A BF16 call that requested Sage but is
ineligible is an unexpected fallback. F32 VAE calls are outside this scope. A
requested flag or aggregate dispatch count is not backend evidence.
- One cold and at least three warm trials per engine. Conditioning, model load,
  keyframe encode, DiT, both VAEs and mux timing stay separate.

`engine_recommended` is a secondary throughput track. ComfyUI
`res_multistep/simple` versus h3cspeed Euler is intentionally
apples-to-oranges and cannot be used for algorithm or quality parity.

## Create immutable input evidence

Prepare three private files outside the source, ComfyUI and model trees:

1. `spec.json` contains labels and the fixed public contract, but no prompt,
   paths or secrets. `models.<engine>` and `engines.<engine>` are lists of
   portable labels.
2. `bindings.json` maps model, conditioning and engine labels to actual regular
   files to hash. ComfyUI binds `token_ids`, `token_tags` and `qwen_hidden`;
   h3cspeed binds the single v2 `sidecar` artifact that contains those three
   payloads. Paths are consumed locally and never copied into the manifest.
3. `prompt.txt` contains the private prompt. Only its SHA-256 is recorded.

Then run:

```powershell
python scripts/run_perf002_ab.py create-input `
  --spec E:\private\perf002\spec.json `
  --bindings E:\private\perf002\bindings.json `
  --reference-png E:\private\perf002\reference-864x480.png `
  --prompt-file E:\private\perf002\prompt.txt `
  --output-dir E:\private\perf002\evidence

python scripts/validate_perf002_manifest.py `
  E:\private\perf002\evidence\input-manifest.json
```

The writer hashes every bound file, publishes canonical JSON without
overwriting existing evidence and writes an exact checksum sidecar. Inputs and
all ancestors must be regular, non-link paths. The manifest stores no raw
prompt, absolute path, environment, command line, stdout, hostname or PID.

## Media harness

For a completed engine output, run `validate-media`. It requires H.264
864x480, exactly 124 frames at 24 fps, AAC 32 kHz stereo, full video/audio
decode, non-silent PCM and five sampled frame hashes. It publishes an
individual engine result as `NOT_RUN` until scheduler and attention runtime
evidence and actual timings are supplied. Automated QA always leaves visual
review as `MANUAL_REQUIRED`.

Synthetic media and 22-frame smokes prove only the harness. A performance
claim requires both engines to complete the fixed 124-frame contract, one cold
plus three warm trials, immutable pre/post hashes, backend traces, full media
QA and human five-frame review. Until then matched A/B remains `NOT_RUN`.

## PERF-002C isolated engine smoke

`run_perf002_smoke.py` executes one engine in its own process, output directory
and private log. Its private command config provides:

- `engine`: `h3cspeed` or `comfyui`;
- an absolute `argv` list (never stored in the result);
- `command_artifacts` entries that bind `argv[0]` and the executable driver
  back to SHA-256 artifacts in the immutable engine manifest;
- the same private `bindings`, `reference_png` and `prompt_file` used by the
  immutable manifest;
- distinct `output_media`, `scheduler_trace` and `attention_trace` paths;
- `protected_roots` for the h3cspeed source, ComfyUI and model directories;
- a small allowlisted `environment` object.

The adapter rehashes all bound inputs before and after the child process,
rejects output paths inside the three protected roots, runs without a shell,
and requires the 864x480/22-frame/24 fps/2-step smoke media contract. It accepts
the smoke only when scheduler evidence is `dual_clock_euler/native_flow` with
12/3 shifts and the `dit_bf16` attention trace records real Sage hits with zero
unexpected fallback.

The h3cspeed binary now contains an opt-in producer for these two runtime
traces. Set both `H3CSPEED_PERF002_SCHEDULER_TRACE` and
`H3CSPEED_PERF002_ATTENTION_TRACE` to new absolute files outside source/model
trees. The producer serializes the actual serving sigma arrays and scoped CUDA
dispatch counters after denoising, and publishes scheduler evidence only after
all requested audio Euler updates execute; partial, concurrent and existing-target
publishes fail closed. This instrumentation has portable contract coverage,
but a real 22-frame engine run is still `NOT_RUN`.

### h3cspeed direct-binary command contract

The h3cspeed smoke is deliberately a direct binary invocation: both
`command_artifacts.executable` and `command_artifacts.driver` must be the
manifest-bound `engines.h3cspeed.binary` at `argv[0]`. Its h3-only
`command_inputs` object must contain exactly these bindings:

```json
{
  "-d": {"section": "config", "label": "model_root"},
  "-p": {"section": "config", "label": "prompt_file"},
  "--first-frame": {"section": "fixture", "label": "reference_png"},
  "-o": {"section": "config", "label": "output_media"},
  "H3CSPEED_TEXT_EMBEDDING": {"section": "conditioning", "label": "sidecar"},
  "H3CSPEED_TEXT_ENCODER_SHA256": {"section": "models", "label": "qwen"},
  "H3_FFMPEG": {"section": "engines", "label": "ffmpeg"},
  "H3CSPEED_PERF002_SCHEDULER_TRACE": {"section": "config", "label": "scheduler_trace"},
  "H3CSPEED_PERF002_ATTENTION_TRACE": {"section": "config", "label": "attention_trace"}
}
```

Before the adapter is invoked, run the existing ComfyUI helper
`scripts/encode_h3_quantized_prompt.py` to create a v2 FL2VA sidecar and its
canonical `.h3c.first.png`. Bind that sidecar as the additional
`conditioning.h3cspeed.sidecar` file in `bindings.json`; the adapter checks its
pre/post SHA-256 and requires `H3CSPEED_TEXT_EMBEDDING` to point to that exact
file. It also requires `H3CSPEED_TEXT_ENCODER_SHA256` to equal the manifest's
`models.h3cspeed.qwen` hash. The h3 model bindings must include `fl2va`,
`qwen`, `video_vae`, `audio_vae`, `transformer_config`, `tokenizer`,
`video_vae_config` and `audio_vae_config`. All eight files must resolve to the
native loader's exact FL2VA component paths below `-d`; each weight directory
must contain exactly the one bound safetensors payload, and an enabled Ref2VA
index is rejected.

The adapter requires each of `-d`, `-p`, `--first-frame`, `-o`, `--width`,
`--height`, `--frames`, `--steps`, `--layers`, `--reuse`, `--core-reuse` and
`--seed` exactly once with values `864`, `480`, `22`, `2`, `50`, `1`, `1` and
`42` respectively. `-p` must equal the prompt file bytes, `--first-frame` must
be the immutable manifest reference, and the trace environment paths must
equal the configured trace files. `--last-frame`, render-size overrides and
unknown/duplicate flags are rejected, so a render downgrade cannot silently
pass as the smoke contract. Additional positional arguments are rejected,
`H3_CUDA_ATTENTION` must be `sage`, and `H3_CUDA_TF32` must be exactly `0`.
Before launch, the adapter parses the bound sidecar as v2 first-frame FL2VA and
checks its 864x480 geometry, prompt, Qwen digest, canonical first-frame digest
and payload lengths. The h3 engine bindings must also include absolute
`ffmpeg` and `ffprobe` executables. `H3_FFMPEG` must point to the bound FFmpeg
used by native first-frame decode/mux, and the adapter requires its own
`--ffmpeg`/`--ffprobe` QA arguments to resolve to those same hashed tools.

An individual h3 `SMOKE_PASS` proves only the isolated process, immutable
bindings, 22-frame media QA and runtime traces. It keeps
`matched_ab_status: NOT_RUN`; it is not a matched A/B, speed or quality result.

```powershell
python scripts/run_perf002_smoke.py `
  --manifest E:\private\perf002\evidence\input-manifest.json `
  --command-config E:\private\perf002\h3cspeed-smoke.private.json `
  --output-dir E:\private\perf002\h3cspeed-smoke `
  --ffmpeg C:\ffmpeg\bin\ffmpeg.exe `
  --ffprobe C:\ffmpeg\bin\ffprobe.exe
```

`SMOKE_PASS` is an individual process, media and trace result only. The result
always keeps `matched_ab_status: NOT_RUN`; it cannot establish throughput or
quality parity. Run ComfyUI with a separate config and directory. Neither
engine reads or overwrites the other's output.

## Repo-owned ComfyUI smoke driver

`scripts/perf002_comfy_trace.py` is the ComfyUI-side `argv[1]` driver for the
isolated adapter. It starts the bound ComfyUI checkout in-process on a random
loopback port, sends the fixed first-frame `I2VA` FL2VA graph at 864x480,
22 frames, two `dual_clock_euler/native_flow` steps and seed 42, then copies the
real `SaveVideo` output into the private result directory. ComfyUI output,
input, temporary files, user data and its SQLite database are redirected to a
new private runtime directory; the source/custom-node/model trees are not
written. Python bytecode generation is disabled for the child.

The driver wraps the loaded T8 `setup_dual_clock_sampling` and
`sample_minimax_h3_dual_clock_euler` functions, and the loaded ComfyUI
`sageattn`/`attention_pytorch` globals. It publishes scheduler and attention
JSON only after the actual graph completes, the raw-audio update count equals
the requested steps, at least one Sage backend call occurred and no fallback
was observed. Trace and media publishing are no-clobber operations. A startup,
graph, media, scheduler or Sage failure leaves the smoke unverified and exits
non-zero; it does not create a synthetic replacement file.

This producer deliberately binds a bounded entry-point file set: `main.py`,
the T8 `sampling.py`/`nodes.py`, ComfyUI `attention.py`, and the four model
files. This is not the runtime import closure: T8 helpers, other ComfyUI model
modules, and the Python environment remain outside the current lock.
Full-tree/venv hashing, matched A/B timing, and visual QA remain `NOT_RUN`
until the bound-host run.
The driver is a one-shot process-bound child: it stops and joins its private
loopback server before returning, and process exit owns final Comfy cleanup.

The 002C command config must declare `command_inputs` for all eight producer
flags. Each entry maps to the manifest's `engines.comfyui` labels
`comfy_main`, `t8_sampling`, `t8_nodes`, `comfy_attention`, `model_file`,
`clip_file`, `video_vae_file`, and `audio_vae_file`; the adapter rehashes each
absolute argv path before and after execution.
The config must also declare `runtime_dir`; the matching argument must name a
new or empty non-linked directory outside the source, ComfyUI, and model roots.
The reference PNG, prompt file, output media, and both trace arguments are
required exactly once and must equal the corresponding private config values.
The Comfy attention report's
`expected_native_calls: 0` means this bounded run did not execute the separate
h3cspeed native-control path; it is not a native-control PASS.

Example private command shape (paths and model names are local bindings and
must not be copied into evidence):

```powershell
python scripts/perf002_comfy_trace.py `
  --comfy-main E:\minimax-h3\ComfyUI\main.py `
  --t8-sampling E:\minimax-h3\ComfyUI\custom_nodes\comfyui-minimax-h3-audio-T8\sampling.py `
  --t8-nodes E:\minimax-h3\ComfyUI\custom_nodes\comfyui-minimax-h3-audio-T8\nodes.py `
  --comfy-attention E:\minimax-h3\ComfyUI\comfy\ldm\modules\attention.py `
  --model-file E:\minimax-h3\ComfyUI\models\diffusion_models\minimax_h3_fl2va_pruned_int8_convrot.safetensors `
  --clip-file E:\minimax-h3\ComfyUI\models\text_encoders\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors `
  --video-vae-file E:\minimax-h3\ComfyUI\models\vae\minimax_h3_video_vae_fp16.safetensors `
  --audio-vae-file E:\minimax-h3\ComfyUI\models\vae\minimax_h3_audio_vae_fp32.safetensors `
  --runtime-dir E:\private\perf002\comfy-runtime `
  --reference-png E:\private\perf002\reference-864x480.png `
  --prompt-file E:\private\perf002\prompt.txt `
  --output-media E:\private\perf002\comfy-smoke\smoke.mp4 `
  --scheduler-trace E:\private\perf002\comfy-smoke\scheduler.json `
  --attention-trace E:\private\perf002\comfy-smoke\attention.json
```

The real GPU driver, model availability, full decode and five-frame visual
review remain `NOT_RUN` until this command is executed on the bound host.
