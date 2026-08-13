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
- SageAttention is accepted only when the runtime trace records at least one
  backend hit and zero fallbacks. A requested flag or dispatch aggregate is
  not backend evidence.
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
2. `bindings.json` maps model, conditioning (`token_ids`, `token_tags` and
   `qwen_hidden`) and engine labels to actual regular files to hash. Paths are
   consumed locally and never copied into the manifest.
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
12/3 shifts and the attention trace records real Sage hits with zero fallback.

```powershell
python scripts/run_perf002_smoke.py `
  --manifest E:\private\perf002\evidence\input-manifest.json `
  --command-config E:\private\perf002\h3cspeed-smoke.private.json `
  --output-dir E:\private\perf002\h3cspeed-smoke
```

`SMOKE_PASS` is an individual process, media and trace result only. The result
always keeps `matched_ab_status: NOT_RUN`; it cannot establish throughput or
quality parity. Run ComfyUI with a separate config and directory. Neither
engine reads or overwrites the other's output.
