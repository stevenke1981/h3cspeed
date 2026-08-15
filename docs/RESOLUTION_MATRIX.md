# Native resolution matrix

This matrix is a separate throughput comparison for H3 and ComfyUI. It does
not change the fixed PERF-002 or PERF-007 contracts.

| Profile | Native output and render | H3 grid | Frames | Duration at 24 fps |
| --- | ---: | ---: | ---: | ---: |
| 240p class | 448x256 | 32 px | 124 | 5.166667 s |
| 480p class | 864x480 | 32 px | 124 | 5.166667 s |
| 720p class | 1280x704 | 32 px | 124 | 5.166667 s |

The names are resolution classes, not claims that the output is the legacy
426x240, 854x480, or 1280x720 raster. Those common rasters do not satisfy the
H3 32-pixel geometry contract. Both engines receive the exact same per-profile
PNG, prompt, seed, steps, frame count, and native width/height. The H3 matrix
argv deliberately contains no render-dimension override; the native runner's
default is therefore `render == output`. The matrix rejects render/resize/
stretch flags and reference images whose PNG IHDR does not exactly match the
profile; no runtime resize or stretch is part of the benchmark path.

The generated talking-presenter reference images are local benchmark inputs,
not repository payloads. Bind them by SHA-256 in the private configuration.
The Codex-generated master was aspect-preservingly centre-cropped and resized
to each valid grid profile; this changes framing slightly between profiles but
does not geometrically stretch the presenter. Each H3/Comfy pair uses the same
profile PNG, so the within-profile timing comparison remains fair.

## Dry plan

Create a private JSON configuration with the bound ComfyUI entrypoints/models,
H3 model root/text encoder/binary, prompt, and one exact-size `reference_png`
plus `timeout_seconds` under each of `profiles.240`, `profiles.480`, and
`profiles.720`. The top-level `powershell_executable` must be an absolute path
to the intended `pwsh.exe`; the plan never resolves a bare executable name from
`PATH`. Keep the configuration and output directory outside the source,
ComfyUI, and model trees. Then run:

```powershell
python scripts/run_resolution_matrix.py `
  --config E:\private\resolution-matrix.json `
  --output-dir E:\private\resolution-matrix-run
```

The default only publishes a no-clobber plan and reports `NOT_RUN`. Every
profile has one H3 command and one ComfyUI command; H3 is invoked with the
layer-major VAE, async refill, and one-ahead DiT prefetch switches, while both
engines remain native-size.

On Windows the runner replaces inherited permissions on the new output root
with a protected, current-user-only, inheritable DACL and verifies it before
writing the private plan. This dry-plan ACL step uses only OS-owned executables
resolved from the Win32 system directory; it never launches the config-selected
PowerShell producer. If the DACL cannot be established, planning fails closed.
POSIX output roots are verified as mode `0700`.

## Real execution and evidence

Real execution is deliberately explicit and sequential:

```powershell
python scripts/run_resolution_matrix.py `
  --config E:\private\resolution-matrix.json `
  --output-dir E:\private\resolution-matrix-run `
  --execute
```

For a counterbalanced follow-up, add `--reverse-order`. This keeps the
canonical plan and per-profile pairing unchanged but launches ComfyUI before
H3cspeed within each profile, recording `execution_order` as
`comfyui_then_h3cspeed` in the summary. The flag is rejected unless
`--execute` is also present. It changes process order only; it does not flush
the Windows filesystem cache or by itself establish a cold-cache speed gate.

This second command may point at the output directory created by the preceding
dry run. The runner reads the existing `resolution-matrix-plan.json`, re-loads
all bound inputs, rebuilds the expected canonical plan, and requires an exact
byte match before execution. It never recreates or overwrites an existing plan.
An existing output directory without that plan, or with a stale/non-canonical
plan, fails closed. A fresh one-command `--execute` remains supported and first
publishes the same immutable dry plan.

`--execute` publishes `resolution-matrix-execution.json` with one wall-time
entry for each engine in each selected profile, plus an observed
`h3cspeed_over_comfyui_wall_ratio`. This is process-completion evidence only;
it does not mark media, scheduler, audio, visual quality, or H3-vs-Comfy
parity as passed. Those require ffprobe, full ffmpeg decode, audio/non-silent
checks, profile traces, and visual inspection per profile. The summary labels
speed as `OBSERVED_ONLY`; it never converts a single ordered run into a formal
counterbalanced speed-alignment result.
The 480p native H3 path has previously exceeded a one-hour timeout, and 720p
may take hours or run out of 8 GiB VRAM; timeout/OOM is recorded as a failure,
never silently replaced by a lower internal render.

Each profile timeout applies to a complete child process tree. On Windows the
runner starts a new process group and uses the absolute system `taskkill.exe`
with `/T /F` on timeout; on POSIX it starts a new session and kills that process
group. Failure to confirm shutdown is itself a contract failure.

## First real smoke observation

On 2026-08-15 the 240p-class profile was exercised on the bound RTX 3070 Ti
using the exact 448x256 reference PNG, the same prompt, 22 frames, 24 fps, and
2 steps. Both outputs passed ffprobe, full `ffmpeg -xerror` decode, audio
non-silence, and a manual frame-11 visual check. Sage attention was selected
with zero fallbacks in both routes.

| Engine | Whole command wall | Engine-reported work | MP4 SHA-256 | Size |
| --- | ---: | ---: | --- | ---: |
| h3cspeed | 498.1 s (includes first-run sidecar) | 248.93 s | `798eafef7e692bc1df81b214f3bd7f886de86682dff63fc5bb26299a545799f8` | 53,717 B |
| ComfyUI | 447.9 s | 345.38 s prompt execution | `2dc5ca5e9f7a6189f12d7ae106bdd9a93061574962afb24cd0654f05f43eb3df` | 31,024 B |

The first-run end-to-end H3 command was about 11% slower because it included
FL2VA sidecar generation (conditioning plus canonical first frame). The H3
engine work itself was shorter than the ComfyUI prompt execution, but those
two reported clocks have different scope and are not an A/B speed claim. A
cached-sidecar counterbalanced pair is still required before declaring a
throughput win.

The larger H3 MP4 is an encoding-policy difference, not a resolution or frame
mismatch: H3 writes `libx264 -preset fast -crf 18` and AAC at 192 kbps, while
the ComfyUI `SaveVideo(codec=auto)` path preserves its own stream defaults.
Both files are 448x256, 22 frames, H.264/AAC, 24 fps, and approximately
0.917--0.925 seconds long. 480p and 720p real runs remain `NOT_RUN` until
their independent resource/timeouts are exercised.

## Formal 240p-class five-second run

On the same bound host, a separate 124-frame run reused the already validated
sidecar and wrote to a fresh private output tree. Both children produced
448x256, 124-frame, 24-fps H.264/AAC media; full `ffmpeg -xerror` video/audio
decode, non-silent audio, Sage selection, zero attention fallback, and a manual
frame-62 check passed. This was a sequential pair, not a counterbalanced
cold-cache A/B: the clocks have different scopes and the Windows filesystem
cache was not flushed.

| Engine | Engine/prompt wall | MP4 SHA-256 | Size |
| --- | ---: | --- | ---: |
| h3cspeed | 664.741099 s | `1a36e98a9d1a5f2974777b319343b3c4c18e8bb06544c47586620774873ba452` | 254,998 B |
| ComfyUI | 345.26 s prompt execution | `8de2d04454bf08ef9627ea1ec5419854a096df836053aa333fb2f706fc85ec4f` | 150,654 B |

The observed H3 engine clock is about 1.93x the ComfyUI prompt clock for this
non-counterbalanced pair; it is not a formal speed-alignment result. H3's
profile identifies DiT offload churn as the dominant cost (496.836 s DiT,
359.261 s eviction, 124.051 s file reads, 1,160 evictions and 40.26 GiB of
uploads). A fresh cache-tuning candidate at 1,962 MiB changed the DiT profile
to 497.997 s (359.869 s eviction, 123.235 s file reads, 1,146 evictions) and
the total engine wall to 664.960 s, so the larger cache did not materially
improve this shape. No cache default was changed on the strength of that one
run; the next optimization should target eviction/file-read overlap and be
validated with a matched pair.

## Counterbalanced 240p-class pair

On 2026-08-16 the cached-sidecar 240p profile was run twice with the same
`build-perf008/h3cspeed.exe`, manifest, sidecar, prompt, reference PNG, seed,
and native 448x256/124-frame/24-fps/2-step contract. The first run used the
canonical H3-then-Comfy order; the second used `--reverse-order`, which runs
ComfyUI before H3 within the profile. Both private output trees passed the
same media checks as the formal run: H.264/AAC, 448x256, 124 frames, 24 fps,
full video/audio `ffmpeg -xerror` decode, non-silent audio, Sage selected with
zero attention fallbacks, and the same manually inspected speaking frame.

| Order | H3 child wall | ComfyUI child wall | Whole-command ratio | H3 MP4 SHA-256 | ComfyUI MP4 SHA-256 |
| --- | ---: | ---: | ---: | --- | --- |
| H3 then ComfyUI | 855.971099 s | 435.675537 s | 1.9647x | `1a36e98a9d1a5f2974777b319343b3c4c18e8bb06544c47586620774873ba452` | `8de2d04454bf08ef9627ea1ec5419854a096df836053aa333fb2f706fc85ec4f` |
| ComfyUI then H3 | 853.185943 s | 452.711302 s | 1.8846x | `1a36e98a9d1a5f2974777b319343b3c4c18e8bb06544c47586620774873ba452` | `8de2d04454bf08ef9627ea1ec5419854a096df836053aa333fb2f706fc85ec4f` |

The process-wall means are 854.578521 s for H3 and 444.193419 s for ComfyUI
(1.9239x). The H3 order spread was 2.79 s, while the ComfyUI spread was
17.04 s; output hashes were identical across orders. This is useful
counterbalanced evidence, but it is still `OBSERVED_ONLY`: Windows filesystem
cache flushing was not performed, and the two child clocks include different
producer scopes. It is not a formal cold-cache speed-alignment PASS.

Two follow-up H3-only churn probes also stayed opt-in and outside the matrix
default. Raising the weight cache from 1,536 to 1,962 MiB produced a 669.961 s
engine result with a 502.521 s DiT profile, 364.713 s eviction time and 1,146
logical evictions. Limiting each one-ahead batch to four weights produced a
668.469 s engine result with a 501.029 s DiT profile, 363.038 s eviction time
and 1,160 logical evictions. Both probes emitted the same H3 MP4 SHA as the
baseline and passed full media/audio decode, but neither reduced the churn
materially; no default cache or prefetch setting was changed. The next source
slice should measure and reduce eviction/file-read serialization itself,
rather than stacking another unvalidated capacity knob. The 480p and 720p
profiles remain `NOT_RUN`.

The formal 240p media artifacts are kept outside the repository under the
private benchmark directory; the 480p and 720p profiles remain `NOT_RUN`.
