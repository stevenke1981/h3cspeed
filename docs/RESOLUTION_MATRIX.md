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
PNG, prompt, seed, steps, frame count, and native width/height. The matrix
rejects `render_width`/`render_height` overrides and reference images whose PNG
IHDR does not exactly match the profile; no resize or stretch is part of the
benchmark path.

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

`--execute` only proves that the child processes completed and wrote their
private outputs. It does not mark media, scheduler, audio, visual quality, or
H3-vs-Comfy parity as passed. Those require ffprobe, full ffmpeg decode,
audio/non-silent checks, profile traces, and visual inspection per profile.
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
