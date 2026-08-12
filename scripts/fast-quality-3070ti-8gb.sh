#!/usr/bin/env bash
set -euo pipefail

# RTX 3070 Ti / 8 GiB quality-oriented preset. This requests a 480p, five-second
# clip (aligned by H3 to 124 frames / about 5.17 s) while using a 288x160
# internal render canvas and all 50 DiT blocks.
# Refresh the persistent core every four denoising steps.  Core reuse and
# denoiser reuse are mutually exclusive when either value is greater than one;
# this preset pins denoiser reuse to 1.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_DIR="${1:-${H3_FAST_QUALITY_MODEL_DIR:-}}"
PROMPT="${2:-${H3_FAST_QUALITY_PROMPT:-A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind.}}"
# Installed helpers may live under a read-only prefix such as /usr/local/bin.
# Keep the default artifact in the caller's working directory instead.
OUTPUT="${3:-${H3_FAST_QUALITY_OUTPUT:-$PWD/outputs/3070ti-fast-quality.mp4}}"

if [[ -z "$MODEL_DIR" ]]; then
    echo "usage: $0 MODEL_DIR [PROMPT] [OUTPUT]" >&2
    echo "or set H3_FAST_QUALITY_MODEL_DIR, H3_FAST_QUALITY_PROMPT and H3_FAST_QUALITY_OUTPUT" >&2
    exit 2
fi
if [[ $# -gt 3 ]]; then
    echo "error: fast-quality accepts only MODEL_DIR, PROMPT and OUTPUT" >&2
    exit 2
fi

if [[ -x "$ROOT/scripts/run-3070ti-8gb.sh" ]]; then
    RUNNER="$ROOT/scripts/run-3070ti-8gb.sh"
elif [[ -x "$SCRIPT_DIR/h3cspeed-3070ti-8gb" ]]; then
    RUNNER="$SCRIPT_DIR/h3cspeed-3070ti-8gb"
elif command -v h3cspeed-3070ti-8gb >/dev/null 2>&1; then
    RUNNER="$(command -v h3cspeed-3070ti-8gb)"
else
    echo "h3cspeed RTX 3070 Ti runner was not found; build or install h3cspeed first." >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"

# The common runner owns low-VRAM plumbing, including RAM + file offload and
# SSD streaming.  Keep this preset's quality knobs explicit and stable:
# token reduction is intentionally omitted (CLI default is full tokens).
exec "$RUNNER" \
    -d "$MODEL_DIR" \
    -p "$PROMPT" \
    -o "$OUTPUT" \
    --seed 42 \
    --width 864 --height 480 \
    --render-width 288 --render-height 160 \
    --seconds 5 --steps 20 \
    --layers 50 --reuse 1 --core-reuse 4 \
    --ssd-streaming
