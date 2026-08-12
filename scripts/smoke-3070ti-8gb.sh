#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="${1:-}"
PROMPT="${2:-一隻紅狐狸在雪地慢慢行走，固定鏡頭，自然光，環境風聲。}"

if [[ -x "$ROOT/scripts/run-3070ti-8gb.sh" ]]; then
    RUNNER="$ROOT/scripts/run-3070ti-8gb.sh"
    DEFAULT_OUTPUT="$ROOT/outputs/3070ti-8gb-smoke.mp4"
elif [[ -x "$SCRIPT_DIR/h3cspeed-3070ti-8gb" ]]; then
    # Installed layout: both helper executables live in the same bin directory.
    RUNNER="$SCRIPT_DIR/h3cspeed-3070ti-8gb"
    DEFAULT_OUTPUT="$PWD/outputs/3070ti-8gb-smoke.mp4"
elif command -v h3cspeed-3070ti-8gb >/dev/null 2>&1; then
    RUNNER="$(command -v h3cspeed-3070ti-8gb)"
    DEFAULT_OUTPUT="$PWD/outputs/3070ti-8gb-smoke.mp4"
else
    echo "RTX 3070 Ti runner was not found. Build or install h3cspeed first." >&2
    exit 1
fi

OUTPUT="${3:-$DEFAULT_OUTPUT}"
if [[ -z "$MODEL_DIR" ]]; then
    echo "usage: $0 MODEL_DIR [PROMPT] [OUTPUT]" >&2
    exit 2
fi
mkdir -p "$(dirname "$OUTPUT")"

exec "$RUNNER" \
    -d "$MODEL_DIR" \
    -p "$PROMPT" \
    -o "$OUTPUT" \
    --seed 42 \
    --width 256 --height 256 \
    --frames 22 --steps 4 \
    --layers 50 --reuse 1 --core-reuse 1 \
    --ssd-streaming
