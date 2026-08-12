#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${1:-$(<"$ROOT/VERSION")}"
OUTPUT="${2:-$ROOT/../h3cspeed-v${VERSION}.zip}"
exec python3 "$ROOT/scripts/package.py" --root "$ROOT" \
  --version "$VERSION" --output "$OUTPUT"
