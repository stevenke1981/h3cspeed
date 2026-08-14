#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${H3CSPEED_MODEL_COMPAT_BUILD_DIR:-$ROOT/build-model-compat}"
CC_BIN="${CC:-cc}"

mkdir -p "$BUILD_DIR"

python3 -m py_compile \
    "$ROOT/scripts/h3_model_info.py" \
    "$ROOT/scripts/h3_model_metadata.py" \
    "$ROOT/scripts/h3_safetensors_info.py" \
    "$ROOT/tests/test_h3_model_info.py"
python3 -m unittest discover \
    -s "$ROOT/tests" \
    -p 'test_h3_model_info.py' \
    -v

"$CC_BIN" \
    -std=c11 \
    -Wall -Wextra -Wpedantic -Werror \
    -I"$ROOT/src" \
    "$ROOT/tests/test_model_config.c" \
    "$ROOT/src/h3_model_config.c" \
    "$ROOT/src/h3_model_compat.c" \
    -o "$BUILD_DIR/test_model_config"
"$BUILD_DIR/test_model_config"
