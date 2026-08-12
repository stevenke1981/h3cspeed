#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${H3CSPEED_BUILD_DIR:-$ROOT/build}"
BUILD_TYPE="${H3CSPEED_BUILD_TYPE:-Release}"
GENERATOR="${CMAKE_GENERATOR:-Ninja}"

python3 "$ROOT/scripts/bootstrap.py"

cmake -S "$ROOT" -B "$BUILD_DIR" -G "$GENERATOR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DH3CSPEED_CUDA_ARCHITECTURES="${H3CSPEED_CUDA_ARCHITECTURES:-native}" \
  "${@}"
cmake --build "$BUILD_DIR" --parallel

echo "Built: $BUILD_DIR/h3cspeed"
