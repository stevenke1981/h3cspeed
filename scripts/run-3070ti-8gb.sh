#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$SCRIPT_DIR/h3cspeed" ]]; then
    H3_BINARY="$SCRIPT_DIR/h3cspeed"
elif [[ -x "$SCRIPT_DIR/h3cspeed.exe" ]]; then
    # Installed native Windows layout when invoked through Git Bash.
    H3_BINARY="$SCRIPT_DIR/h3cspeed.exe"
elif [[ -x "$ROOT/bin/h3cspeed" ]]; then
    # Extracted Linux runtime archive layout.
    H3_BINARY="$ROOT/bin/h3cspeed"
elif [[ -x "$ROOT/bin/h3cspeed.exe" ]]; then
    # Extracted Windows runtime archive layout when invoked through Git Bash.
    H3_BINARY="$ROOT/bin/h3cspeed.exe"
elif [[ -x "$ROOT/build-native/h3cspeed.exe" ]]; then
    # Native Windows developer layout when invoked through Git Bash.
    H3_BINARY="$ROOT/build-native/h3cspeed.exe"
elif [[ -x "$ROOT/build/h3cspeed" ]]; then
    H3_BINARY="$ROOT/build/h3cspeed"
else
    echo "h3cspeed binary was not found. Build sm_86 first:" >&2
    echo "  H3CSPEED_CUDA_ARCHITECTURES=86 ./scripts/build.sh" >&2
    exit 1
fi

# Conservative RTX 3070 Ti / 8 GiB defaults. Export any variable before this
# wrapper to override it. The runtime clamps the budget to currently free VRAM.
export H3_CUDA_LOW_VRAM="${H3_CUDA_LOW_VRAM:-1}"
export H3_CUDA_OFFLOAD="${H3_CUDA_OFFLOAD:-ram+file}"
export H3_CUDA_VRAM_BUDGET_MIB="${H3_CUDA_VRAM_BUDGET_MIB:-5888}"
export H3_CUDA_WEIGHT_CACHE_MIB="${H3_CUDA_WEIGHT_CACHE_MIB:-1536}"
export H3_CUDA_PINNED_HOST_MIB="${H3_CUDA_PINNED_HOST_MIB:-128}"
export H3_CUDA_STAGING_MIB="${H3_CUDA_STAGING_MIB:-64}"
export H3_CUDA_RELEASE_SCRATCH="${H3_CUDA_RELEASE_SCRATCH:-1}"
export H3_PROFILE="${H3_PROFILE:-1}"

has_option() {
    local wanted="$1"
    shift
    local item
    for item in "$@"; do
        [[ "$item" == "$wanted" || "$item" == "$wanted="* ]] && return 0
    done
    return 1
}

args=("$@")
if ! has_option --ssd-streaming "${args[@]}"; then
    args+=(--ssd-streaming)
fi
# Whole-denoiser reuse and core reuse are mutually exclusive when either is
# greater than one.  Leave both unset by default so upstream's quality
# defaults (reuse 1 / core-reuse 1) remain in force.
has_reuse=0
has_core_reuse=0
if has_option --reuse "${args[@]}"; then
    has_reuse=1
fi
if has_option --core-reuse "${args[@]}"; then
    has_core_reuse=1
fi
if (( has_reuse && has_core_reuse )); then
    reuse_value=""
    core_reuse_value=""
    for ((index = 0; index < ${#args[@]}; index++)); do
        case "${args[index]}" in
            --reuse=*) reuse_value="${args[index]#--reuse=}" ;;
            --reuse)
                if (( index + 1 < ${#args[@]} )); then
                    reuse_value="${args[index + 1]}"
                fi
                ;;
            --core-reuse=*) core_reuse_value="${args[index]#--core-reuse=}" ;;
            --core-reuse)
                if (( index + 1 < ${#args[@]} )); then
                    core_reuse_value="${args[index + 1]}"
                fi
                ;;
        esac
    done
    if [[ "$reuse_value" != "1" && "$core_reuse_value" != "1" ]]; then
        echo "error: --reuse >1 and --core-reuse >1 cannot be combined; choose one reuse mode" >&2
        exit 2
    fi
fi
if ! has_option --frames "${args[@]}" &&
   ! has_option --seconds "${args[@]}"; then
    args+=(--frames 22)
fi
# When no output or render dimensions are supplied, make the output itself a
# square 256x256 canvas. This avoids pairing a square render canvas with the
# upstream 864x480 output and triggering an aspect-ratio rejection. Explicit
# output or render dimensions always win.
if ! has_option --width "${args[@]}" &&
   ! has_option --height "${args[@]}" &&
   ! has_option --render-width "${args[@]}" &&
   ! has_option --render-height "${args[@]}"; then
    args+=(--width 256 --height 256)
fi

exec "$H3_BINARY" "${args[@]}"
