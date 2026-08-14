#!/usr/bin/env python3
"""Inspect MiniMax-H3 metadata and current h3cspeed CUDA compatibility."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

from h3_model_metadata import (
    InspectionError,
    InspectionResult,
    align_canvas_dimension,
    align_frame_count,
    geometry_report,
    inspect_path,
)
from h3_safetensors_info import DTYPE_BYTES, read_safetensors_header


def _print_human(results: Sequence[InspectionResult], geometry: Mapping[str, object]) -> None:
    for index, result in enumerate(results):
        if index:
            print()
        config = result.config
        print(f"Component: {result.component}")
        print(f"  shards/tensors: {result.shards}/{result.tensors}")
        print(f"  tensor prefix: {result.prefix or '(none)'}")
        print(f"  variant: {config.variant}")
        print(
            "  architecture: "
            f"layers={config.num_layers}, refiner={config.token_refiner_num_layers}, "
            f"hidden={config.hidden_size}, heads={config.num_attention_heads}, "
            f"head_dim={config.attention_head_dim}, ffn={config.ffn_hidden_size}"
        )
        print(
            "  modalities: "
            f"video={config.video_latent_channels}, "
            f"audio={config.audio_latent_channels}, text={config.text_dim}"
        )
        print(
            "  time/rope: "
            f"input={config.timestep_input_dim}, hidden={config.time_embed_hidden_size}, "
            f"output={config.time_embed_dim}, rope={config.rope_inv_freq_len}, "
            f"curve_grid={config.adaln_curve_grid}"
        )
        if result.compatible:
            print("  h3cspeed CUDA: compatible")
        else:
            print("  h3cspeed CUDA: incompatible")
            for issue in result.compatibility_issues:
                print(f"    - {issue}")
    if geometry:
        print("\nRequested geometry normalization:")
        for label, item in geometry.items():
            assert isinstance(item, dict)
            suffix = " (already aligned)" if not item["changed"] else ""
            print(f"  {label}: {item['requested']} -> {item['aligned']}{suffix}")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect MiniMax-H3 safetensors headers and verify compatibility "
            "with current h3cspeed CUDA kernels."
        )
    )
    parser.add_argument("path", type=Path, help="model root, component directory, or shard")
    parser.add_argument(
        "--prefix",
        help="explicit tensor namespace, for example model.diffusion_model",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--strict-h3cspeed",
        action="store_true",
        help="exit 3 when a coherent H3 checkpoint is not executable by current kernels",
    )
    parser.add_argument("--width", type=_positive_int)
    parser.add_argument("--height", type=_positive_int)
    parser.add_argument("--frames", type=_positive_int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        results = inspect_path(args.path, args.prefix)
        geometry = geometry_report(args.width, args.height, args.frames)
    except (InspectionError, ValueError) as exc:
        if args.json:
            print(json.dumps({"schema_version": 1, "error": str(exc)}, ensure_ascii=False))
        else:
            print(f"h3-model-info: {exc}", file=sys.stderr)
        return 2

    payload = {
        "schema_version": 1,
        "source": os.fspath(args.path),
        "components": [result.to_dict() for result in results],
        "geometry": geometry,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_human(results, geometry)
    if args.strict_h3cspeed and any(not result.compatible for result in results):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
