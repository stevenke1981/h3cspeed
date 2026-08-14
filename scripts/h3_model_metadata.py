"""MiniMax-H3 architecture detection and h3cspeed compatibility checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Iterable, Mapping

from h3_safetensors_info import (
    InspectionError,
    TensorInfo,
    component_groups,
    load_component,
)

CANVAS_MULTIPLE = 32
EXPECTED: dict[str, int | str] = {
    "variant": "time-embedder",
    "hidden_size": 5376,
    "num_layers": 50,
    "token_refiner_num_layers": 2,
    "num_attention_heads": 56,
    "attention_head_dim": 128,
    "ffn_hidden_size": 14336,
    "video_latent_channels": 24,
    "audio_latent_channels": 32,
    "text_dim": 5120,
    "timestep_input_dim": 256,
    "time_embed_hidden_size": 5376,
    "time_embed_dim": 2688,
    "rope_inv_freq_len": 16,
}
ANCHOR_SUFFIXES = (
    "video_patch_proj.weight",
    "audio_patch_proj.weight",
    "blocks.0.attn.q_norm.weight",
    "blocks.0.attn.qkv_proj.weight",
    "blocks.0.mlp.fc1.weight",
    "condition_proj.weight",
    "rope.inv_freq",
)


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 5376
    num_layers: int = 50
    token_refiner_num_layers: int = 2
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    video_latent_channels: int = 24
    audio_latent_channels: int = 32
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    rope_inv_freq_len: int = 16
    adaln_curve_grid: int = 0
    variant: str = "unknown"


@dataclass(frozen=True)
class InspectionResult:
    component: str
    shards: int
    tensors: int
    prefix: str
    config: ModelConfig
    compatible: bool
    compatibility_issues: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "component": self.component,
            "shards": self.shards,
            "tensors": self.tensors,
            "prefix": self.prefix,
            "config": asdict(self.config),
            "h3cspeed": {
                "compatible": self.compatible,
                "issues": list(self.compatibility_issues),
            },
        }


def _qualified(prefix: str, suffix: str) -> str:
    return f"{prefix}.{suffix}" if prefix else suffix


def _candidate_prefixes(names: Iterable[str]) -> set[str]:
    prefixes: set[str] = set()
    suffix = "video_patch_proj.weight"
    for name in names:
        if name == suffix:
            prefixes.add("")
        elif name.endswith("." + suffix):
            prefixes.add(name[: -(len(suffix) + 1)])
    return prefixes


def _block_indices(names: Iterable[str], prefix: str, block_path: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(_qualified(prefix, block_path))}(\d+)\.")
    return sorted({int(match.group(1)) for name in names if (match := pattern.match(name))})


def _prefix_score(tensors: Mapping[str, TensorInfo], prefix: str) -> tuple[int, int]:
    anchors = sum(_qualified(prefix, suffix) in tensors for suffix in ANCHOR_SUFFIXES)
    return anchors, len(_block_indices(tensors, prefix, "blocks."))


def detect_prefix(tensors: Mapping[str, TensorInfo], requested: str | None = None) -> str:
    if requested is not None:
        prefix = requested.rstrip(".")
        if _prefix_score(tensors, prefix)[0] < 2:
            raise InspectionError(f"prefix {requested!r} does not identify an H3 component")
        return prefix
    scored = sorted(
        ((_prefix_score(tensors, prefix), prefix) for prefix in _candidate_prefixes(tensors)),
        reverse=True,
    )
    if not scored or scored[0][0][0] < 4:
        raise InspectionError("no MiniMax-H3 transformer metadata found")
    best_score = scored[0][0]
    best = [prefix for score, prefix in scored if score == best_score]
    if len(best) != 1:
        labels = ", ".join(repr(prefix) for prefix in sorted(best))
        raise InspectionError(f"ambiguous H3 tensor prefixes: {labels}")
    return best[0]


def _required(tensors: Mapping[str, TensorInfo], prefix: str, suffix: str) -> TensorInfo:
    name = _qualified(prefix, suffix)
    tensor = tensors.get(name)
    if tensor is None:
        raise InspectionError(f"required tensor is absent: {name}")
    return tensor


def _shape(tensor: TensorInfo, rank: int) -> tuple[int, ...]:
    if len(tensor.shape) != rank or any(value <= 0 for value in tensor.shape):
        raise InspectionError(
            f"tensor {tensor.name} has shape {list(tensor.shape)}, expected rank {rank}"
        )
    return tensor.shape


def _contiguous_count(
    tensors: Mapping[str, TensorInfo], prefix: str, block_path: str, *, required: bool
) -> int:
    indices = _block_indices(tensors, prefix, block_path)
    if not indices:
        if required:
            raise InspectionError(f"no tensors found below {_qualified(prefix, block_path)}")
        return 0
    expected = list(range(indices[-1] + 1))
    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        raise InspectionError(
            f"block indices below {_qualified(prefix, block_path)} are not contiguous; "
            f"missing {missing[0]}"
        )
    return len(indices)


def detect_config(tensors: Mapping[str, TensorInfo], prefix: str) -> ModelConfig:
    video_patch = _shape(_required(tensors, prefix, "video_patch_proj.weight"), 2)
    if video_patch[1] % 4:
        raise InspectionError("video patch projection input is not divisible by 2x2")
    hidden_size = video_patch[0]

    audio_tensor = _required(tensors, prefix, "audio_patch_proj.weight")
    audio_patch = _shape(audio_tensor, 2)
    if audio_patch[0] != hidden_size:
        raise InspectionError(f"tensor {audio_tensor.name} output does not match hidden size")

    head_dim = _shape(_required(tensors, prefix, "blocks.0.attn.q_norm.weight"), 1)[0]
    qkv_tensor = _required(tensors, prefix, "blocks.0.attn.qkv_proj.weight")
    qkv = _shape(qkv_tensor, 2)
    if qkv[1] != hidden_size:
        raise InspectionError(f"tensor {qkv_tensor.name} input does not match hidden size")
    divisor = 3 * head_dim
    if qkv[0] % divisor:
        raise InspectionError(f"tensor {qkv_tensor.name} output is not divisible by 3 * head_dim")

    fc1_tensor = _required(tensors, prefix, "blocks.0.mlp.fc1.weight")
    fc1 = _shape(fc1_tensor, 2)
    if fc1[1] != hidden_size or fc1[0] % 2:
        raise InspectionError(f"tensor {fc1_tensor.name} is not a fused SwiGLU projection")

    condition_tensor = _required(tensors, prefix, "condition_proj.weight")
    condition = _shape(condition_tensor, 2)
    if condition[0] != hidden_size:
        raise InspectionError(f"tensor {condition_tensor.name} output does not match hidden size")

    rope = _shape(_required(tensors, prefix, "rope.inv_freq"), 1)
    values: dict[str, int | str] = {
        "hidden_size": hidden_size,
        "num_layers": _contiguous_count(tensors, prefix, "blocks.", required=True),
        "token_refiner_num_layers": _contiguous_count(
            tensors, prefix, "token_refiner.blocks.", required=False
        ),
        "num_attention_heads": qkv[0] // divisor,
        "attention_head_dim": head_dim,
        "ffn_hidden_size": fc1[0] // 2,
        "video_latent_channels": video_patch[1] // 4,
        "audio_latent_channels": audio_patch[1],
        "text_dim": condition[1],
        "timestep_input_dim": 256,
        "time_embed_hidden_size": 5376,
        "time_embed_dim": 2688,
        "rope_inv_freq_len": rope[0],
        "adaln_curve_grid": 0,
        "variant": "unknown",
    }
    adaln = tensors.get(_qualified(prefix, "adaln_t_table"))
    if adaln is not None:
        grid, dimension = _shape(adaln, 2)
        values.update(
            variant="adaln-curves",
            adaln_curve_grid=grid,
            time_embed_dim=dimension,
        )
    else:
        proj_in = _shape(_required(tensors, prefix, "time_embedder.proj_in.weight"), 2)
        proj_out = _shape(_required(tensors, prefix, "time_embedder.proj_out.weight"), 2)
        if proj_out[1] != proj_in[0]:
            raise InspectionError("time embedder projections have inconsistent widths")
        values.update(
            variant="time-embedder",
            timestep_input_dim=proj_in[1],
            time_embed_hidden_size=proj_in[0],
            time_embed_dim=proj_out[0],
        )
    return ModelConfig(**values)  # type: ignore[arg-type]


def compatibility_issues(config: ModelConfig) -> tuple[str, ...]:
    return tuple(
        f"{field}={getattr(config, field)} (expected {expected})"
        for field, expected in EXPECTED.items()
        if getattr(config, field) != expected
    )


def inspect_path(path: Path, prefix: str | None = None) -> list[InspectionResult]:
    results: list[InspectionResult] = []
    candidate_errors: list[str] = []
    for directory, shards in component_groups(path):
        tensors = load_component(shards)
        try:
            detected_prefix = detect_prefix(tensors, prefix)
        except InspectionError as exc:
            if prefix is not None or _candidate_prefixes(tensors):
                candidate_errors.append(f"{directory}: {exc}")
            continue
        try:
            config = detect_config(tensors, detected_prefix)
        except InspectionError as exc:
            candidate_errors.append(f"{directory}: {exc}")
            continue
        issues = compatibility_issues(config)
        results.append(
            InspectionResult(
                str(directory), len(shards), len(tensors), detected_prefix,
                config, not issues, issues,
            )
        )
    if candidate_errors:
        raise InspectionError("; ".join(candidate_errors))
    if not results:
        raise InspectionError(f"no coherent MiniMax-H3 component found below {path}")
    return results


def align_canvas_dimension(requested: int) -> int:
    if requested < 1:
        raise ValueError("canvas dimensions must be positive")
    return max(
        CANVAS_MULTIPLE,
        ((requested + CANVAS_MULTIPLE - 1) // CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
    )


def align_frame_count(requested: int) -> int:
    value = max(requested, 5)
    remainder = (value - 5) % 17
    return value if remainder == 0 else value + 17 - remainder


def geometry_report(
    width: int | None, height: int | None, frames: int | None
) -> dict[str, dict[str, int | bool]]:
    report: dict[str, dict[str, int | bool]] = {}
    for label, requested, aligner in (
        ("width", width, align_canvas_dimension),
        ("height", height, align_canvas_dimension),
        ("frames", frames, align_frame_count),
    ):
        if requested is not None:
            aligned = aligner(requested)
            report[label] = {
                "requested": requested,
                "aligned": aligned,
                "changed": aligned != requested,
            }
    return report
