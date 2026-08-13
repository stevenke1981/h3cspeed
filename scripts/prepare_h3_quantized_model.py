#!/usr/bin/env python3
"""Validate and stage the local MiniMax H3 ComfyUI quantized FL2VA model pack.

The four safetensors files are intentionally inspected without reading their
large payloads.  Only the safetensors header and the tiny ``comfy_quant``
marker tensors are read.  A successful preparation creates hardlinks for the
four large files, copies small FL2VA configuration/tokenizer files, and emits
one manifest declaring T2V and keyframe-I2V capabilities. Existing output
directories are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_COMFY_MODELS = Path(
    os.environ.get("H3_COMFY_MODELS", r"E:\minimax-h3\ComfyUI\models")
)
DEFAULT_BASE_MODEL = Path(
    os.environ.get("H3_BASE_MODEL", r"E:\models\MiniMax-H3")
)
DEFAULT_OUTPUT = DEFAULT_COMFY_MODELS / "h3_fl2va_quantized"

PACK_KIND = "minimax-h3-comfy-fl2va-quantized-pack"
PACK_MODEL_FAMILY = "FL2VA"
PACK_CAPABILITIES = (
    "t2v",
    "fl2va_i2v_first_frame",
    "fl2va_i2v_last_frame",
    "fl2va_i2v_first_and_last_frames",
)

HEADER_LIMIT = 256 * 1024 * 1024
SMALL_FILE_LIMIT = 16 * 1024 * 1024

PACK_FILES: Mapping[str, tuple[str, str]] = {
    "fl2va": (
        "diffusion_models",
        "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    ),
    "qwen_nvfp4": (
        "text_encoders",
        "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    ),
    "video_vae": ("vae", "minimax_h3_video_vae_fp16.safetensors"),
    "audio_vae": ("vae", "minimax_h3_audio_vae_fp32.safetensors"),
}

# The native H3 loader consumes a model root with the original FL2VA
# component tree.  Keep ComfyUI's source directories above separate from the
# prepared model-root paths below; the latter are all inside ``output/base``.
PACK_COMPONENT_PATHS: Mapping[str, Path] = {
    "fl2va": Path("FL2VA/transformer"),
    "qwen_nvfp4": Path("FL2VA/text_encoder"),
    "video_vae": Path("FL2VA/video_vae/source"),
    "audio_vae": Path("FL2VA/audio_vae"),
}

# These names are deliberately an allow-list.  In particular, no source
# Python module, model shard, cache entry, or safetensors payload can enter the
# prepared root through the small-file copy path.
SMALL_CONFIG_NAMES = frozenset(
    {
        "chat_template.json",
        "config.json",
        "config.yaml",
        "merges.txt",
        "metadata.json",
        "model_index.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    }
)
REQUIRED_BASE_FILES = (
    "FL2VA/transformer/config.json",
    "FL2VA/text_encoder/config.json",
    "FL2VA/tokenizer/tokenizer.json",
    "FL2VA/tokenizer/tokenizer_config.json",
    "FL2VA/tokenizer/merges.txt",
    "FL2VA/tokenizer/vocab.json",
    "FL2VA/video_vae/config.json",
    "FL2VA/audio_vae/config.json",
)

DTYPE_BYTES: Mapping[str, int] = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


class PreparationError(RuntimeError):
    """A fail-closed validation or staging error."""

    def __init__(self, message: str, errors: Iterable[str] = ()):
        self.errors = tuple(errors) or (message,)
        super().__init__(message)


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    @property
    def byte_size(self) -> int:
        size = 1
        for value in self.shape:
            size *= value
        return size * DTYPE_BYTES[self.dtype]


@dataclass
class HeaderReport:
    path: Path
    file_size: int
    header_size: int
    data_start: int
    tensors: dict[str, TensorSpec]
    metadata: dict[str, Any]
    header_sha256: str
    dtype_counts: dict[str, int] = field(default_factory=dict)

    @property
    def payload_size(self) -> int:
        return self.file_size - self.data_start


@dataclass
class PackReport:
    models_root: Path
    base_root: Path
    files: dict[str, HeaderReport]
    small_files: list[Path]
    details: dict[str, Any] = field(default_factory=dict)


def _fail(path: Path, errors: list[str], message: str) -> None:
    errors.append(f"{path}: {message}")


def _read_exact(handle: Any, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise PreparationError(f"short read: expected {size} bytes, got {len(data)}")
    return data


def _json_object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _parse_header(path: Path) -> HeaderReport:
    """Read and structurally validate a safetensors header only."""

    errors: list[str] = []
    if not path.is_file():
        raise PreparationError(f"missing safetensors file: {path}")
    file_size = path.stat().st_size
    if file_size < 8:
        raise PreparationError(f"{path}: file is shorter than the safetensors prefix")

    try:
        with path.open("rb") as handle:
            raw_length = _read_exact(handle, 8)
            header_size = struct.unpack("<Q", raw_length)[0]
            if header_size > HEADER_LIMIT:
                raise PreparationError(
                    f"{path}: header is {header_size} bytes, above the {HEADER_LIMIT}-byte limit"
                )
            data = _read_exact(handle, header_size)
    except OSError as exc:
        raise PreparationError(f"cannot read {path}: {exc}") from exc
    except PreparationError:
        raise

    data_start = 8 + header_size
    if data_start > file_size:
        raise PreparationError(f"{path}: header extends beyond the file")
    try:
        document = json.loads(
            data.decode("utf-8"), object_pairs_hook=_json_object_no_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PreparationError(f"{path}: invalid UTF-8/JSON safetensors header: {exc}") from exc
    if not isinstance(document, dict):
        raise PreparationError(f"{path}: safetensors header must be a JSON object")

    metadata = document.get("__metadata__", {})
    if not isinstance(metadata, dict):
        _fail(path, errors, "__metadata__ must be an object when present")
        metadata = {}

    tensors: dict[str, TensorSpec] = {}
    intervals: list[tuple[int, int, str]] = []
    dtype_counts: dict[str, int] = {}
    payload_size = file_size - data_start
    for name, raw in document.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(raw, dict):
            _fail(path, errors, f"invalid tensor entry {name!r}")
            continue
        dtype = raw.get("dtype")
        shape = raw.get("shape")
        offsets = raw.get("data_offsets")
        if dtype not in DTYPE_BYTES:
            _fail(path, errors, f"{name}: unsupported or missing dtype {dtype!r}")
            continue
        if not isinstance(shape, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in shape
        ):
            _fail(path, errors, f"{name}: shape must be a list of non-negative integers")
            continue
        if not isinstance(offsets, list) or len(offsets) != 2 or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in offsets
        ):
            _fail(path, errors, f"{name}: data_offsets must be [start, end]")
            continue
        start, end = offsets
        if end < start:
            _fail(path, errors, f"{name}: data_offsets end precedes start")
            continue
        if end > payload_size:
            _fail(path, errors, f"{name}: data_offsets exceed payload ({end}>{payload_size})")
            continue
        spec = TensorSpec(name, str(dtype), tuple(shape), (start, end))
        actual_size = end - start
        expected_size = spec.byte_size
        if actual_size != expected_size:
            _fail(
                path,
                errors,
                f"{name}: byte size {actual_size} does not match {dtype}{shape} ({expected_size})",
            )
        tensors[name] = spec
        intervals.append((start, end, name))
        dtype_counts[str(dtype)] = dtype_counts.get(str(dtype), 0) + 1

    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] < previous[1]:
            _fail(path, errors, f"tensor intervals overlap: {previous[2]} and {current[2]}")

    if errors:
        raise PreparationError(f"invalid safetensors header: {path}", errors)
    return HeaderReport(
        path=path,
        file_size=file_size,
        header_size=header_size,
        data_start=data_start,
        tensors=tensors,
        metadata=metadata,
        header_sha256=hashlib.sha256(raw_length + data).hexdigest(),
        dtype_counts=dtype_counts,
    )


def read_safetensors_header(path: str | os.PathLike[str]) -> HeaderReport:
    """Public header-only reader used by tests and downstream tooling."""

    return _parse_header(Path(path))


# Short aliases keep this utility convenient for callers that do not need to
# distinguish the structural reader from the schema validator.
read_header = read_safetensors_header


def _read_small_tensor(report: HeaderReport, tensor: TensorSpec, limit: int = 4096) -> bytes:
    size = tensor.data_offsets[1] - tensor.data_offsets[0]
    if size > limit:
        raise PreparationError(
            f"{report.path}: marker {tensor.name} is {size} bytes, above the {limit}-byte limit"
        )
    with report.path.open("rb") as handle:
        handle.seek(report.data_start + tensor.data_offsets[0])
        return _read_exact(handle, size)


def _marker_json(report: HeaderReport, name: str) -> dict[str, Any]:
    tensor = report.tensors.get(name)
    if tensor is None:
        raise PreparationError(f"{report.path}: missing marker tensor {name}")
    if tensor.dtype != "U8" or len(tensor.shape) != 1:
        raise PreparationError(f"{report.path}: marker {name} must be a one-dimensional U8 tensor")
    try:
        decoded = _read_small_tensor(report, tensor).rstrip(b"\0").decode("utf-8")
        value = json.loads(decoded, object_pairs_hook=_json_object_no_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreparationError(f"{report.path}: invalid JSON marker {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreparationError(f"{report.path}: marker {name} must contain a JSON object")
    return value


def _require_tensor(
    report: HeaderReport,
    name: str,
    dtype: str | None = None,
    shape: tuple[int, ...] | None = None,
) -> TensorSpec:
    tensor = report.tensors.get(name)
    if tensor is None:
        raise PreparationError(f"{report.path}: missing required tensor {name}")
    if dtype is not None and tensor.dtype != dtype:
        raise PreparationError(f"{report.path}: {name} dtype {tensor.dtype}, expected {dtype}")
    if shape is not None and tensor.shape != shape:
        raise PreparationError(f"{report.path}: {name} shape {tensor.shape}, expected {shape}")
    return tensor


def _validate_fl2va(report: HeaderReport) -> dict[str, Any]:
    _require_tensor(report, "adaln_t_table", "F32", (1025, 8))
    _require_tensor(report, "audio_patch_proj.weight", "F32", (5376, 32))
    _require_tensor(report, "video_patch_proj.weight", "F32", (5376, 96))
    _require_tensor(report, "condition_proj.weight", "BF16", (5376, 5120))
    _require_tensor(report, "final_layer.audio_out.weight", "F32", (32, 5376))
    _require_tensor(report, "final_layer.video_out.weight", "F32", (96, 5376))
    _require_tensor(report, "rope.inv_freq", "F32", (16,))
    _require_tensor(report, "blocks.0.attn.q_norm.weight", "BF16", (128,))
    _require_tensor(report, "blocks.0.attn.k_norm.weight", "BF16", (128,))
    for layer in range(50):
        prefix = f"blocks.{layer}.adaln_proj.linear"
        _require_tensor(report, f"{prefix}.weight", "F16", (96768, 8))
        _require_tensor(report, f"{prefix}.bias", "F16", (96768,))
    _require_tensor(
        report, "final_layer.adaln_proj.linear.weight", "F16", (10752, 8)
    )
    _require_tensor(
        report, "final_layer.adaln_proj.linear.bias", "F16", (10752,)
    )

    expected: set[str] = set()
    pattern = re.compile(r"^blocks\.(\d+)\.(attn\.(?:out_proj|qkv_proj)|mlp\.(?:fc1|fc2))$")
    expected_shapes = {
        "attn.out_proj": (5376, 7168),
        "attn.qkv_proj": (21504, 5376),
        "mlp.fc1": (28672, 5376),
        "mlp.fc2": (5376, 14336),
    }
    marker_names = {
        name[: -len(".comfy_quant")]
        for name in report.tensors
        if name.endswith(".comfy_quant")
    }
    for layer in range(50):
        for projection, shape in expected_shapes.items():
            base = f"blocks.{layer}.{projection}"
            expected.add(base)
            weight = _require_tensor(report, f"{base}.weight", "I8", shape)
            scale = _require_tensor(report, f"{base}.weight_scale", "F32", (shape[0], 1))
            _ = scale
            marker = _marker_json(report, f"{base}.comfy_quant")
            if marker.get("format") != "int8_tensorwise":
                raise PreparationError(f"{report.path}: {base} marker is not int8_tensorwise")
            if marker.get("convrot") is not True:
                raise PreparationError(f"{report.path}: {base} marker does not enable ConvRot")
            if marker.get("convrot_groupsize") != 256:
                raise PreparationError(f"{report.path}: {base} ConvRot group size is not 256")
            if weight.byte_size != shape[0] * shape[1]:
                raise PreparationError(f"{report.path}: {base} INT8 shape/size mismatch")
    if marker_names != expected:
        missing = sorted(expected - marker_names)
        extra = sorted(marker_names - expected)
        raise PreparationError(
            f"{report.path}: FL2VA marker coverage mismatch; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    return {
        "format": "FL2VA ConvRot INT8",
        "layers": 50,
        "quantized_linears": len(expected),
        "convrot_groupsize": 256,
    }


QWEN_PROJECTIONS: Mapping[str, tuple[int, int]] = {
    "mlp.down_proj": (5120, 12800),
    "mlp.gate_proj": (25600, 2560),
    "mlp.up_proj": (25600, 2560),
    "self_attn.k_proj": (1024, 2560),
    "self_attn.o_proj": (5120, 4096),
    "self_attn.q_proj": (8192, 2560),
    "self_attn.v_proj": (1024, 2560),
}


def _validate_qwen(report: HeaderReport) -> dict[str, Any]:
    _require_tensor(report, "model.embed_tokens.weight", "I8", (151936, 5120))
    _require_tensor(report, "model.embed_tokens.weight_scale", "F32", (151936, 1))
    embed_marker = _marker_json(report, "model.embed_tokens.comfy_quant")
    if embed_marker.get("format") != "int8_tensorwise":
        raise PreparationError(f"{report.path}: Qwen embedding marker is not int8_tensorwise")

    expected_markers: set[str] = {"model.embed_tokens"}
    for layer in range(50):
        _require_tensor(report, f"model.layers.{layer}.input_layernorm.weight", "BF16", (5120,))
        for projection, shape in QWEN_PROJECTIONS.items():
            base = f"model.layers.{layer}.{projection}"
            expected_markers.add(base)
            weight = _require_tensor(report, f"{base}.weight", "U8", shape)
            scale = _require_tensor(
                report,
                f"{base}.weight_scale",
                "F8_E4M3",
                (shape[0], shape[1] // 8),
            )
            _require_tensor(report, f"{base}.weight_scale_2", "F32", ())
            marker = _marker_json(report, f"{base}.comfy_quant")
            if marker.get("format") != "nvfp4":
                raise PreparationError(f"{report.path}: {base} marker is not NVFP4")
            if weight.byte_size != shape[0] * shape[1]:
                raise PreparationError(f"{report.path}: {base} NVFP4 data shape/size mismatch")
            if scale.byte_size != shape[0] * (shape[1] // 8):
                raise PreparationError(f"{report.path}: {base} NVFP4 scale shape/size mismatch")

        # Comfy's Qwen conversion uses these two pre-quantization aliases.  A
        # missing alias is a load-time failure even when all NVFP4 tensors exist.
        _require_tensor(
            report,
            f"model.layers.{layer}.mlp.down_proj.pre_quant_scale",
            "BF16",
            (25600,),
        )
        _require_tensor(
            report,
            f"model.layers.{layer}.self_attn.o_proj.pre_quant_scale",
            "BF16",
            (8192,),
        )

    marker_names = {
        name[: -len(".comfy_quant")]
        for name in report.tensors
        if name.endswith(".comfy_quant")
    }
    if marker_names != expected_markers:
        missing = sorted(expected_markers - marker_names)
        extra = sorted(marker_names - expected_markers)
        raise PreparationError(
            f"{report.path}: Qwen marker coverage mismatch; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    return {
        "format": "Qwen3-VL 32B NVFP4",
        "layers": 50,
        "nvfp4_linears": 350,
        "aliases": {
            "pre_quant_scale": 100,
            "weight_scale_2": 350,
            "comfy_quant": 351,
        },
    }


def _validate_vae(report: HeaderReport, role: str) -> dict[str, Any]:
    expected_dtype = "F16" if role == "video_vae" else "F32"
    if set(report.dtype_counts) != {expected_dtype}:
        raise PreparationError(
            f"{report.path}: {role} must contain only {expected_dtype}, "
            f"got {sorted(report.dtype_counts)}"
        )
    if role == "video_vae":
        metadata_key = "minimax_h3_video_vae"
        _require_tensor(report, "latents_mean", "F16", (24,))
        _require_tensor(report, "latents_std", "F16", (24,))
    else:
        metadata_key = "minimax_h3_audio_vae"
        _require_tensor(report, "latents_mean", "F32", (32,))
        _require_tensor(report, "latents_std", "F32", (32,))
        _require_tensor(report, "dec_in_proj.weight", "F32", (2048, 32, 1))
        _require_tensor(report, "decoder.conv_pre.weight", "F32", (1024, 2048, 7))
        _require_tensor(report, "decoder.conv_pre.bias", "F32", (1024,))
    if metadata_key not in report.metadata:
        raise PreparationError(f"{report.path}: missing {metadata_key} metadata")
    return {"format": expected_dtype, "tensor_count": len(report.tensors), "metadata": metadata_key}


def validate_safetensors_header(
    path: str | os.PathLike[str], role: str | None = None
) -> HeaderReport:
    """Validate a header and, when requested, the H3 model schema."""

    report = _parse_header(Path(path))
    if role in {"fl2va", "fl2va_convrot", "fl2va_int8"}:
        _validate_fl2va(report)
    elif role in {"qwen_nvfp4", "qwen", "qwen3vl_nvfp4"}:
        _validate_qwen(report)
    elif role in {"video_vae", "audio_vae"}:
        _validate_vae(report, role)
    elif role is not None:
        raise ValueError(f"unknown H3 model role: {role}")
    return report


validate_header = validate_safetensors_header


def _collect_small_files(base_root: Path) -> list[Path]:
    fl2va = base_root / "FL2VA"
    if not fl2va.is_dir():
        raise PreparationError(f"missing base model directory: {fl2va}")
    errors: list[str] = []
    for relative in REQUIRED_BASE_FILES:
        source = base_root / Path(relative)
        if source.is_symlink():
            _fail(source, errors, "required config/tokenizer must not be a symlink")
        elif not source.is_file():
            _fail(source, errors, "required small config/tokenizer is missing")
        elif source.stat().st_size > SMALL_FILE_LIMIT:
            _fail(
                source,
                errors,
                f"required config/tokenizer exceeds {SMALL_FILE_LIMIT}-byte small-file limit",
            )
    selected: list[Path] = []
    for source in sorted(fl2va.rglob("*")):
        if (
            source.is_symlink()
            or not source.is_file()
            or source.name not in SMALL_CONFIG_NAMES
        ):
            continue
        if source.suffix.lower() == ".safetensors" or source.stat().st_size > SMALL_FILE_LIMIT:
            continue
        selected.append(source)
    if errors:
        raise PreparationError("base model small-file validation failed", errors)
    if not selected:
        raise PreparationError(f"no allow-listed small config/tokenizer files found under {fl2va}")
    return selected


def validate_pack(
    models_root: str | os.PathLike[str] = DEFAULT_COMFY_MODELS,
    base_root: str | os.PathLike[str] = DEFAULT_BASE_MODEL,
) -> PackReport:
    """Validate the four local files and the source of small config files."""

    models = Path(models_root).expanduser().resolve()
    base = Path(base_root).expanduser().resolve()
    reports: dict[str, HeaderReport] = {}
    details: dict[str, Any] = {}
    for role, (directory, filename) in PACK_FILES.items():
        source = models / directory / filename
        report = validate_safetensors_header(source, role)
        reports[role] = report
        if role == "fl2va":
            details[role] = _validate_fl2va(report)
        elif role == "qwen_nvfp4":
            details[role] = _validate_qwen(report)
        else:
            details[role] = _validate_vae(report, role)
    small = _collect_small_files(base)
    return PackReport(models, base, reports, small, details)


def _same_file(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except (FileNotFoundError, OSError):
        left_stat = left.stat()
        right_stat = right.stat()
        return left_stat.st_dev == right_stat.st_dev and left_stat.st_ino == right_stat.st_ino


def _link_large_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise PreparationError(f"refusing to overwrite {destination}")
    try:
        os.link(source, destination)
    except OSError as exc:
        raise PreparationError(f"cannot create required hardlink {destination}: {exc}") from exc
    if not _same_file(source, destination) or destination.stat().st_size != source.stat().st_size:
        raise PreparationError(f"hardlink verification failed for {destination}")


def prepare_pack(
    output_root: str | os.PathLike[str],
    models_root: str | os.PathLike[str] = DEFAULT_COMFY_MODELS,
    base_root: str | os.PathLike[str] = DEFAULT_BASE_MODEL,
) -> Path:
    """Validate and atomically create a hardlink/copy/manifest model root."""

    output = Path(output_root).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise PreparationError(f"refusing to overwrite existing output root: {output}")
    report = validate_pack(models_root, base_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise PreparationError(f"staging path already exists: {staging}")
    try:
        staging.mkdir()
        large_entries: list[dict[str, Any]] = []
        for role, (directory, filename) in PACK_FILES.items():
            source = report.models_root / directory / filename
            destination = staging / "base" / PACK_COMPONENT_PATHS[role] / filename
            _link_large_file(source, destination)
            header = report.files[role]
            large_entries.append(
                {
                    "role": role,
                    "path": str(Path("base") / PACK_COMPONENT_PATHS[role] / filename),
                    "source": str(source),
                    "mode": "hardlink",
                    "bytes": header.file_size,
                    "header_bytes": header.header_size,
                    "payload_bytes": header.payload_size,
                    "tensor_count": len(header.tensors),
                    "dtype_counts": header.dtype_counts,
                    "header_sha256": header.header_sha256,
                    "schema": report.details[role],
                }
            )

        small_entries: list[dict[str, Any]] = []
        for source in report.small_files:
            relative = source.relative_to(report.base_root)
            destination = staging / "base" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                raise PreparationError(f"refusing to overwrite {destination}")
            shutil.copyfile(source, destination)
            if destination.stat().st_size != source.stat().st_size:
                raise PreparationError(f"small-file copy verification failed for {destination}")
            small_entries.append(
                {
                    "path": str(Path("base") / relative),
                    "source": str(source),
                    "mode": "copy",
                    "bytes": source.stat().st_size,
                }
            )

        manifest = {
            "schema_version": 2,
            "kind": PACK_KIND,
            "model_family": PACK_MODEL_FAMILY,
            "capabilities": list(PACK_CAPABILITIES),
            "unsupported_model_families": ["Ref2VA"],
            "conditioning_sidecar_versions": [1, 2],
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_models_root": str(report.models_root),
            "source_base_model": str(report.base_root),
            "model_root": str(output / "base"),
            "model_root_relative": "base",
            "large_payloads_copied": False,
            "large_payloads": large_entries,
            "small_configs": small_entries,
        }
        manifest_path = staging / "manifest.json"
        with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")

        # Do not replace a path that appeared while we staged.  os.replace is
        # atomic on the same volume, and the second check keeps this operation
        # fail-closed against a concurrent creator.
        if output.exists() or output.is_symlink():
            raise PreparationError(f"refusing to overwrite output created during staging: {output}")
        os.replace(staging, output)
    except Exception:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return output


def prepare_h3_quantized_model(
    models_root: str | os.PathLike[str] = DEFAULT_COMFY_MODELS,
    base_root: str | os.PathLike[str] = DEFAULT_BASE_MODEL,
    output_root: str | os.PathLike[str] = DEFAULT_OUTPUT,
) -> Path:
    """Convenience wrapper with source roots first, matching the CLI order."""

    return prepare_pack(output_root, models_root, base_root)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-root",
        "--comfy-models",
        dest="models_root",
        type=Path,
        default=DEFAULT_COMFY_MODELS,
    )
    parser.add_argument(
        "--base-root",
        "--base-model",
        dest="base_root",
        type=Path,
        default=DEFAULT_BASE_MODEL,
    )
    parser.add_argument(
        "--output-root",
        "--output-dir",
        "--output",
        dest="output_root",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate headers and small files without creating an output root",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.validate_only:
            report = validate_pack(args.models_root, args.base_root)
            print(
                f"PASS: validated {len(report.files)} safetensors files, "
                f"{len(report.small_files)} small config/tokenizer files"
            )
        else:
            output = prepare_pack(args.output_root, args.models_root, args.base_root)
            print(f"PASS: prepared {output}")
    except PreparationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        for detail in exc.errors:
            if detail != str(exc):
                print(f"  - {detail}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
