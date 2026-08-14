"""Header-only safetensors inventory helpers for h3cspeed tooling."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

MAX_HEADER_BYTES = 256 * 1024 * 1024
MAX_DIMS = 8

DTYPE_BYTES: dict[str, int] = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


class InspectionError(RuntimeError):
    """Raised for malformed, ambiguous, or unsupported metadata layouts."""


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    source: str


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _read_exact(stream: object, size: int, label: str) -> bytes:
    data = stream.read(size)  # type: ignore[attr-defined]
    if len(data) != size:
        raise InspectionError(f"truncated {label}")
    return data


def read_safetensors_header(path: Path) -> dict[str, TensorInfo]:
    """Read and validate one safetensors header without mapping its payload."""

    try:
        file_size = path.stat().st_size
        with path.open("rb") as stream:
            raw_length = _read_exact(stream, 8, f"safetensors length in {path}")
            header_size = int.from_bytes(raw_length, "little", signed=False)
            if header_size <= 0 or header_size > MAX_HEADER_BYTES:
                raise InspectionError(
                    f"invalid safetensors header size {header_size} in {path}"
                )
            if 8 + header_size > file_size:
                raise InspectionError(f"safetensors header exceeds file size: {path}")
            raw_header = _read_exact(stream, header_size, f"safetensors header in {path}")
    except OSError as exc:
        raise InspectionError(f"cannot read {path}: {exc}") from exc

    try:
        document = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InspectionError(f"invalid safetensors JSON in {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise InspectionError(f"safetensors root is not an object: {path}")

    payload_size = file_size - 8 - header_size
    result: dict[str, TensorInfo] = {}
    for name, metadata in document.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not name:
            raise InspectionError(f"invalid tensor name in {path}")
        if not isinstance(metadata, dict):
            raise InspectionError(f"tensor {name} metadata is not an object in {path}")
        dtype = metadata.get("dtype")
        shape = metadata.get("shape")
        offsets = metadata.get("data_offsets")
        if not isinstance(dtype, str) or not dtype:
            raise InspectionError(f"tensor {name} has no dtype in {path}")
        if (
            not isinstance(shape, list)
            or len(shape) > MAX_DIMS
            or any(not isinstance(value, int) or value < 0 for value in shape)
        ):
            raise InspectionError(f"tensor {name} has an invalid shape in {path}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or offsets[1] > payload_size
        ):
            raise InspectionError(f"tensor {name} has invalid data offsets in {path}")
        if dtype in DTYPE_BYTES:
            expected_bytes = _product(shape) * DTYPE_BYTES[dtype]
            if offsets[1] - offsets[0] != expected_bytes:
                raise InspectionError(
                    f"tensor {name} byte range does not match {dtype} shape in {path}"
                )
        if name in result:
            raise InspectionError(f"duplicate tensor {name} inside {path}")
        result[name] = TensorInfo(name, dtype, tuple(shape), str(path))
    return result


def component_groups(path: Path) -> list[tuple[Path, list[Path]]]:
    if path.is_file():
        if path.suffix != ".safetensors":
            raise InspectionError(f"expected a .safetensors file: {path}")
        return [(path.parent, [path])]
    if not path.is_dir():
        raise InspectionError(f"model path does not exist or is not readable: {path}")

    grouped: dict[Path, list[Path]] = {}
    try:
        for shard in path.rglob("*.safetensors"):
            if shard.is_file():
                grouped.setdefault(shard.parent, []).append(shard)
    except OSError as exc:
        raise InspectionError(f"cannot enumerate {path}: {exc}") from exc
    if not grouped:
        raise InspectionError(f"no safetensors files found below {path}")
    return [
        (directory, sorted(shards, key=lambda item: item.name))
        for directory, shards in sorted(grouped.items(), key=lambda item: str(item[0]))
    ]


def load_component(shards: Sequence[Path]) -> dict[str, TensorInfo]:
    tensors: dict[str, TensorInfo] = {}
    for shard in shards:
        for name, tensor in read_safetensors_header(shard).items():
            previous = tensors.get(name)
            if previous is not None:
                raise InspectionError(
                    f"duplicate tensor {name} across {previous.source} and {tensor.source}"
                )
            tensors[name] = tensor
    return tensors
