#!/usr/bin/env python3
"""Export GPU-produced ComfyUI Qwen conditioning for the native H3 path.

This is deliberately a sidecar bridge: ComfyUI owns quantized Qwen loading and
execution, while h3cspeed consumes the raw ``conditioning[0][0]`` tensor.  The
script fails closed when CUDA, ComfyUI, or the requested quantized encoder is
not available; it never moves model execution to CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

MAGIC = b"H3CSEV01"
VERSION = 1
HEADER_SIZE = 128
WIDTH = 5120
FLAGS_TAGS = 1
RECIPE = "h3cspeed-conditioning-v1"
MAX_RECIPE_BYTES = 65536
HASH_CHUNK_BYTES = 8 * 1024 * 1024


class SidecarError(RuntimeError):
    """A fail-closed configuration, model, or output error."""


def _resolve_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise SidecarError(f"{label} does not exist or is not a file: {path}")
    return path


def _model_sha256(path: Path) -> bytes:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(HASH_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise SidecarError(f"cannot hash text encoder: {exc}") from exc
    return digest.digest()


def _token_ids(path: Path, prompt: str) -> list[int]:
    try:
        tokenizers = importlib.import_module("tokenizers")
    except ImportError as exc:
        raise SidecarError("tokenizers is required for the configured tokenizer") from exc
    try:
        tokenizer = tokenizers.Tokenizer.from_file(str(path))
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        ids = [int(value) for value in encoded.ids]
    except Exception as exc:  # tokenizers exposes backend-specific exceptions
        raise SidecarError(f"cannot encode prompt with tokenizer: {exc}") from exc
    if not ids:
        raise SidecarError("tokenizer produced no token IDs")
    if any(value < 0 or value > 0xFFFFFFFF for value in ids):
        raise SidecarError("tokenizer returned an out-of-range token ID")
    return ids


def _load_comfy_clip(comfy_root: Path, encoder: Path, device: str) -> Any:
    if str(comfy_root) not in sys.path:
        sys.path.insert(0, str(comfy_root))
    try:
        torch = importlib.import_module("torch")
        model_management = importlib.import_module("comfy.model_management")
        sd = importlib.import_module("comfy.sd")
    except Exception as exc:
        raise SidecarError(f"cannot import ComfyUI GPU runtime: {exc}") from exc
    if not torch.cuda.is_available():
        raise SidecarError("CUDA is unavailable; quantized Qwen CPU fallback is forbidden")
    if not device.startswith("cuda"):
        raise SidecarError("--device must be a CUDA device; CPU fallback is forbidden")
    try:
        selected = torch.device(device)
        if selected.type != "cuda":
            raise SidecarError("resolved device is not CUDA")
        torch.cuda.set_device(selected)
        # Comfy's model-management layer is the authoritative device route.
        actual = model_management.get_torch_device()
        if getattr(actual, "type", None) != "cuda":
            raise SidecarError("ComfyUI selected a non-CUDA torch device")
        loader = sd.load_clip
        parameters = inspect.signature(loader).parameters
        kwargs: dict[str, Any] = {"ckpt_paths": [str(encoder)]}
        if "clip_type" not in parameters:
            raise SidecarError("ComfyUI load_clip lacks required clip_type support")
        clip_type = getattr(getattr(sd, "CLIPType", None), "MINIMAX", None)
        if clip_type is None:
            raise SidecarError("ComfyUI does not expose CLIPType.MINIMAX")
        kwargs["clip_type"] = clip_type
        if "embedding_directory" in parameters:
            kwargs["embedding_directory"] = None
        clip = loader(**kwargs)
    except SidecarError:
        raise
    except Exception as exc:
        raise SidecarError(
            f"ComfyUI could not load the configured quantized Qwen encoder on {device}: {exc}"
        ) from exc
    if clip is None:
        raise SidecarError("ComfyUI returned no text encoder")
    return clip, torch


def _extract_token_ids(tokens: Any) -> list[int]:
    """Extract Comfy token IDs without imposing a second tokenizer template."""
    candidate = tokens
    if isinstance(tokens, dict):
        for key in ("qwen3vl_32b", "input_ids", "tokens", "l"):
            # Comfy Qwen3-VL token batches commonly expose only the model key.
            if key in tokens:
                candidate = tokens[key]
                break
    while hasattr(candidate, "tolist"):
        candidate = candidate.tolist()
    if isinstance(candidate, tuple):
        candidate = list(candidate)
    if not isinstance(candidate, list) or not candidate:
        raise SidecarError("ComfyUI tokenizer returned no usable token IDs")
    if isinstance(candidate[0], (list, tuple)):
        # A batch is commonly [[ids...]]; token/weight pairs are commonly
        # [[id, weight], ...].  Normalize the batch wrapper first, then take
        # the first member of each pair when rows remain.
        if len(candidate) == 1 and isinstance(candidate[0], (list, tuple)):
            candidate = list(candidate[0])
        if candidate and isinstance(candidate[0], (list, tuple)):
            candidate = [row[0] for row in candidate if row]
    try:
        ids = [int(row[0] if isinstance(row, (list, tuple)) else row) for row in candidate]
    except (TypeError, ValueError, IndexError) as exc:
        raise SidecarError(f"cannot extract ComfyUI token IDs: {exc}") from exc
    if any(value < 0 or value > 0xFFFFFFFF for value in ids):
        raise SidecarError("ComfyUI returned an out-of-range token ID")
    return ids


def _encode_raw_conditioning(clip: Any, torch: Any, prompt: str, device: str) -> tuple[list[int], Any, list[int]]:
    selected = torch.device(device)
    torch.cuda.reset_peak_memory_stats(selected)
    try:
        # Use Comfy's own template/tokenizer and raw conditioning result.  Do
        # not apply H3 projection or any CPU-side replacement here.
        tokens = clip.tokenize(prompt)
        ids = _extract_token_ids(tokens)
        result = clip.encode_from_tokens(tokens, return_dict=True)
        if not isinstance(result, dict) or "cond" not in result:
            raise SidecarError("ComfyUI encode_from_tokens did not return cond")
        raw = result["cond"]
        raw_tags = result.get("minimax_token_tags")
    except Exception as exc:
        raise SidecarError(f"ComfyUI Qwen conditioning failed: {exc}") from exc
    if not isinstance(raw, torch.Tensor):
        raise SidecarError("ComfyUI conditioning[0][0] is not a torch tensor")
    # DynamicVRAM may return the final tensor in host RAM immediately after
    # executing Qwen on CUDA. Prove GPU execution through allocator telemetry
    # instead of requiring the returned view to remain device-resident.
    peak_cuda = int(torch.cuda.max_memory_allocated(selected))
    if peak_cuda < 256 * 1024 * 1024:
        raise SidecarError(
            "ComfyUI did not allocate enough CUDA memory to prove GPU Qwen execution"
        )
    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]
    if raw.ndim != 2 or int(raw.shape[-1]) != WIDTH:
        raise SidecarError(
            f"raw conditioning shape must be [tokens,{WIDTH}], got {tuple(raw.shape)}"
        )
    if int(raw.shape[0]) <= 0:
        raise SidecarError("raw conditioning contains no tokens")
    if raw_tags is None:
        raw_tags = [1] * int(raw.shape[0])
    while hasattr(raw_tags, "tolist"):
        raw_tags = raw_tags.tolist()
    if isinstance(raw_tags, list) and raw_tags and isinstance(raw_tags[0], list):
        raw_tags = raw_tags[0]
    if not isinstance(raw_tags, list) or len(raw_tags) != int(raw.shape[0]):
        raise SidecarError("ComfyUI minimax_token_tags length does not match conditioning")
    tags = [int(value) for value in raw_tags]
    if any(value not in (0, 1) for value in tags):
        raise SidecarError("ComfyUI minimax_token_tags contains values other than 0/1")
    # BF16 export may occur after DynamicVRAM has returned the completed
    # conditioning to host RAM; Qwen execution itself was proven above.
    try:
        values = raw.detach().to(dtype=torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy()
        return ids, values, tags
    except Exception as exc:
        raise SidecarError(f"cannot export CUDA BF16 conditioning: {exc}") from exc


def _write_sidecar(output: Path, prompt: str, ids: list[int], values: Any,
                   model_hash: bytes, tags: list[int]) -> None:
    prompt_bytes = prompt.encode("utf-8")
    recipe_bytes = RECIPE.encode("utf-8")
    if len(recipe_bytes) == 0 or len(recipe_bytes) > MAX_RECIPE_BYTES:
        raise SidecarError("conditioning recipe length is invalid")
    if values.ndim != 2 or int(values.shape[1]) != WIDTH or int(values.shape[0]) != len(ids):
        raise SidecarError("conditioning/token count mismatch")
    embedding = values.tobytes(order="C")
    token_payload = b"".join(struct.pack("<I", value) for value in ids)
    if len(tags) != len(ids) or any(value not in (0, 1) for value in tags):
        raise SidecarError("conditioning tags must be one 0/1 value per token")
    tag_payload = bytes(tags)
    header = struct.pack(
        "<8sIIQQQIIQQQ32s24s",
        MAGIC, VERSION, HEADER_SIZE, len(prompt_bytes), len(ids), len(recipe_bytes),
        WIDTH, FLAGS_TAGS, len(embedding), len(tag_payload), len(token_payload), model_hash,
        bytes(24),
    )
    if len(header) != HEADER_SIZE:
        raise SidecarError("internal sidecar header size mismatch")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{output.name}.", suffix=".tmp", dir=output.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            temporary.write(header)
            temporary.write(prompt_bytes)
            temporary.write(recipe_bytes)
            temporary.write(token_payload)
            temporary.write(embedding)
            temporary.write(tag_payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui", default=os.environ.get("H3_COMFYUI"), required=False,
                        help="ComfyUI root (or H3_COMFYUI)")
    parser.add_argument("--text-encoder", default=os.environ.get("H3_TEXT_ENCODER"),
                        required=False, help="quantized Qwen safetensors path")
    parser.add_argument("--tokenizer", default=os.environ.get("H3_TOKENIZER"),
                        required=False, help="optional diagnostic tokenizer.json path")
    parser.add_argument("--output", default=os.environ.get("H3_CONDITIONING_OUTPUT"),
                        required=False, help="atomic sidecar output path")
    parser.add_argument("--prompt", default=os.environ.get("H3_PROMPT"),
                        required=False, help="exact UTF-8 prompt (or H3_PROMPT)")
    parser.add_argument("--device", default=os.environ.get("H3_CUDA_DEVICE", "cuda:0"),
                        help="indexed CUDA device, e.g. cuda:0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        required = {
            "--comfyui": args.comfyui,
            "--text-encoder": args.text_encoder,
            "--output": args.output,
            "--prompt": args.prompt,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise SidecarError(f"missing required configuration: {', '.join(missing)}")
        comfy_root = Path(args.comfyui).expanduser().resolve()
        if not comfy_root.is_dir():
            raise SidecarError(f"ComfyUI root is not a directory: {comfy_root}")
        encoder = _resolve_file(args.text_encoder, "text encoder")
        tokenizer = _resolve_file(args.tokenizer, "tokenizer") if args.tokenizer else None
        model_hash = _model_sha256(encoder)
        clip, torch = _load_comfy_clip(comfy_root, encoder, args.device)
        ids, values, tags = _encode_raw_conditioning(clip, torch, args.prompt, args.device)
        if tokenizer is not None:
            diagnostic_ids = _token_ids(tokenizer, args.prompt)
            if diagnostic_ids != ids:
                raise SidecarError("diagnostic tokenizer IDs differ from ComfyUI token IDs")
        if int(values.shape[0]) != len(ids):
            raise SidecarError(
                f"ComfyUI token count {int(values.shape[0])} differs from conditioning token IDs {len(ids)}"
            )
        _write_sidecar(Path(args.output), args.prompt, ids, values, model_hash, tags)
        print(
            f"conditioning-sidecar path={Path(args.output).expanduser().resolve()} "
            f"tokens={len(ids)} shape=({len(ids)},{WIDTH}) "
            f"model_sha256={model_hash.hex()} device={args.device}"
        )
        return 0
    except SidecarError as exc:
        print(f"encode_h3_quantized_prompt: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
