#!/usr/bin/env python3
"""Export GPU-produced ComfyUI Qwen conditioning for the native H3 path.

This is deliberately a sidecar bridge: ComfyUI owns quantized Qwen loading and
execution, while h3cspeed consumes the raw final conditioning tensor. Version 2
also exports Comfy's image-aware FL2VA token expansion and binds the result to
the exact first/last keyframe bytes and render geometry. The script fails
closed when CUDA, ComfyUI, or the requested quantized encoder is not available;
it never moves model execution to CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import io
import os
import subprocess
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

MAGIC = b"H3CSEV01"
VERSION = 2
HEADER_SIZE = 128
WIDTH = 5120
FLAGS_TAGS = 1
RECIPE = "h3cspeed-conditioning-v2"
VISION_START = 151652
VISION_END = 151653
VISION_PAD = 151655
MODE_T2V = 0
MODE_FL2VA_I2V = 1
ROLE_FIRST = 1
ROLE_LAST = 2
RESIZE_STRETCH = 0
RESIZE_COVER = 1
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


def _extract_token_batch(tokens: Any) -> list[Any]:
    """Extract one Comfy token-weight sequence, retaining image entries."""
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
        # A batch is commonly [[(id, weight), ...]].
        if len(candidate) == 1 and isinstance(candidate[0], (list, tuple)):
            candidate = list(candidate[0])
    return candidate


def _image_grid(height: int, width: int) -> tuple[int, int]:
    """Mirror Comfy Qwen3-VL patch geometry (patch 16, merge 2)."""
    factor = 32
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    min_pixels = 3136
    max_pixels = 12845056
    if h_bar * w_bar > max_pixels:
        beta = ((height * width) / max_pixels) ** 0.5
        h_bar = max(factor, int(height / beta // factor) * factor)
        w_bar = max(factor, int(width / beta // factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = (min_pixels / (height * width)) ** 0.5
        import math
        h_bar = int(math.ceil(height * beta / factor) * factor)
        w_bar = int(math.ceil(width * beta / factor) * factor)
    return h_bar // 16, w_bar // 16


def _expanded_token_ids(entries: list[Any]) -> list[int]:
    """Expand Comfy image dict entries to the IDs emitted by MiniMax H3."""
    ids: list[int] = []
    for entry in entries:
        token = entry[0] if isinstance(entry, (list, tuple)) and entry else entry
        if isinstance(token, dict):
            if token.get("type") != "image" or "data" not in token:
                raise SidecarError("Comfy tokenizer returned an unsupported visual entry")
            data = token["data"]
            shape = getattr(data, "shape", None)
            if shape is None or len(shape) != 4 or int(shape[0]) != 1:
                raise SidecarError("Comfy image entry must have shape [1,H,W,3]")
            height, width = int(shape[1]), int(shape[2])
            grid_h, grid_w = _image_grid(height, width)
            patch_count = (grid_h * grid_w) // 4
            if patch_count <= 0:
                raise SidecarError("Comfy image entry has no visual patches")
            # MiniMax's tokenizer already surrounds the image dictionary with
            # VISION_START/VISION_END.  The dictionary itself expands only to
            # the merged visual patch span.
            ids.extend([VISION_PAD] * patch_count)
            continue
        try:
            value = int(token)
        except (TypeError, ValueError) as exc:
            raise SidecarError(f"cannot extract ComfyUI token ID: {exc}") from exc
        if value < 0 or value > 0xFFFFFFFF:
            raise SidecarError("ComfyUI returned an out-of-range token ID")
        ids.append(value)
    return ids


def _load_image_tensor(data: bytes, label: Path, torch: Any) -> Any:
    try:
        from PIL import Image, ImageOps
        import numpy as np
        with Image.open(io.BytesIO(data)) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            values = np.asarray(rgb, dtype=np.float32) / 255.0
        return torch.from_numpy(values).unsqueeze(0).contiguous()
    except Exception as exc:
        raise SidecarError(f"cannot load keyframe image {label}: {exc}") from exc


def _canonicalize_keyframe(source: Path, destination: Path, *, is_first: bool,
                           width: int, height: int) -> Path:
    """Create the exact RGB keyframe consumed by both Comfy and native H3."""
    ffmpeg = os.environ.get("H3_FFMPEG", "ffmpeg")
    source_dimensions: tuple[int, int] | None = None
    if is_first:
        try:
            from PIL import Image
            with Image.open(source) as image:
                source_dimensions = tuple(int(value) for value in image.size)
        except Exception as exc:
            raise SidecarError(f"cannot inspect first-frame dimensions: {exc}") from exc
    # Matrix profiles bind an exact-size PNG.  Use a format-only conversion for
    # that case so the H3 canonical first frame has the same pixels instead of
    # passing through a nominal scale filter.  Legacy/non-exact inputs retain
    # the existing resize/crop behavior and are not part of the no-stretch gate.
    if is_first and source_dimensions == (width, height):
        filter_graph = "format=rgb24"
    elif is_first:
        filter_graph = f"scale={width}:{height}:flags=lanczos"
    else:
        filter_graph = (f"scale={width}:{height}:force_original_aspect_ratio=increase:"
                        f"flags=lanczos,crop={width}:{height}")
    destination = destination.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise SidecarError("canonical keyframe destination must not be a symlink")
    staged_path: Path | None = None
    try:
        # FFmpeg writes only inside the OS per-user temporary directory. This
        # prevents a shared output directory from pre-planting a symlink at
        # FFmpeg's -y target. A separately exclusive staging file is atomically
        # installed at the requested destination afterwards.
        with tempfile.TemporaryDirectory(prefix="h3cspeed-keyframe-") as directory:
            temporary = Path(directory) / "keyframe.png"
            completed = subprocess.run(
                [ffmpeg, "-v", "error", "-nostdin", "-y", "-i", str(source),
                 "-frames:v", "1", "-vf", filter_graph, "-pix_fmt", "rgb24",
                 str(temporary)],
                check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True,
            )
            if completed.returncode != 0 or not temporary.is_file() or temporary.is_symlink():
                detail = completed.stderr.strip() or f"exit code {completed.returncode}"
                raise SidecarError(f"FFmpeg canonical keyframe failed: {detail}")
            canonical_bytes = temporary.read_bytes()
        with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{destination.name}.", suffix=".png",
                dir=destination.parent, delete=False) as staged:
            staged_path = Path(staged.name)
            staged.write(canonical_bytes)
            staged.flush()
            os.fsync(staged.fileno())
        os.replace(staged_path, destination)
        staged_path = None
    except OSError as exc:
        raise SidecarError(f"cannot create canonical keyframe: {exc}") from exc
    finally:
        if staged_path is not None:
            staged_path.unlink(missing_ok=True)
    return destination


def _encode_raw_conditioning(clip: Any, torch: Any, prompt: str, device: str,
                             images: list[Any]) -> tuple[list[int], Any, list[int]]:
    selected = torch.device(device)
    torch.cuda.reset_peak_memory_stats(selected)
    try:
        # Use Comfy's own template/tokenizer and raw conditioning result.  Do
        # not apply H3 projection or any CPU-side replacement here.
        # MiniMax's image-aware tokenizer owns the Picture prefix and visual
        # token insertion. Do not synthesize a second prompt template.
        tokens = clip.tokenize(prompt, images=images)
        entries = _extract_token_batch(tokens)
        ids = _expanded_token_ids(entries)
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
                   model_hash: bytes, tags: list[int], *, mode: int,
                   keyframe_role: int, render_width: int,
                   render_height: int, image_hashes: list[bytes]) -> None:
    if output.suffix.lower() != ".h3c":
        raise SidecarError("conditioning sidecar output must end in .h3c")
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
    keyframe_count = len(image_hashes)
    if mode == MODE_T2V and keyframe_count:
        raise SidecarError("T2V sidecar cannot contain keyframe hashes")
    if mode == MODE_FL2VA_I2V and keyframe_count not in (1, 2):
        raise SidecarError("FL2VA I2V requires one or two keyframe images")
    if mode == MODE_FL2VA_I2V and (render_width < 64 or render_height < 64):
        raise SidecarError("I2V render width and height must both be at least 64")
    if mode == MODE_FL2VA_I2V and (render_width % 32 or render_height % 32):
        raise SidecarError("I2V render width and height must be divisible by 32")
    expected_role = {0: 0, 1: keyframe_role, 2: ROLE_FIRST | ROLE_LAST}.get(keyframe_count)
    if (keyframe_count > 2 or expected_role != keyframe_role or
            (keyframe_count == 0 and mode != MODE_T2V) or
            (keyframe_count > 0 and mode != MODE_FL2VA_I2V) or
            (keyframe_count > 0 and keyframe_role not in (ROLE_FIRST, ROLE_LAST, ROLE_FIRST | ROLE_LAST))):
        raise SidecarError("keyframe role/order metadata is invalid")
    metadata = b"".join(image_hashes)
    if len(metadata) != keyframe_count * 32:
        raise SidecarError("keyframe SHA-256 metadata is invalid")
    reserved = bytearray(24)
    reserved[0] = mode
    reserved[1] = keyframe_role
    reserved[2] = keyframe_count
    reserved[3] = keyframe_role  # canonical order: first, then last
    reserved[4] = RESIZE_STRETCH
    reserved[5] = RESIZE_COVER
    struct.pack_into("<II", reserved, 8, render_width if mode else 0,
                     render_height if mode else 0)
    struct.pack_into("<I", reserved, 16, len(metadata))
    header = struct.pack(
        "<8sIIQQQIIQQQ32s24s",
        MAGIC, VERSION, HEADER_SIZE, len(prompt_bytes), len(ids), len(recipe_bytes),
        WIDTH, FLAGS_TAGS, len(embedding), len(tag_payload), len(token_payload), model_hash,
        bytes(reserved),
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
            temporary.write(metadata)
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
    parser.add_argument("--mode", choices=("t2v", "fl2va-i2v"), default="t2v")
    parser.add_argument("--first-frame", help="FL2VA first keyframe image")
    parser.add_argument("--last-frame", help="FL2VA last keyframe image")
    parser.add_argument("--width", type=int, default=0, help="native render width")
    parser.add_argument("--height", type=int, default=0, help="native render height")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.device = args.device.lower()
        device_parts = args.device.split(":", 1)
        if (device_parts[0] != "cuda" or
                (len(device_parts) == 2 and not device_parts[1].isdigit())):
            raise SidecarError("device must be cuda or cuda:N; CPU fallback is forbidden")
        required = {
            "--comfyui": args.comfyui,
            "--text-encoder": args.text_encoder,
            "--output": args.output,
            "--prompt": args.prompt,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise SidecarError(f"missing required configuration: {', '.join(missing)}")
        if Path(args.output).suffix.lower() != ".h3c":
            raise SidecarError("conditioning sidecar output must end in .h3c")
        comfy_root = Path(args.comfyui).expanduser().resolve()
        if not comfy_root.is_dir():
            raise SidecarError(f"ComfyUI root is not a directory: {comfy_root}")
        encoder = _resolve_file(args.text_encoder, "text encoder")
        tokenizer = _resolve_file(args.tokenizer, "tokenizer") if args.tokenizer else None
        first = _resolve_file(args.first_frame, "first-frame") if args.first_frame else None
        last = _resolve_file(args.last_frame, "last-frame") if args.last_frame else None
        if args.mode == "t2v" and (first or last):
            raise SidecarError("--first-frame/--last-frame require --mode fl2va-i2v")
        if args.mode == "fl2va-i2v" and not (first or last):
            raise SidecarError("FL2VA I2V requires --first-frame or --last-frame")
        if args.mode == "fl2va-i2v" and (args.width < 64 or args.height < 64):
            raise SidecarError("I2V render width and height must both be at least 64")
        if args.mode == "fl2va-i2v" and (args.width % 32 or args.height % 32):
            raise SidecarError("I2V render width and height must be divisible by 32")
        model_hash = _model_sha256(encoder)
        source_images = [(first, True), (last, False)]
        canonical_images: list[Path] = []
        for source, is_first in source_images:
            if source is None:
                continue
            suffix = "first" if is_first else "last"
            canonical_images.append(_canonicalize_keyframe(
                source, Path(str(args.output) + f".{suffix}.png"),
                is_first=is_first, width=args.width, height=args.height))
        canonical_bytes = [path.read_bytes() for path in canonical_images]
        image_hashes = [hashlib.sha256(data).digest() for data in canonical_bytes]
        clip, torch = _load_comfy_clip(comfy_root, encoder, args.device)
        images = [_load_image_tensor(data, path, torch)
                  for data, path in zip(canonical_bytes, canonical_images)]
        ids, values, tags = _encode_raw_conditioning(clip, torch, args.prompt, args.device, images)
        if [_model_sha256(path) for path in canonical_images] != image_hashes:
            raise SidecarError("canonical keyframe changed while Qwen conditioning was generated")
        if tokenizer is not None and not canonical_images:
            diagnostic_ids = _token_ids(tokenizer, args.prompt)
            if diagnostic_ids != ids:
                raise SidecarError("diagnostic tokenizer IDs differ from ComfyUI token IDs")
        if int(values.shape[0]) != len(ids):
            raise SidecarError(
                f"ComfyUI token count {int(values.shape[0])} differs from conditioning token IDs {len(ids)}"
            )
        role = (ROLE_FIRST if first else 0) | (ROLE_LAST if last else 0)
        mode = MODE_FL2VA_I2V if canonical_images else MODE_T2V
        _write_sidecar(Path(args.output), args.prompt, ids, values, model_hash, tags,
                       mode=mode, keyframe_role=role,
                       render_width=args.width, render_height=args.height,
                       image_hashes=image_hashes)
        print(
            f"conditioning-sidecar path={Path(args.output).expanduser().resolve()} "
            f"tokens={len(ids)} shape=({len(ids)},{WIDTH}) "
            f"model_sha256={model_hash.hex()} device={args.device} mode={mode} "
            f"keyframes={len(canonical_images)}"
        )
        if first:
            print(f"conditioning-canonical-first={Path(str(args.output) + '.first.png').resolve()}")
        if last:
            print(f"conditioning-canonical-last={Path(str(args.output) + '.last.png').resolve()}")
        return 0
    except SidecarError as exc:
        print(f"encode_h3_quantized_prompt: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
