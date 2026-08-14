#!/usr/bin/env python3
"""Run an isolated, real ComfyUI 22-frame H3 smoke and publish PERF-002 traces.

The driver starts ComfyUI in-process on a private loopback port, points all
mutable ComfyUI directories at a private runtime directory, and keeps the
external checkout/model tree read-only.  Hooks are installed in the loaded T8
sampler and ComfyUI attention module; traces are written only after the queued
graph, real media output, raw-audio protocol and Sage dispatch all succeed.
The command manifest intentionally binds only an entry-point file set:
``main.py``, T8 sampling/nodes, Comfy attention and four model files. Runtime
imports beyond that set are not a reproducible source/venv lock; the matched
A/B stage must add the full closure before making performance claims.
This driver intentionally does not claim the 124-frame benchmark or a quality
result.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import socket
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable


CONTRACT = {"width": 864, "height": 480, "frames": 22, "fps": 24,
            "steps": 2, "layers": 50, "seed": 42}
T8_NODE_DIRECTORY = "comfyui-minimax-h3-audio-T8"


class TraceError(RuntimeError):
    """A fail-closed producer error."""


def _windows_reparse(path: Path) -> bool:
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError as error:
        raise TraceError(f"could not inspect {path}") from error
    return bool(attributes & 0x400)


def _reject_link_chain(path: Path, label: str) -> None:
    """Reject symlink/reparse points in a path and every existing ancestor."""
    absolute = Path(os.path.abspath(path))
    for candidate in (absolute, *absolute.parents):
        if not candidate.exists() and not candidate.is_symlink():
            continue
        if candidate.is_symlink() or _windows_reparse(candidate):
            raise TraceError(f"{label} path must not contain links or reparse points")


def _regular(path: Path, label: str) -> Path:
    _reject_link_chain(path, label)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise TraceError(f"{label} must be an absolute regular file")
    return path


def _directory(path: Path, label: str) -> Path:
    _reject_link_chain(path, label)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise TraceError(f"{label} must be an absolute directory")
    return path


def _private_directory(path: Path, label: str) -> Path:
    """Accept only a private empty directory, never a linked/reused tree."""
    _reject_link_chain(path, label)
    if not path.is_absolute() or path.is_symlink():
        raise TraceError(f"{label} must be an absolute non-linked directory")
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise TraceError(f"{label} must be a new or empty directory")
    else:
        if not path.parent.is_dir() or path.parent.is_symlink():
            raise TraceError(f"{label} parent must be an existing directory")
        path.mkdir()
    _reject_link_chain(path, label)
    return path


def _new_destination(path: Path, label: str) -> Path:
    _reject_link_chain(path, label)
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise TraceError(f"{label} must be a new absolute path")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise TraceError(f"{label} parent must be an existing directory")
    return path


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _publish_json(path: Path, value: dict[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{path.name}.",
                                         suffix=".tmp", dir=path.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(_canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise TraceError(f"refusing to overwrite existing {path.name}") from error
    except OSError as error:
        raise TraceError(f"could not publish {path.name}") from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _copy_no_clobber(source: Path, destination: Path) -> None:
    temporary: Path | None = None
    try:
        with source.open("rb") as input_stream, tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{destination.name}.", suffix=".tmp",
                dir=destination.parent, delete=False) as output_stream:
            temporary = Path(output_stream.name)
            shutil.copyfileobj(input_stream, output_stream)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.link(temporary, destination)
    except FileExistsError as error:
        raise TraceError("refusing to overwrite existing output media") from error
    except OSError as error:
        raise TraceError("could not publish output media") from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request_json(url: str, payload: dict[str, Any] | None = None,
                  timeout: float = 30.0) -> tuple[int | None, Any]:
    data = None if payload is None else _canonical(payload)
    request = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            body = response.read()
    except urllib.error.HTTPError as error:
        return int(error.code), None
    except (urllib.error.URLError, TimeoutError, OSError):
        return None, None
    try:
        return status, json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return status, None


def _validate_t8_source(comfy_root: Path) -> Path:
    node_root = comfy_root / "custom_nodes" / T8_NODE_DIRECTORY
    if (not node_root.is_dir() or node_root.is_symlink() or
            any(path.is_symlink() for path in node_root.rglob("*.py"))):
        raise TraceError("bound T8 custom-node directory is missing or linked")
    for name in ("__init__.py", "nodes.py", "sampling.py", "conditioning.py", "core.py"):
        _regular(node_root / name, f"T8 {name}")
    return node_root


def _bound_file(path: str, label: str, expected: Path | None = None) -> Path:
    value = _regular(Path(path), label)
    if expected is not None and value.resolve() != expected.resolve():
        raise TraceError(f"{label} is not the bound project file")
    return value


def _derive_bound_roots(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Path]]:
    comfy_main = _bound_file(args.comfy_main, "ComfyUI main.py")
    comfy_root = comfy_main.parent
    if comfy_main.name != "main.py":
        raise TraceError("--comfy-main must point to main.py")
    t8_dir = comfy_root / "custom_nodes" / T8_NODE_DIRECTORY
    _bound_file(args.t8_sampling, "T8 sampling.py", t8_dir / "sampling.py")
    _bound_file(args.t8_nodes, "T8 nodes.py", t8_dir / "nodes.py")
    _bound_file(args.comfy_attention, "Comfy attention.py",
                comfy_root / "comfy" / "ldm" / "modules" / "attention.py")
    model_paths = {
        "model": _bound_file(args.model_file, "FL2VA model"),
        "clip": _bound_file(args.clip_file, "Qwen model"),
        "video_vae": _bound_file(args.video_vae_file, "video VAE"),
        "audio_vae": _bound_file(args.audio_vae_file, "audio VAE"),
    }
    model_root = comfy_root / "models"
    for label, path in model_paths.items():
        try:
            path.resolve().relative_to(model_root.resolve())
        except ValueError as error:
            raise TraceError(f"{label} is outside the bound ComfyUI models root") from error
    return comfy_root, model_root, model_paths


def _verify_model_resolution(paths: dict[str, Path]) -> dict[str, str]:
    try:
        import folder_paths
    except Exception as error:
        raise TraceError("ComfyUI folder_paths import failed") from error
    categories = {"model": "diffusion_models", "clip": "text_encoders",
                  "video_vae": "vae", "audio_vae": "vae"}
    names: dict[str, str] = {}
    for label, path in paths.items():
        resolved = folder_paths.get_full_path(categories[label], path.name)
        if not isinstance(resolved, str) or not resolved:
            raise TraceError(f"ComfyUI could not resolve bound {label} model")
        candidate = Path(resolved)
        if (candidate.is_symlink() or not candidate.is_file() or
                candidate.resolve() != path.resolve()):
            raise TraceError(f"ComfyUI resolved a different {label} model file")
        names[label] = path.name
    return names


def _validate_sigma_grid(values: list[float], label: str, steps: int) -> None:
    if (len(values) != steps + 1 or
            not all(math.isfinite(value) for value in values) or
            values[0] != 1.0 or values[-1] != 0.0 or
            any(values[index] < values[index + 1]
                for index in range(len(values) - 1))):
        raise TraceError(f"Comfy {label} sigma grid is invalid")


def _build_workflow(image_name: str, prompt: str, model: str, clip: str,
                    video_vae: str, audio_vae: str) -> dict[str, dict[str, Any]]:
    """Build the fixed first-frame FL2VA I2V 864x480/22f/2-step graph."""
    return {
        "1": {"class_type": "UNETLoader", "inputs":
              {"unet_name": model, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs":
              {"clip_name": clip, "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": video_vae}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": audio_vae}},
        "5": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "6": {"class_type": "MiniMaxH3AudioConditioningT8", "inputs": {
            "clip": ["2", 0], "video_vae": ["3", 0], "audio_vae": ["4", 0],
            "prompt": prompt, "width": 864, "height": 480, "length": 22,
            "task_type": "I2VA", "audio_mode": "native",
            "audio_denoise_strength": 1.0, "add_source_as_reference": False,
            "prompt_primary_audio_ordinal": 0, "strict_prompt_tags": True,
            "ref_image_size": "match", "reference_video_policy": "official_2_to_15s",
            "first_frame": ["5", 0]}},
        "7": {"class_type": "MiniMaxH3DualClockSamplerT8", "inputs": {
            "model": ["1", 0], "av_latent": ["6", 1], "steps": 2,
            "shift_video": 12.0, "shift_audio": 3.0,
            "sampler_name": "dual_clock_euler", "scheduler": "native_flow"}},
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
        "9": {"class_type": "BasicGuider", "inputs":
              {"model": ["7", 0], "conditioning": ["6", 0]}},
        "10": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["8", 0], "guider": ["9", 0], "sampler": ["7", 1],
            "sigmas": ["7", 2], "latent_image": ["6", 1]}},
        "11": {"class_type": "MiniMaxH3AVDecodeT8", "inputs": {
            "av_latent": ["10", 0], "video_vae": ["3", 0], "audio_vae": ["4", 0]}},
        "12": {"class_type": "CreateVideo", "inputs": {
            "images": ["11", 0], "audio": ["11", 1], "fps": 24.0}},
        "13": {"class_type": "SaveVideo", "inputs": {
            "video": ["12", 0], "filename_prefix": "perf002_comfy_smoke",
            "format": "mp4", "codec": "auto"}},
    }


def _install_runtime_hooks(state: dict[str, Any]) -> None:
    """Patch already-imported Comfy/T8 modules, never files in the checkout."""
    # Comfy startup has already imported torch.  Reusing that module keeps the
    # contract tests portable on a clean Python environment and avoids loading
    # a second, potentially different torch installation in the producer.
    torch_module = sys.modules.get("torch")
    if torch_module is None or not hasattr(torch_module, "bfloat16"):
        raise TraceError("ComfyUI torch runtime is unavailable")
    state["bf16_dtype"] = torch_module.bfloat16
    sampling_modules: list[Any] = []
    for module in list(sys.modules.values()):
        if module is None:
            continue
        if callable(getattr(module, "sample_minimax_h3_dual_clock_euler", None)):
            sampling_modules.append(module)
    if not sampling_modules:
        raise TraceError("loaded ComfyUI graph did not expose the T8 sampler")
    source = sampling_modules[0]
    original_sample = source.sample_minimax_h3_dual_clock_euler
    original_setup = getattr(source, "setup_dual_clock_sampling", None)
    if original_setup is None:
        raise TraceError("loaded T8 graph did not expose sampler setup")

    def setup_hook(*args: Any, **kwargs: Any) -> Any:
        result = original_setup(*args, **kwargs)
        try:
            steps = int(kwargs.get("steps", args[2] if len(args) > 2 else -1))
            shift_video = float(kwargs.get("shift_video", args[3] if len(args) > 3 else math.nan))
            shift_audio = float(kwargs.get("shift_audio", args[4] if len(args) > 4 else math.nan))
            sampler_name = str(kwargs.get("sampler_name", args[5] if len(args) > 5 else ""))
            scheduler_name = str(kwargs.get("scheduler", args[6] if len(args) > 6 else ""))
            if (steps != CONTRACT["steps"] or shift_video != 12.0 or
                    shift_audio != 3.0 or sampler_name != "dual_clock_euler" or
                    scheduler_name != "native_flow"):
                raise TraceError("Comfy runtime changed the PERF-002 sampler contract")
            sigmas = result[2].detach().cpu().tolist()
            state["sigma_video"] = [float(value) for value in sigmas]
            audio = source.time_shift_sigma(result[2], shift_video, shift_audio)
            state["sigma_audio"] = [float(value) for value in audio.detach().cpu().tolist()]
            _validate_sigma_grid(state["sigma_video"], "video", steps)
            _validate_sigma_grid(state["sigma_audio"], "audio", steps)
            state["setup_success"] = True
        except Exception as error:
            raise TraceError("could not capture Comfy scheduler output") from error
        return result

    def sample_hook(*args: Any, **kwargs: Any) -> Any:
        state["sampling_active"] = True
        try:
            result = original_sample(*args, **kwargs)
            state["sample_success"] = True
            sigmas = args[2] if len(args) > 2 else kwargs.get("sigmas")
            state["audio_steps"] = int(len(sigmas) - 1)
            state["raw_audio_protocol"] = bool(
                kwargs.get("audio_velocity_is_raw", args[-1] if args else False))
            return result
        finally:
            state["sampling_active"] = False

    for module in sampling_modules:
        module.sample_minimax_h3_dual_clock_euler = sample_hook
        if getattr(module, "setup_dual_clock_sampling", None) is original_setup:
            module.setup_dual_clock_sampling = setup_hook
    # nodes.py imported setup_dual_clock_sampling into its own module namespace.
    for module in list(sys.modules.values()):
        if getattr(module, "setup_dual_clock_sampling", None) is original_setup:
            module.setup_dual_clock_sampling = setup_hook

    attention_modules = [module for module in list(sys.modules.values())
                         if module is not None
                         and callable(getattr(module, "attention_sage", None))
                         and callable(getattr(module, "sageattn", None))
                         and callable(getattr(module, "attention_pytorch", None))]
    if not attention_modules:
        raise TraceError("ComfyUI attention module was not loaded")
    # Capture aliases before replacing the provider module.  H3's minimax
    # model does ``from ...attention import optimized_attention`` at import
    # time, so rebinding only ``comfy.ldm.modules.attention`` would leave the
    # model calling the old function and bypass the accounting hook.
    original_aliases: list[Any] = []
    for module in attention_modules:
        original_attention_sage = module.attention_sage
        original_aliases.append(original_attention_sage)
        for alias_name in ("optimized_attention", "optimized_attention_masked"):
            alias = getattr(module, alias_name, None)
            if alias is not None and any(alias is item for item in original_aliases):
                original_aliases.append(alias)

    alias_consumers: list[tuple[Any, str]] = []
    for consumer in list(sys.modules.values()):
        if consumer is None:
            continue
        for alias_name in ("attention_sage", "optimized_attention",
                           "optimized_attention_masked"):
            value = getattr(consumer, alias_name, None)
            if any(value is original for original in original_aliases):
                alias_consumers.append((consumer, alias_name))

    rebound_consumers: set[str] = set()
    for module in attention_modules:
        original_sage = module.sageattn
        original_pytorch: Callable[..., Any] = module.attention_pytorch
        original_attention_sage = module.attention_sage

        def sage_hook(*args: Any, _orig=original_sage, **kwargs: Any) -> Any:
            try:
                result = _orig(*args, **kwargs)
            except Exception:
                # attention_sage catches this exception and calls the wrapped
                # attention_pytorch fallback below; do not count it as a hit.
                raise
            if state.get("sampling_active") and state.get("in_sage"):
                state["sage_hits"] += 1
            return result

        def pytorch_hook(*args: Any, _orig=original_pytorch, **kwargs: Any) -> Any:
            if state.get("sampling_active") and state.get("in_sage"):
                state["fallbacks"] += 1
            return _orig(*args, **kwargs)

        def attention_sage_hook(*args: Any, _orig=original_attention_sage,
                                **kwargs: Any) -> Any:
            if state.get("sampling_active"):
                state["sage_attempts"] += 1
                tensors = args[:3]
                if len(tensors) != 3 or any(
                        getattr(tensor, "dtype", None) != state["bf16_dtype"]
                        for tensor in tensors):
                    state["all_bf16"] = False
                state["in_sage"] += 1
                try:
                    return _orig(*args, **kwargs)
                finally:
                    state["in_sage"] -= 1
            return _orig(*args, **kwargs)

        module.sageattn = sage_hook
        module.attention_pytorch = pytorch_hook
        # Comfy selects optimized_attention once at import time.  Rebind both
        # aliases so a captured original cannot evade the entry accounting.
        module.attention_sage = attention_sage_hook
        module.optimized_attention = attention_sage_hook
        module.optimized_attention_masked = attention_sage_hook
        if not getattr(module, "SAGE_ATTENTION_IS_AVAILABLE", False):
            raise TraceError("ComfyUI SageAttention package is unavailable")
        # Rebind every captured consumer alias, including minimax/model.py.
        # The provider itself is already patched above; consumer aliases are
        # the important part because they hold a copied function object.
        for consumer, alias_name in alias_consumers:
            value = getattr(consumer, alias_name, None)
            if any(value is original for original in original_aliases):
                setattr(consumer, alias_name, attention_sage_hook)
                rebound_consumers.add(getattr(consumer, "__name__", "<anonymous>"))

    if not rebound_consumers:
        raise TraceError("ComfyUI Sage optimized-attention aliases were not rebound")
    state["attention_alias_rebound_modules"] = sorted(rebound_consumers)


def _comfy_argv(comfy_root: Path, runtime: Path, model_root: Path,
                port: int) -> list[str]:
    return [str(comfy_root / "main.py"), "--listen", "127.0.0.1",
            "--port", str(port), "--use-sage-attention", "--disable-metadata",
            "--disable-all-custom-nodes", "--whitelist-custom-nodes",
            T8_NODE_DIRECTORY, "--disable-auto-launch", "--dont-print-server",
            "--output-directory", str(runtime / "output"),
            "--input-directory", str(runtime / "input"),
            "--temp-directory", str(runtime / "temp"),
            "--user-directory", str(runtime / "user"),
            "--models-directory", str(model_root),
            "--database-url", f"sqlite:///{runtime / 'comfy.db'}"]


def _start_server(comfy_root: Path, runtime: Path, model_root: Path,
                  state: dict[str, Any], port: int) -> tuple[threading.Thread, queue.Queue]:
    result: queue.Queue = queue.Queue(maxsize=1)
    def target() -> None:
        try:
            sys.dont_write_bytecode = True
            if str(comfy_root) not in sys.path:
                sys.path.insert(0, str(comfy_root))
            sys.argv = _comfy_argv(comfy_root, runtime, model_root, port)
            comfy_main = importlib.import_module("main")
            loop, _server, start_all = comfy_main.start_comfyui()
            _install_runtime_hooks(state)
            state["loop"] = loop
            state["startup_sent"] = True
            result.put((loop, _server))
            try:
                loop.run_until_complete(start_all())
            except RuntimeError:
                if not state.get("stop_requested"):
                    raise
            except BaseException as error:
                if not state.get("stop_requested"):
                    state["server_error"] = error
                raise
            finally:
                loop.close()
        except BaseException as error:  # delivered to parent thread
            if not state.get("startup_sent"):
                try:
                    result.put_nowait(error)
                except queue.Full:
                    pass
            elif not state.get("stop_requested"):
                state["server_error"] = error
    thread = threading.Thread(target=target, name="perf002-comfy", daemon=True)
    thread.start()
    return thread, result


def _wait_for_server(base: str, timeout: float, state: dict[str, Any]) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if state.get("server_error") is not None:
            raise TraceError("private ComfyUI server stopped during startup") from state["server_error"]
        status, _ = _request_json(f"{base}/system_stats", timeout=2)
        if status == 200:
            return
        time.sleep(0.5)
    raise TraceError("ComfyUI loopback server did not become ready")


def _queue_and_wait(base: str, workflow: dict[str, dict[str, Any]], timeout: float,
                    client_id: str, state: dict[str, Any]) -> dict[str, Any]:
    status, payload = _request_json(
        f"{base}/prompt", {"prompt": workflow, "client_id": client_id}, timeout=60)
    if status != 200 or not isinstance(payload, dict):
        raise TraceError("ComfyUI rejected the PERF-002 graph")
    prompt_id = payload.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise TraceError("ComfyUI did not return a prompt id")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if state.get("server_error") is not None:
            raise TraceError("private ComfyUI server stopped during graph execution") from state["server_error"]
        history_status, history_payload = _request_json(
            f"{base}/history/{prompt_id}", timeout=30)
        if history_status == 200 and isinstance(history_payload, dict):
            history = history_payload.get(prompt_id)
            if isinstance(history, dict):
                status = history.get("status", {})
                messages = status.get("messages", []) if isinstance(status, dict) else []
                if status.get("status_str") == "error" or any(
                        isinstance(message, list) and message and message[0] == "execution_error"
                        for message in messages):
                    raise TraceError("ComfyUI graph execution failed")
                if status.get("completed"):
                    return history
        time.sleep(1.0)
    raise TraceError("ComfyUI graph timed out")


def _find_media(history: dict[str, Any], output_root: Path) -> Path:
    for node_output in history.get("outputs", {}).values():
        if not isinstance(node_output, dict):
            continue
        for key in ("videos", "images"):
            for media in node_output.get(key, []):
                if not isinstance(media, dict) or not media.get("filename"):
                    continue
                relative = Path(str(media["filename"]))
                if media.get("subfolder"):
                    relative = Path(str(media["subfolder"])) / relative
                raw_candidate = output_root / relative
                _reject_link_chain(raw_candidate, "ComfyUI media output")
                candidate = raw_candidate.resolve()
                try:
                    candidate.relative_to(output_root.resolve())
                except ValueError:
                    continue
                if candidate.is_file() and not candidate.is_symlink():
                    return candidate
    raise TraceError("ComfyUI completed without a media output")


def run(args: argparse.Namespace) -> None:
    comfy_root, model_root, model_paths = _derive_bound_roots(args)
    _directory(comfy_root, "ComfyUI root")
    _directory(model_root, "model root")
    _validate_t8_source(comfy_root)
    reference = _regular(Path(args.reference_png), "reference PNG")
    prompt_file = _regular(Path(args.prompt_file), "prompt file")
    media = _new_destination(Path(args.output_media), "output media")
    scheduler = _new_destination(Path(args.scheduler_trace), "scheduler trace")
    attention = _new_destination(Path(args.attention_trace), "attention trace")
    if any(args_value != CONTRACT[name] for name, args_value in
           (("width", args.width), ("height", args.height),
            ("frames", args.frames), ("steps", args.steps))):
        raise TraceError("Comfy producer requires the fixed 864x480/22f/2-step contract")
    runtime = _private_directory(Path(args.runtime_dir), "private runtime")
    for name in ("output", "input", "temp", "user"):
        child = runtime / name
        _reject_link_chain(child, f"private runtime {name}")
        if child.exists() and (child.is_symlink() or not child.is_dir() or any(child.iterdir())):
            raise TraceError(f"private runtime {name} directory must be new and unlinked")
        child.mkdir(exist_ok=True)
        _reject_link_chain(child, f"private runtime {name}")
    image_destination = runtime / "input" / reference.name
    if image_destination.exists() or image_destination.is_symlink():
        raise TraceError("private input image target already exists")
    shutil.copy2(reference, image_destination)
    if prompt_file.stat().st_size > 1024 * 1024:
        raise TraceError("prompt file exceeds the 1 MiB smoke limit")
    prompt = prompt_file.read_text(encoding="utf-8")
    state: dict[str, Any] = {"setup_success": False, "sample_success": False,
                             "sampling_active": False, "raw_audio_protocol": False,
                             "audio_steps": 0, "sigma_video": [], "sigma_audio": [],
                             "sage_attempts": 0, "sage_hits": 0, "fallbacks": 0,
                             "all_bf16": True, "in_sage": 0}
    port = _free_port()
    thread, result = _start_server(comfy_root, runtime, model_root, state, port)
    source_media: Path | None = None
    scheduler_report: dict[str, Any] | None = None
    attention_report: dict[str, Any] | None = None
    try:
        startup = result.get(timeout=600)
        if isinstance(startup, BaseException):
            raise TraceError("ComfyUI failed during startup") from startup
        model_names = _verify_model_resolution(model_paths)
        base = f"http://127.0.0.1:{port}"
        _wait_for_server(base, 120, state)
        history = _queue_and_wait(
            base, _build_workflow(image_destination.name, prompt, model_names["model"],
                                  model_names["clip"], model_names["video_vae"],
                                  model_names["audio_vae"]), args.timeout,
            str(uuid.uuid4()), state)
        source_media = _find_media(history, runtime / "output")
        if not (state["setup_success"] and state["sample_success"] and
                state["raw_audio_protocol"] and state["audio_steps"] == args.steps and
                len(state["sigma_video"]) == args.steps + 1 and
                len(state["sigma_audio"]) == args.steps + 1 and state["sage_hits"] > 0 and
                state["sage_attempts"] == state["sage_hits"] + state["fallbacks"] and
                state["sage_attempts"] > 0 and state["all_bf16"] and
                state["fallbacks"] == 0):
            raise TraceError("Comfy runtime completed without complete scheduler/Sage proof")
        scheduler_report = {"schema_version": 1, "engine": "comfyui",
            "sampler": "dual_clock_euler", "schedule": "native_flow",
            "video_shift": 12.0, "audio_shift": 3.0, **CONTRACT,
            "sigma_video": state["sigma_video"], "sigma_audio": state["sigma_audio"],
            "raw_audio_protocol_verified": True}
        attention_report = {"schema_version": 1, "engine": "comfyui",
            "requested": "sage", "selected": "sage", "scope": "dit_bf16",
            "backend_hits": state["sage_hits"], "expected_native_calls": 0,
            "unexpected_fallbacks": state["fallbacks"]}
    finally:
        # Stop only the private in-process server; never signal a user-owned
        # ComfyUI instance.  Joining also prevents a successful smoke from
        # leaving a listening loopback thread behind.
        state["stop_requested"] = True
        loop = state.get("loop")
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=15)
        if thread.is_alive():
            raise TraceError("private ComfyUI server did not stop cleanly")
    if source_media is None or scheduler_report is None or attention_report is None:
        raise TraceError("Comfy runtime completed without publishable evidence")
    media_published = False
    scheduler_published = False
    attention_published = False
    try:
        _copy_no_clobber(source_media, media)
        media_published = True
        _publish_json(scheduler, scheduler_report)
        scheduler_published = True
        _publish_json(attention, attention_report)
        attention_published = True
    except Exception:
        for path, published in ((attention, attention_published),
                                (scheduler, scheduler_published),
                                (media, media_published)):
            if published:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
        raise
    print("PERF-002C ComfyUI real 22-frame smoke complete")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-main", required=True)
    parser.add_argument("--t8-sampling", required=True)
    parser.add_argument("--t8-nodes", required=True)
    parser.add_argument("--comfy-attention", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--clip-file", required=True)
    parser.add_argument("--video-vae-file", required=True)
    parser.add_argument("--audio-vae-file", required=True)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--reference-png", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-media", required=True)
    parser.add_argument("--scheduler-trace", required=True)
    parser.add_argument("--attention-trace", required=True)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=22)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    try:
        run(args)
    except (TraceError, OSError) as error:
        print(f"PERF-002C ComfyUI real smoke failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
