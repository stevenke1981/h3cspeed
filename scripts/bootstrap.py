#!/usr/bin/env python3
"""Fetch and patch the pinned antirez/h3.c source tree for h3cspeed.

The NVIDIA port is maintained as a small overlay so upstream H3 model loading,
CLI, safetensor parsing and multimodal orchestration remain auditable.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import uuid
import zipfile

UPSTREAM_REPO = "antirez/h3.c"
UPSTREAM_COMMIT = "8974cc055ea9c02fcd14cc27dfda3e1027c05153"
ARCHIVE_URL = f"https://github.com/{UPSTREAM_REPO}/archive/{UPSTREAM_COMMIT}.zip"
ARCHIVE_SHA256 = "dc6d3cd25cb70d5c723292e60f3f3b9093688a731467008a691d9a7412d3e8f3"
PREPARED_TREE_SHA256 = "ad08f146af8cada5ad9977f9c1ca4b7e4c512bab39a619a864a723c6ce74323b"

# Git blob SHA-1 values, not ordinary file hashes. They pin the exact interfaces
# on which the CUDA overlay was developed.
EXPECTED_GIT_BLOBS = {
    "h3.c": "d5dca2594a92413f5ff845c2f1e43d08a7697d5c",
    "h3.h": "29640b37abaa056341cf5e827ecc9df501ca7c5c",
    "h3_gpu.h": "3a47cc3507106a4deee5b4805b936a8c8ef34d02",
    "h3_host.c": "a04a2a0a0edafb111dadaa470206f0cb89344fb3",
    "h3_tokenizer.h": "e95e49b1d5149c445723d9552f6964917245a7de",
    "h3_tokenizer.m": "0d519fbd95c2728b2ab13a609695cfe311c626a7",
    "Makefile": "bb202379b4f30f997d9b168b6c0488cfe61d6946",
    "h3_safetensors.c": "60ab177a02b318e5f2b07a217560cb64ac7f1445",
    "h3_weights.c": "9a0224dab1d009241f3a306b68740945c372b5ac",
    "h3_text_encoder.c": "9c0e8fdb385f8017f0f476093c79c0f58569ba91",
    "h3_dit_schedule.c": "cbc343d7cc9eb6e565d5f710b5d2ab5f2f5cabe6",
    "h3_dit.c": "667e49b28eab928dcc34fbe0cee8ac67c2ddba1c",
    "h3_video_vae.c": "149e11364384ebcbeb463f1140ff86400cd519ee",
    "h3_video_encoder.c": "f60a96d58e04f277fb1479122f1e528727e8b1b1",
    "h3_audio_vae.c": "f563ae1d0188b2abe20721d54e8bc35c8978cb43",
    "h3_ffmpeg.c": "66762425d2b2d8d166e867b7ddb96364d87976a9",
    "h3_terminal.c": "5b216e2e24ebf6468b80f4fd00332430240f16a3",
    "h3_vision_encoder.c": "8867231b3b88064ee01ee865ebaa30af53a5b5e3",
    "h3_multimodal.c": "f7b25455865b6fe937b0e2f08c72770e3d39e96d",
    "main.c": "7f11e470a3c1a86a80f6f61fd5dffe67e8bbf621",
    "h3_cli.c": "79339c8237bd1bf61f6eabb58612b94e7655f25e",
    "linenoise.c": "2345b851b738d1106f5015b623a64a926a751216",
    "h3_metal.h": "beaf628df369e3a29c403b8d27474eb7fdb78ee3",
}

# Filled with deterministic post-patch blob hashes.  Keeping this independent
# of the generated marker detects edited or stale prepared trees without a
# network request on every configure.
PREPARED_GIT_BLOBS: dict[str, str] = {
    "h3.c": "587dece23c5a4325ab40f9ba202fc46ff3782aa6",
    "h3.h": "29640b37abaa056341cf5e827ecc9df501ca7c5c",
    "h3_gpu.h": "7fb2871f0c7fca24c029fff80e1a135e06092a72",
    "h3_host.c": "6e875effe9dbe4294a3e35207806282decbad304",
    "h3_tokenizer.h": "e95e49b1d5149c445723d9552f6964917245a7de",
    "h3_tokenizer.m": "0d519fbd95c2728b2ab13a609695cfe311c626a7",
    "Makefile": "bb202379b4f30f997d9b168b6c0488cfe61d6946",
    "h3_safetensors.c": "2f1abd4202cbe854f0840c4db7acd2f2250ae46b",
    "h3_weights.c": "a7321078b58e86bd87ad1f55192d818d81499bad",
    "h3_text_encoder.c": "1089f0395f69b11f2327785de3c7aef291e85df7",
    "h3_dit_schedule.c": "bbffd91d5615deebcc093d1c97c3392a043e5e0b",
    "h3_dit.c": "980348657be5471fdeb50fee8e361618a4c96e26",
    "h3_video_vae.c": "149e11364384ebcbeb463f1140ff86400cd519ee",
    "h3_video_encoder.c": "f60a96d58e04f277fb1479122f1e528727e8b1b1",
    "h3_audio_vae.c": "2eef6e2e3e85968f056c6d1da1691cdd036c23e5",
    "h3_ffmpeg.c": "b051cf185be1ae8e415a4024b526c1bf685cfbef",
    "h3_terminal.c": "5b216e2e24ebf6468b80f4fd00332430240f16a3",
    "h3_vision_encoder.c": "8867231b3b88064ee01ee865ebaa30af53a5b5e3",
    "h3_multimodal.c": "f7b25455865b6fe937b0e2f08c72770e3d39e96d",
    "main.c": "7f11e470a3c1a86a80f6f61fd5dffe67e8bbf621",
    "h3_cli.c": "dddb2e655e2137e315762a96ba9db014a93c85a1",
    "linenoise.c": "2345b851b738d1106f5015b623a64a926a751216",
    "h3_metal.h": "c4fa79f06680c0d23012f4e500c46f1314561cb0",
    "h3_safetensors.h": "90335175f84b7d48349783c06fd044f0b4306981",
    "h3_weights.h": "f73d3766797e1ee7c871a24225d2af273092ed14",
}


def git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def tree_sha256(root: Path) -> str:
    """Hash every prepared upstream file, including paths and lengths."""
    digest = hashlib.sha256()
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "h3cspeed-bootstrap/0.2"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        if getattr(response, "status", 200) != 200:
            raise RuntimeError(f"download failed with HTTP {response.status}")
        destination.write_bytes(response.read())


def verify_archive(path: Path) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != ARCHIVE_SHA256:
        raise RuntimeError(
            f"upstream archive SHA-256 mismatch: expected {ARCHIVE_SHA256}, got {actual}"
        )


def verify_tree(root: Path) -> None:
    errors: list[str] = []
    for relative, expected in EXPECTED_GIT_BLOBS.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"missing {relative}")
            continue
        actual = git_blob_sha(path.read_bytes())
        if actual != expected:
            errors.append(f"{relative}: expected {expected}, got {actual}")
    if errors:
        joined = "\n  - ".join(errors)
        raise RuntimeError(
            "upstream source does not match the pinned H3 revision:\n  - " + joined
        )


def replace_host_resize(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace("#include <Accelerate/Accelerate.h>\n\n", "")
    include_marker = '#include "h3_host.h"\n'
    if '#include "h3_resize_portable.h"' not in text:
        text = text.replace(
            include_marker,
            include_marker + '#include "h3_resize_portable.h"\n',
            1,
        )

    pattern = re.compile(
        r"int h3_resize_rgb24_high_quality\(.*?\n\}\n\nstatic double h3_phi1",
        re.DOTALL,
    )
    replacement = """int h3_resize_rgb24_high_quality(const uint8_t *input, int frames,
                                 int input_width, int input_height,
                                 int output_width, int output_height,
                                 uint8_t **output) {
    return h3cspeed_resize_rgb24_lanczos(input, frames,
                                         input_width, input_height,
                                         output_width, output_height,
                                         output);
}

static double h3_phi1"""
    patched, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError("unable to locate the vImage resize implementation")
    path.write_text(patched, encoding="utf-8", newline="\n")


def patch_cli_name(root: Path) -> None:
    # Keep command-line flags and behavior identical while exposing the new name
    # in help text and diagnostics. This intentionally does not rename API symbols.
    for relative in ("main.c", "h3_cli.c"):
        path = root / relative
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        text = text.replace("Usage: h3 ", "Usage: h3cspeed ")
        text = text.replace("usage: h3 ", "usage: h3cspeed ")
        text = text.replace("./h3 ", "./h3cspeed ")
        path.write_text(text, encoding="utf-8", newline="\n")


def patch_windows_stat(root: Path) -> None:
    path = root / "h3.c"
    text = path.read_text(encoding="utf-8")
    old = '    return h3_key_append(key, "|%s=%zu:%s:%lld:%lld:%ld", role, strlen(path),\n                         path, (long long)status.st_size,\n                         (long long)status.st_mtimespec.tv_sec,\n                         status.st_mtimespec.tv_nsec);'
    linux = '    return h3_key_append(key, "|%s=%zu:%s:%lld:%lld:%ld", role, strlen(path),\n                         path, (long long)status.st_size,\n                         (long long)status.st_mtim.tv_sec,\n                         status.st_mtim.tv_nsec);'
    new = '#if defined(_WIN32)\n    return h3_key_append(key, "|%s=%zu:%s:%lld:%lld:%d", role, strlen(path),\n                         path, (long long)status.st_size,\n                         (long long)status.st_mtime, 0);\n#elif defined(__APPLE__)\n' + old + '\n#else\n' + linux + '\n#endif'
    if old not in text:
        raise RuntimeError("unable to locate h3.c stat key")
    path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")

    safetensors = root / "h3_safetensors.c"
    source = safetensors.read_text(encoding="utf-8")
    old_offset = "(off_t)(tensor->file_offset + done)"
    new_offset = """#if defined(_WIN32)
                              (int64_t)(tensor->file_offset + done)
#else
                              (off_t)(tensor->file_offset + done)
#endif
                             """
    if old_offset not in source:
        raise RuntimeError("unable to locate safetensors pread offset")
    safetensors.write_text(source.replace(old_offset, new_offset, 1),
                           encoding="utf-8", newline="\n")


def patch_ffmpeg_limits(root: Path) -> None:
    """Make POSIX SSIZE_MAX use self-contained on Linux and BSD hosts."""
    path = root / "h3_ffmpeg.c"
    text = path.read_text(encoding="utf-8")
    marker = "#include <errno.h>\n"
    if "#include <limits.h>" not in text:
        if marker not in text:
            raise RuntimeError("unable to locate h3_ffmpeg errno include")
        text = text.replace(marker, marker + "#include <limits.h>\n", 1)
    path.write_text(text, encoding="utf-8", newline="\n")


def patch_linux_random_seed(root: Path) -> None:
    """Use Linux getrandom instead of the newer glibc arc4random_buf API."""
    path = root / "h3_cli.c"
    text = path.read_text(encoding="utf-8")
    include_marker = "#include <sys/stat.h>\n"
    include_patch = (
        "#include <sys/stat.h>\n"
        "#if defined(__linux__)\n"
        "#include <sys/random.h>\n"
        "#endif\n"
    )
    old = (
        "static uint64_t random_seed(void) {\n"
        "    uint64_t value;\n"
        "    arc4random_buf(&value, sizeof(value));\n"
        "    return value;\n"
        "}\n"
    )
    new = (
        "static uint64_t random_seed(void) {\n"
        "    uint64_t value = 0;\n"
        "#if defined(__linux__)\n"
        "    unsigned char *cursor = (unsigned char *)&value;\n"
        "    size_t remaining = sizeof(value);\n"
        "    while (remaining) {\n"
        "        ssize_t count = getrandom(cursor, remaining, 0);\n"
        "        if (count > 0) {\n"
        "            cursor += (size_t)count;\n"
        "            remaining -= (size_t)count;\n"
        "            continue;\n"
        "        }\n"
        "        if (count < 0 && errno == EINTR) continue;\n"
        "        break;\n"
        "    }\n"
        "    if (!remaining) return value;\n"
        "    value = ((uint64_t)time(NULL) << 32) ^ (uint64_t)getpid();\n"
        "#else\n"
        "    arc4random_buf(&value, sizeof(value));\n"
        "#endif\n"
        "    return value;\n"
        "}\n"
    )
    if "#include <sys/random.h>" not in text:
        if include_marker not in text:
            raise RuntimeError("unable to locate h3_cli sys/stat include")
        text = text.replace(include_marker, include_patch, 1)
    if old in text:
        text = text.replace(old, new, 1)
    elif new not in text:
        raise RuntimeError("unable to locate h3_cli random_seed")
    path.write_text(text, encoding="utf-8", newline="\n")


def patch_text_embedding_sidecar(root: Path) -> None:
    """Add the explicit text sidecar bridge to h3.c."""
    path = root / "h3.c"
    text = path.read_text(encoding="utf-8")
    include_marker = '#include "h3_text_embedding_file.h"\n'
    if include_marker not in text:
        include = '#include "h3_text_encoder.h"\n'
        if include not in text:
            raise RuntimeError("unable to locate h3 text encoder include")
        text = text.replace(include, include + include_marker, 1)
    fl2va_include = '#include "h3_fl2va_sidecar.h"\n'
    if fl2va_include not in text:
        text = text.replace(include_marker, include_marker + fl2va_include, 1)
    if "h3_parse_sha256_hex" not in text:
        include = '#include "h3_text_embedding_file.h"\n'
        helper = '''#include <string.h>

static int h3_parse_sha256_hex(const char *text, uint8_t output[32]) {
    if (!text || strlen(text) != 64) return 0;
    for (size_t index = 0; index < 32; index++) {
        unsigned value = 0;
        for (unsigned nibble = 0; nibble < 2; nibble++) {
            unsigned char digit = (unsigned char)text[index * 2 + nibble];
            if (digit >= '0' && digit <= '9') value = (value << 4) + digit - '0';
            else if (digit >= 'a' && digit <= 'f') value = (value << 4) + digit - 'a' + 10u;
            else if (digit >= 'A' && digit <= 'F') value = (value << 4) + digit - 'A' + 10u;
            else return 0;
        }
        output[index] = (uint8_t)value;
    }
    return 1;
}
'''
        text = text.replace(include, include + helper, 1)
    if "text-sidecar" not in text:
        old = ("    if (!h3_key_append(&key, \"mode=%d|prompt=%zu:%s\", ref2va,\n"
               "                       strlen(prompt), prompt)) goto failed;\n"
               "    if (!ref2va && !params->first_frame && !params->last_frame) return key.text;")
        new = ("    if (!h3_key_append(&key, \"mode=%d|prompt=%zu:%s\", ref2va,\n"
               "                       strlen(prompt), prompt)) goto failed;\n"
               "    /* Keep the explicit sidecar path and stat in the cache key. */\n"
               "    const char *sidecar = getenv(\"H3CSPEED_TEXT_EMBEDDING\");\n"
               "    if (sidecar && !h3_key_file(&key, \"text-sidecar\", sidecar)) goto failed;\n"
               "    if (sidecar && !h3_key_append(&key, \"|text-sha=%s\",\n"
               "                                  getenv(\"H3CSPEED_TEXT_ENCODER_SHA256\") ?\n"
               "                                  getenv(\"H3CSPEED_TEXT_ENCODER_SHA256\") :\n"
               "                                  \"missing\")) goto failed;\n"
               "    if (!ref2va && !params->first_frame && !params->last_frame) return key.text;")
        if old not in text:
            raise RuntimeError("unable to locate h3 conditioning cache key")
        text = text.replace(old, new, 1)
    if 'const char *text_sidecar_path = getenv("H3CSPEED_TEXT_EMBEDDING");' not in text:
        old = ("    int ref2va = params->reference_count != 0;\n"
               "    if (ref2va && !ctx->model.ref2va_transformer.files) {")
        new = ("    int ref2va = params->reference_count != 0;\n"
               "    const char *text_sidecar_path = getenv(\"H3CSPEED_TEXT_EMBEDDING\");\n"
               "    uint8_t text_sidecar_sha256[32];\n"
               "    if (text_sidecar_path &&\n"
               "        ref2va) {\n"
               "        h3_set_error(ctx,\n"
               "            \"H3CSPEED_TEXT_EMBEDDING does not support Ref2VA references\");\n"
               "        return NULL;\n"
               "    }\n"
               "    if (text_sidecar_path &&\n"
               "        !h3_parse_sha256_hex(getenv(\"H3CSPEED_TEXT_ENCODER_SHA256\"),\n"
               "                             text_sidecar_sha256)) {\n"
               "        h3_set_error(ctx,\n"
               "            \"H3CSPEED_TEXT_ENCODER_SHA256 must be exactly 64 hexadecimal \"\n"
               "            \"characters when H3CSPEED_TEXT_EMBEDDING is set\");\n"
               "        return NULL;\n"
               "    }\n"
               "    /* Never reuse conditioning/prepared caches across a mutable sidecar. */\n"
               "    if (text_sidecar_path) h3_cache_clear(ctx);\n"
               "    if (ref2va && !ctx->model.ref2va_transformer.files) {")
        if old not in text:
            raise RuntimeError("unable to locate h3 generation mode check")
        text = text.replace(old, new, 1)
    if "skipping native Qwen text encoder" not in text:
        old = ("        if (!h3_text_encode_bf16(\n"
               "                text_path, \"h3_shaders.metal\", ids, token_count,\n"
               "                h3_text_progress_bridge, &progress, &text,\n"
               "                detail, sizeof(detail))) {\n"
               "            h3_set_error(ctx, \"%s\", detail);\n"
               "            goto cleanup;\n"
               "        }")
        new = ("        int text_ok;\n"
               "        if (text_sidecar_path) {\n"
               "            fprintf(stderr,\n"
               "                \"h3: using text sidecar %s; skipping native Qwen text encoder\\n\",\n"
               "                text_sidecar_path);\n"
               "            text_ok = h3cspeed_text_embedding_load_file(\n"
               "                text_sidecar_path, prompt, ids, token_count,\n"
               "                text_sidecar_sha256, &text,\n"
               "                detail, sizeof(detail));\n"
               "        } else {\n"
               "            text_ok = h3_text_encode_bf16(\n"
               "                text_path, \"h3_shaders.metal\", ids, token_count,\n"
               "                h3_text_progress_bridge, &progress, &text,\n"
               "                detail, sizeof(detail));\n"
               "        }\n"
               "        if (!text_ok) {\n"
               "            h3_set_error(ctx, \"%s\", detail);\n"
               "            goto cleanup;\n"
               "        }")
        if old not in text:
            raise RuntimeError("unable to locate h3 pure-T2V text encoder call")
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8", newline="\n")


def patch_text_embedding_i2v(root: Path) -> None:
    """Allow only the v2 FL2VA sidecar path for first/last keyframes."""
    path = root / "h3.c"
    text = path.read_text(encoding="utf-8")
    if "using FL2VA I2V text sidecar" in text:
        return
    marker = "        size_t vision_cursor = 0;\n"
    branch = '''        if (text_sidecar_path) {
            h3_text_embedding_file_expectation expectation;
            uint8_t first_hash[32], last_hash[32];
            uint32_t *sidecar_ids = NULL;
            size_t sidecar_count = 0;
            memset(&expectation, 0, sizeof(expectation));
            expectation.mode = H3CSPEED_TEXT_EMBEDDING_MODE_FL2VA_I2V;
            expectation.keyframe_role =
                (params->first_frame ? H3CSPEED_TEXT_EMBEDDING_ROLE_FIRST : 0u) |
                (params->last_frame ? H3CSPEED_TEXT_EMBEDDING_ROLE_LAST : 0u);
            expectation.keyframe_count = (uint32_t)visual_count;
            expectation.keyframe_order = expectation.keyframe_role;
            expectation.first_resize_policy = H3CSPEED_TEXT_EMBEDDING_RESIZE_STRETCH;
            expectation.last_resize_policy = H3CSPEED_TEXT_EMBEDDING_RESIZE_COVER;
            expectation.render_width = (uint32_t)render_width;
            expectation.render_height = (uint32_t)render_height;
            if ((params->first_frame && !h3cspeed_sha256_file(
                    params->first_frame, first_hash, detail, sizeof(detail))) ||
                (params->last_frame && !h3cspeed_sha256_file(
                    params->last_frame, last_hash, detail, sizeof(detail)))) {
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            expectation.first_image_sha256 = params->first_frame ? first_hash : NULL;
            expectation.last_image_sha256 = params->last_frame ? last_hash : NULL;
            if (!h3cspeed_fl2va_build_token_ids(
                    tokenizer, prompt, condition_widths, condition_heights,
                    visual_count, &sidecar_ids, &sidecar_count,
                    detail, sizeof(detail))) {
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            h3_progress_emit(&progress, "text encoder", 0, 50);
            fprintf(stderr,
                "h3: using FL2VA I2V text sidecar %s; skipping native Qwen vision/text encoder\\n",
                text_sidecar_path);
            int text_ok = h3cspeed_text_embedding_load_file_ex(
                text_sidecar_path, prompt, sidecar_ids, sidecar_count,
                text_sidecar_sha256, &expectation, &text,
                detail, sizeof(detail));
            free(sidecar_ids);
            for (size_t image = 0; image < visual_count; image++) {
                free(condition_pixels[image]);
                condition_pixels[image] = NULL;
            }
            if (!text_ok) {
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
        } else {
'''
    if marker not in text:
        raise RuntimeError("unable to locate h3 visual text encoder branch")
    text = text.replace(marker, branch + marker, 1)
    close_marker = (
        "        for (size_t image = 0; image < vision_output_count; image++)\n"
        "            h3_vision_output_free(&vision_outputs[image]);\n"
        "    } else {\n"
        "        if (!h3_tokenizer_encode(tokenizer, prompt, 1, &ids, &token_count,")
    close_replacement = (
        "        for (size_t image = 0; image < vision_output_count; image++)\n"
        "            h3_vision_output_free(&vision_outputs[image]);\n"
        "        }\n"
        "    } else {\n"
        "        if (!h3_tokenizer_encode(tokenizer, prompt, 1, &ids, &token_count,")
    if close_marker not in text:
        raise RuntimeError("unable to locate h3 visual branch close")
    text = text.replace(close_marker, close_replacement, 1)
    path.write_text(text, encoding="utf-8", newline="\n")


def patch_sidecar_keyframe_hash_guard(root: Path) -> None:
    """Decode and hash I2V keyframes from one private immutable snapshot."""
    path = root / "h3.c"
    text = path.read_text(encoding="utf-8")
    if "h3cspeed_keyframe_snapshot first_snapshot" in text:
        return
    locals_marker = """    int decoder_is_cached = 0;
"""
    locals_replacement = """    int decoder_is_cached = 0;
    h3cspeed_keyframe_snapshot first_snapshot = {{0}, {0}, {0}};
    h3cspeed_keyframe_snapshot last_snapshot = {{0}, {0}, {0}};
    const char *first_frame_path = params->first_frame;
    const char *last_frame_path = params->last_frame;
"""
    if locals_marker not in text:
        raise RuntimeError("unable to locate keyframe snapshot locals")
    text = text.replace(locals_marker, locals_replacement, 1)
    paths_marker = """        goto cleanup;
    }
    conditioning_key = h3_conditioning_key(
"""
    paths_replacement = """        goto cleanup;
    }
    char detail[512];
    if (text_sidecar_path && params->first_frame) {
        if (!h3cspeed_keyframe_snapshot_create(
                params->first_frame, &first_snapshot,
                detail, sizeof(detail))) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
        first_frame_path = first_snapshot.path;
    }
    if (text_sidecar_path && params->last_frame) {
        if (!h3cspeed_keyframe_snapshot_create(
                params->last_frame, &last_snapshot,
                detail, sizeof(detail))) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
        last_frame_path = last_snapshot.path;
    }
    conditioning_key = h3_conditioning_key(
"""
    if paths_marker not in text:
        raise RuntimeError("unable to locate keyframe snapshot setup")
    text = text.replace(paths_marker, paths_replacement, 1)
    text = text.replace("    char detail[512];\n    if (conditioning_hit) {",
                        "    if (conditioning_hit) {", 1)
    text = text.replace(
        "params->first_frame, render_width, render_height,\n"
        "                    H3_IMAGE_FIT_STRETCH",
        "first_frame_path, render_width, render_height,\n"
        "                    H3_IMAGE_FIT_STRETCH", 1)
    text = text.replace(
        "params->last_frame, render_width, render_height,\n"
        "                    H3_IMAGE_FIT_COVER",
        "last_frame_path, render_width, render_height,\n"
        "                    H3_IMAGE_FIT_COVER", 1)
    hash_block = """            uint8_t first_hash[32], last_hash[32];
"""
    if hash_block not in text:
        raise RuntimeError("unable to locate sidecar hash locals")
    text = text.replace(hash_block, "", 1)
    old_hash = """            if ((params->first_frame && !h3cspeed_sha256_file(
                    params->first_frame, first_hash, detail, sizeof(detail))) ||
                (params->last_frame && !h3cspeed_sha256_file(
                    params->last_frame, last_hash, detail, sizeof(detail)))) {
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            expectation.first_image_sha256 = params->first_frame ? first_hash : NULL;
            expectation.last_image_sha256 = params->last_frame ? last_hash : NULL;
"""
    new_hash = """            expectation.first_image_sha256 =
                params->first_frame ? first_snapshot.sha256 : NULL;
            expectation.last_image_sha256 =
                params->last_frame ? last_snapshot.sha256 : NULL;
"""
    if old_hash not in text:
        raise RuntimeError("unable to bind sidecar to keyframe snapshot")
    text = text.replace(old_hash, new_hash, 1)
    cleanup_marker = """cleanup:
    free(conditioning_key);
"""
    cleanup_replacement = """cleanup:
    h3cspeed_keyframe_snapshot_discard(&first_snapshot);
    h3cspeed_keyframe_snapshot_discard(&last_snapshot);
    free(conditioning_key);
"""
    if cleanup_marker not in text:
        raise RuntimeError("unable to locate keyframe snapshot cleanup")
    text = text.replace(cleanup_marker, cleanup_replacement, 1)
    path.write_text(text, encoding="utf-8", newline="\n")


def patch_frame_anchor_allocation(root: Path) -> None:
    """Do not treat malloc(0) as an I2V frame-anchor allocation failure."""
    path = root / "h3.c"
    text = path.read_text(encoding="utf-8")
    old = ("            !condition_frames || !visual_reference_indices ||\n"
           "            !reference_visual_indices ||\n")
    new = ("            !condition_frames || !visual_reference_indices ||\n"
           "            (params->reference_count && !reference_visual_indices) ||\n")
    if new not in text:
        if old not in text:
            raise RuntimeError("unable to locate h3 visual allocation guard")
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8", newline="\n")


def patch_c_linkage(path: Path, include_marker: str) -> None:
    text = path.read_text(encoding="utf-8")
    opening = '#ifdef __cplusplus\nextern "C" {\n#endif\n\n'
    closing = '\n#ifdef __cplusplus\n}\n#endif\n'
    if opening in text:
        return
    if include_marker not in text:
        raise RuntimeError(f"unable to locate linkage insertion point in {path.name}")
    text = text.replace(include_marker, include_marker + opening, 1)
    guard = text.rfind("\n#endif")
    if guard < 0:
        raise RuntimeError(f"unable to locate include guard end in {path.name}")
    text = text[:guard] + closing + text[guard:]
    path.write_text(text, encoding="utf-8", newline="\n")


def patch_quantized_loader(root: Path) -> None:
    """Apply the audited quantized-loader overlay reproducibly.

    The source files remain pinned upstream inputs.  Their prepared variants
    are copied from versioned overlay templates so a fresh bootstrap cannot
    silently diverge from the tree used by the CUDA build and tests.
    """
    overlay = Path(__file__).resolve().parent / "upstream_overlay"
    files = (
        "h3_safetensors.h",
        "h3_safetensors.c",
        "h3_weights.h",
        "h3_weights.c",
        "h3_text_encoder.c",
        "h3_dit.c",
        "h3_dit_schedule.c",
        "h3_audio_vae.c",
    )
    for relative in files:
        source = overlay / relative
        if not source.is_file():
            raise RuntimeError(f"quantized overlay is missing {relative}")
        shutil.copyfile(source, root / relative)


def patch_tree(root: Path) -> None:
    replace_host_resize(root / "h3_host.c")
    patch_cli_name(root)
    patch_windows_stat(root)
    patch_ffmpeg_limits(root)
    patch_linux_random_seed(root)
    patch_text_embedding_sidecar(root)
    patch_text_embedding_i2v(root)
    patch_sidecar_keyframe_hash_guard(root)
    patch_frame_anchor_allocation(root)
    patch_quantized_loader(root)
    patch_c_linkage(root / "h3_gpu.h", "#include <stdint.h>\n\n")
    patch_c_linkage(root / "h3_metal.h", '#include "h3.h"\n\n')
    marker = root / ".h3cspeed-pinned-revision"
    marker.write_text(UPSTREAM_COMMIT + "\n", encoding="ascii", newline="\n")


def verify_prepared_tree(root: Path) -> None:
    if not PREPARED_GIT_BLOBS:
        raise RuntimeError("prepared-tree hashes are not configured")
    errors: list[str] = []
    tree_actual = tree_sha256(root)
    if tree_actual != PREPARED_TREE_SHA256:
        errors.append(
            "complete tree: expected "
            f"{PREPARED_TREE_SHA256}, got {tree_actual}"
        )
    for relative, expected in PREPARED_GIT_BLOBS.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"missing {relative}")
            continue
        actual = git_blob_sha(path.read_bytes())
        if actual != expected:
            errors.append(f"{relative}: expected {expected}, got {actual}")
    if errors:
        joined = "\n  - ".join(errors)
        raise RuntimeError("prepared upstream tree failed hash verification:\n  - " + joined)


def extract_archive(archive: Path, destination: Path) -> Path:
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.namelist()
        if not members:
            raise RuntimeError("upstream archive is empty")
        top = members[0].split("/", 1)[0]
        bundle.extractall(destination)
    root = destination / top
    if not root.is_dir():
        raise RuntimeError("unexpected upstream archive layout")
    return root


def bootstrap(project_root: Path, force: bool, archive_override: Path | None) -> Path:
    target = project_root / "third_party" / "h3"
    marker = target / ".h3cspeed-pinned-revision"
    if target.exists() and marker.exists() and marker.read_text().strip() == UPSTREAM_COMMIT:
        if not force:
            verify_prepared_tree(target)
            print(f"h3cspeed: upstream already prepared at {target}")
            return target
    if target.exists() and not force:
        raise RuntimeError(
            f"{target} already exists but is not the pinned tree; rerun with --force"
        )

    with tempfile.TemporaryDirectory(prefix="h3cspeed-bootstrap-") as temporary:
        temp = Path(temporary)
        archive = temp / "h3.zip"
        if archive_override:
            shutil.copy2(archive_override, archive)
        else:
            print(f"h3cspeed: downloading {UPSTREAM_REPO}@{UPSTREAM_COMMIT}")
            download(ARCHIVE_URL, archive)
        verify_archive(archive)
        extracted = extract_archive(archive, temp / "extract")
        verify_tree(extracted)
        patch_tree(extracted)
        verify_prepared_tree(extracted)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.parent / f".h3-staging-{uuid.uuid4().hex}"
        shutil.move(str(extracted), str(staging))
        previous = target.parent / f".h3-previous-{uuid.uuid4().hex}"
        try:
            if target.exists():
                target.rename(previous)
            staging.rename(target)
        except Exception:
            if not target.exists() and previous.exists():
                previous.rename(target)
            raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        if previous.exists():
            shutil.rmtree(previous)

    print(f"h3cspeed: prepared {target}")
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="h3cspeed project root",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="prepared source directory to verify (verify-only mode)",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="use a local h3.c ZIP archive instead of downloading",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        root = args.project_root.resolve()
        if args.verify_only:
            target = (args.source_dir.resolve() if args.source_dir
                      else root / "third_party" / "h3")
            marker = target / ".h3cspeed-pinned-revision"
            if not marker.is_file() or marker.read_text().strip() != UPSTREAM_COMMIT:
                raise RuntimeError("prepared upstream revision marker is missing or stale")
            verify_prepared_tree(target)
            print(f"h3cspeed: verified prepared upstream at {target}")
        else:
            bootstrap(root, args.force, args.archive)
    except Exception as exc:  # noqa: BLE001 - CLI should report one clear error.
        print(f"h3cspeed bootstrap failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
