#!/usr/bin/env python3
"""Run source-level syntax checks without requiring a CUDA toolkit.

This is not a replacement for an nvcc build. It asks Clang to parse CUDA host
and kernel syntax against small declaration-only stubs, then parses the portable
C tokenizer against ICU and a yyjson API stub. It catches malformed kernels,
missing braces, many bad calls, and C/C++ linkage mistakes early.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap


def extract_extern_c_signatures(paths: list[Path]) -> list[str]:
    signatures: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        position = 0
        while True:
            start = text.find('extern "C"', position)
            if start < 0:
                break
            cursor = start + len('extern "C"')
            parentheses = 0
            saw_parenthesis = False
            in_string = False
            escaped = False
            while cursor < len(text):
                character = text[cursor]
                if in_string:
                    if escaped:
                        escaped = False
                    elif character == "\\":
                        escaped = True
                    elif character == '"':
                        in_string = False
                elif character == '"':
                    in_string = True
                elif character == "(":
                    parentheses += 1
                    saw_parenthesis = True
                elif character == ")":
                    parentheses -= 1
                elif character == "{" and saw_parenthesis and parentheses == 0:
                    break
                cursor += 1
            if cursor >= len(text):
                raise RuntimeError(f"cannot parse extern C definition in {path}")
            signature = text[start + len('extern "C"'):cursor].strip()
            signature = re.sub(r"\s+", " ", signature)
            signatures.append(signature + ";")
            position = cursor + 1
    return signatures


def write_host_api_stubs(directory: Path) -> None:
    (directory / "h3.h").write_text(textwrap.dedent("""\
        #ifndef H3_H
        #define H3_H
        #include <stddef.h>
        #include <stdint.h>
        typedef struct {
            char name[128];
            char architecture[128];
            uint64_t physical_memory;
            uint64_t recommended_working_set;
            uint64_t max_buffer_length;
            int apple_gpu_family;
            int metal4;
            int unified_memory;
        } h3_device_info;
        #endif
        """), encoding="utf-8")
    (directory / "h3_metal.h").write_text(textwrap.dedent("""\
        #ifndef H3_METAL_H
        #define H3_METAL_H
        #include "h3.h"
        #ifdef __cplusplus
        extern "C" {
        #endif
        int h3_metal_probe(h3_device_info *, char *, size_t);
        #ifdef __cplusplus
        }
        #endif
        #endif
        """), encoding="utf-8")


def write_cuda_stubs(directory: Path, project_root: Path) -> None:
    cuda_sources = sorted((project_root / "src").glob("*.cu"))
    signatures = extract_extern_c_signatures(cuda_sources)
    h3_gpu = """\
#ifndef H3_GPU_H
#define H3_GPU_H
#include <stddef.h>
#include <stdint.h>
#include "h3.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct h3_gpu h3_gpu;
typedef struct h3_gpu_tensor h3_gpu_tensor;
typedef enum { H3_GPU_F32=0, H3_GPU_BF16, H3_GPU_I8, H3_GPU_U32 } h3_gpu_dtype;
typedef struct {
    uint64_t allocated_bytes, live_bytes, peak_live_bytes, tensor_allocations;
    uint64_t direct_dispatches, mps_linear_dispatches, mps_conv_dispatches;
    uint64_t mps_sdpa_dispatches, blit_copies, submissions;
    double command_encode_seconds, command_wait_seconds, gpu_seconds;
} h3_gpu_stats;
""" + "\n".join(signatures) + """
#ifdef __cplusplus
}
#endif
#endif
"""
    (directory / "h3_gpu.h").write_text(h3_gpu, encoding="utf-8")
    write_host_api_stubs(directory)
    (directory / "cuda_runtime.h").write_text(textwrap.dedent(r"""\
        #ifndef CUDA_RUNTIME_H
        #define CUDA_RUNTIME_H
        #include <stddef.h>
        #include <stdint.h>
        #include <math.h>
        /* Declaration-only CUDA attributes keep this parser independent of a
         * toolkit while retaining launch syntax for kernel definitions. */
        #define __global__ __attribute__((global))
        #define __device__ __attribute__((device))
        #define __host__ __attribute__((host))
        #define __shared__ __attribute__((shared))
        #define __forceinline__ inline __attribute__((always_inline))
        struct uint3 { unsigned x,y,z; };
        struct dim3 {
            unsigned x,y,z;
            constexpr dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){}
            constexpr dim3(const uint3 &v):x(v.x),y(v.y),z(v.z){}
        };
        extern const __attribute__((device_builtin)) uint3 threadIdx;
        extern const __attribute__((device_builtin)) uint3 blockIdx;
        extern const __attribute__((device_builtin)) dim3 blockDim;
        extern const __attribute__((device_builtin)) dim3 gridDim;
        extern const __attribute__((device_builtin)) int warpSize;
        extern "C" int cudaConfigureCall(dim3, dim3, size_t = 0, void * = nullptr);
        extern "C" int cudaSetupArgument(const void *, size_t, size_t);
        extern "C" int cudaLaunch(const void *);
        extern __device__ float rsqrtf(float);
        extern __device__ float expf(float);
        extern __device__ float exp2f(float);
        extern __device__ float sqrtf(float);
        extern __device__ float nearbyintf(float);
        extern __device__ float fminf(float, float);
        extern __device__ float fmaxf(float, float);
        extern __device__ float tanhf(float);
        extern __device__ float sinf(float);
        extern __device__ float cosf(float);
        extern __device__ float logf(float);
        extern __device__ float fabsf(float);
        extern __device__ float erff(float);
        extern __device__ float fmaf(float, float, float);
        extern __device__ int __dp4a(int, int, int);
        typedef int cudaError_t;
        typedef void *cudaStream_t;
        typedef void *cudaEvent_t;
        typedef int cudaDataType_t;
        static const cudaError_t cudaSuccess=0;
        #ifndef CUDART_INF_F
        #define CUDART_INF_F (__builtin_inff())
        #endif
        static const unsigned cudaEventDisableTiming=1;
        static const unsigned cudaStreamNonBlocking=1;
        static const unsigned cudaHostAllocPortable=1;
        enum cudaMemcpyKind {
            cudaMemcpyHostToDevice,
            cudaMemcpyDeviceToHost,
            cudaMemcpyDeviceToDevice
        };
        static const cudaDataType_t CUDA_R_32F=0, CUDA_R_16BF=1;
        static const cudaDataType_t CUDA_R_8I=2, CUDA_R_32I=3;
        struct cudaDeviceProp { char name[256]; size_t totalGlobalMem; int major, minor; int managedMemory; };
        extern "C" {
        cudaError_t cudaGetDevice(int *);
        cudaError_t cudaSetDevice(int);
        cudaError_t cudaGetDeviceProperties(cudaDeviceProp *, int);
        cudaError_t cudaMemGetInfo(size_t *, size_t *);
        const char *cudaGetErrorString(cudaError_t);
        cudaError_t cudaGetLastError(void);
        cudaError_t cudaStreamCreate(cudaStream_t *);
        cudaError_t cudaStreamCreateWithFlags(cudaStream_t *, unsigned);
        cudaError_t cudaStreamDestroy(cudaStream_t);
        cudaError_t cudaStreamSynchronize(cudaStream_t);
        cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned);
        cudaError_t cudaEventCreate(cudaEvent_t *);
        cudaError_t cudaEventCreateWithFlags(cudaEvent_t *, unsigned);
        cudaError_t cudaEventDestroy(cudaEvent_t);
        cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = nullptr);
        cudaError_t cudaEventSynchronize(cudaEvent_t);
        cudaError_t cudaEventElapsedTime(float *, cudaEvent_t, cudaEvent_t);
        cudaError_t cudaMalloc(void **, size_t);
        cudaError_t cudaFree(void *);
        cudaError_t cudaHostAlloc(void **, size_t, unsigned);
        cudaError_t cudaFreeHost(void *);
        cudaError_t cudaMemcpy(void *, const void *, size_t, cudaMemcpyKind);
        cudaError_t cudaMemcpyAsync(void *, const void *, size_t, cudaMemcpyKind, cudaStream_t);
        cudaError_t cudaMemsetAsync(void *, int, size_t, cudaStream_t);
        }
        #endif
        """), encoding="utf-8")
    (directory / "cuda_fp16.h").write_text(textwrap.dedent(r"""\
        #ifndef CUDA_FP16_H
        #define CUDA_FP16_H
        #include <stdint.h>
        struct __half { uint16_t bits; };
        extern __device__ float __half2float(__half);
        extern __device__ __half __ushort_as_half(uint16_t);
        #endif
        """), encoding="utf-8")
    # Clang's CUDA host wrapper otherwise pulls the installed MSVC STL and
    # its compiler-specific intrinsic headers.  The kernels only need these
    # two comparisons, so keep the syntax check declaration-only and portable.
    if os.name == "nt":
        (directory / "algorithm").write_text(textwrap.dedent(r"""\
            #ifndef H3CSPEED_SYNTAX_ALGORITHM_H
            #define H3CSPEED_SYNTAX_ALGORITHM_H
            namespace std {
            template <typename T>
            constexpr const T &max(const T &left, const T &right) {
                return left < right ? right : left;
            }
            template <typename T>
            constexpr const T &min(const T &left, const T &right) {
                return right < left ? right : left;
            }
            }
            #endif
            """), encoding="utf-8")
    if os.name == "nt":
        (directory / "limits").write_text(textwrap.dedent(r"""\
            #ifndef H3CSPEED_SYNTAX_LIMITS_H
            #define H3CSPEED_SYNTAX_LIMITS_H
            namespace std {
            template <typename T>
            struct numeric_limits {
                static constexpr T max() {
                    return static_cast<T>((~static_cast<T>(0)) >> 1);
                }
            };
            }
            #endif
            """), encoding="utf-8")
    (directory / "cuda_bf16.h").write_text(textwrap.dedent(r"""\
        #ifndef CUDA_BF16_H
        #define CUDA_BF16_H
        #include "cuda_runtime.h"
        struct __nv_bfloat16 { uint16_t x; };
        __host__ __device__ inline float __bfloat162float(__nv_bfloat16) { return 0.0f; }
        __host__ __device__ inline __nv_bfloat16 __float2bfloat16_rn(float) { return {0}; }
        #endif
        """), encoding="utf-8")
    (directory / "cublas_v2.h").write_text(textwrap.dedent(r"""\
        #ifndef CUBLAS_V2_H
        #define CUBLAS_V2_H
        #include "cuda_runtime.h"
        typedef void *cublasHandle_t;
        typedef int cublasStatus_t;
        typedef int cublasOperation_t;
        typedef int cublasComputeType_t;
        typedef int cublasGemmAlgo_t;
        typedef int cublasMath_t;
        static const cublasStatus_t CUBLAS_STATUS_SUCCESS=0;
        static const cublasStatus_t CUBLAS_STATUS_NOT_INITIALIZED=1;
        static const cublasStatus_t CUBLAS_STATUS_ALLOC_FAILED=3;
        static const cublasStatus_t CUBLAS_STATUS_INVALID_VALUE=7;
        static const cublasStatus_t CUBLAS_STATUS_ARCH_MISMATCH=8;
        static const cublasStatus_t CUBLAS_STATUS_MAPPING_ERROR=11;
        static const cublasStatus_t CUBLAS_STATUS_EXECUTION_FAILED=13;
        static const cublasStatus_t CUBLAS_STATUS_INTERNAL_ERROR=14;
        static const cublasStatus_t CUBLAS_STATUS_NOT_SUPPORTED=15;
        static const cublasOperation_t CUBLAS_OP_N=0, CUBLAS_OP_T=1;
        static const cublasComputeType_t CUBLAS_COMPUTE_32F=0, CUBLAS_COMPUTE_32I=1;
        static const cublasGemmAlgo_t CUBLAS_GEMM_DEFAULT_TENSOR_OP=0;
        static const cublasMath_t CUBLAS_TF32_TENSOR_OP_MATH=0;
        extern "C" {
        cublasStatus_t cublasCreate(cublasHandle_t *);
        cublasStatus_t cublasDestroy(cublasHandle_t);
        cublasStatus_t cublasSetStream(cublasHandle_t, cudaStream_t);
        cublasStatus_t cublasSetMathMode(cublasHandle_t, cublasMath_t);
        cublasStatus_t cublasGemmEx(cublasHandle_t, ...);
        }
        #endif
        """), encoding="utf-8")
    (directory / "cublasLt.h").write_text(textwrap.dedent(r"""\
        #ifndef CUBLAS_LT_H
        #define CUBLAS_LT_H
        typedef void *cublasLtHandle_t;
        typedef int cublasStatus_t;
        extern "C" {
        cublasStatus_t cublasLtCreate(cublasLtHandle_t *);
        cublasStatus_t cublasLtDestroy(cublasLtHandle_t);
        }
        #endif
        """), encoding="utf-8")


def write_c_stubs(directory: Path) -> None:
    write_host_api_stubs(directory)
    (directory / "h3_tokenizer.h").write_text(textwrap.dedent(r"""\
        #ifndef H3_TOKENIZER_H
        #define H3_TOKENIZER_H
        #include <stddef.h>
        #include <stdint.h>
        #define H3_PAD_TOKEN_ID UINT32_C(151643)
        typedef struct h3_tokenizer h3_tokenizer;
        h3_tokenizer *h3_tokenizer_load(const char *, char *, size_t);
        void h3_tokenizer_free(h3_tokenizer *);
        int h3_tokenizer_encode(const h3_tokenizer *, const char *, int,
                                uint32_t **, size_t *, char *, size_t);
        void h3_tokenizer_ids_free(uint32_t *);
        char *h3_tokenizer_decode(const h3_tokenizer *, const uint32_t *,
                                  size_t, char *, size_t);
        #endif
        """), encoding="utf-8")
    (directory / "yyjson.h").write_text(textwrap.dedent(r"""\
        #ifndef YYJSON_H
        #define YYJSON_H
        #include <stdbool.h>
        #include <stddef.h>
        #include <stdint.h>
        typedef struct yyjson_val yyjson_val;
        typedef struct yyjson_doc yyjson_doc;
        typedef struct { size_t idx, max; yyjson_val *cur; } yyjson_obj_iter;
        typedef struct { size_t idx, max; yyjson_val *cur; } yyjson_arr_iter;
        typedef struct { const char *msg; size_t pos; int code; } yyjson_read_err;
        yyjson_val *yyjson_obj_get(yyjson_val *, const char *);
        bool yyjson_is_obj(yyjson_val *);
        bool yyjson_is_arr(yyjson_val *);
        bool yyjson_is_str(yyjson_val *);
        bool yyjson_is_uint(yyjson_val *);
        bool yyjson_is_null(yyjson_val *);
        const char *yyjson_get_str(yyjson_val *);
        uint64_t yyjson_get_uint(yyjson_val *);
        bool yyjson_get_bool(yyjson_val *);
        yyjson_obj_iter yyjson_obj_iter_with(yyjson_val *);
        yyjson_val *yyjson_obj_iter_next(yyjson_obj_iter *);
        yyjson_val *yyjson_obj_iter_get_val(yyjson_val *);
        yyjson_arr_iter yyjson_arr_iter_with(yyjson_val *);
        yyjson_val *yyjson_arr_iter_next(yyjson_arr_iter *);
        size_t yyjson_arr_size(yyjson_val *);
        yyjson_val *yyjson_arr_get_first(yyjson_val *);
        yyjson_val *yyjson_arr_get(yyjson_val *, size_t);
        yyjson_doc *yyjson_read_file(const char *, uint32_t, const void *, yyjson_read_err *);
        yyjson_val *yyjson_doc_get_root(yyjson_doc *);
        void yyjson_doc_free(yyjson_doc *);
        #endif
        """), encoding="utf-8")


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True, env={**dict(__import__('os').environ), "TERM": "dumb"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-cuda", action="store_true")
    parser.add_argument("--skip-tokenizer", action="store_true")
    parser.add_argument("--skip-host-c", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()

    try:
        with tempfile.TemporaryDirectory(prefix="h3cspeed-syntax-") as temporary:
            stub = Path(temporary)
            if not args.skip_cuda:
                clang = shutil.which("clang++")
                if not clang:
                    raise RuntimeError("clang++ is required for CUDA syntax lint")
                write_cuda_stubs(stub, root)
                for source in sorted((root / "src").glob("*.cu")):
                    platform_flags = []
                    if os.name == "nt":
                        # Clang's CUDA host parser otherwise omits the SSE
                        # target features expected by MSVC's intrinsic headers.
                        platform_flags = [
                            "-msse2",
                            "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH=1",
                            "-Wno-deprecated-declarations",
                        ]
                    run([
                        clang, "-x", "cuda", "--cuda-host-only",
                        "-nocudainc", "-nocudalib", "-std=c++17",
                        "-Wall", "-Wextra", "-Wpedantic", "-Wshadow",
                        "-Wconversion", "-Wno-sign-conversion",
                        "-Wno-unused-parameter", "-Werror",
                        *platform_flags,
                        f"-I{stub}", f"-I{root / 'src'}", "-fsyntax-only",
                        str(source),
                    ], root)
                print("CUDA source syntax: passed")

            write_c_stubs(stub)

            if not args.skip_host_c:
                compiler = shutil.which("cc") or shutil.which("clang")
                if not compiler:
                    raise RuntimeError("cc or clang is required for host C syntax lint")
                for source in (
                    root / "src/h3_offload_policy.c",
                    root / "src/h3_cuda_info_main.c",
                ):
                    run([
                        compiler, "-std=c11", "-D_GNU_SOURCE",
                        *(["-D_CRT_SECURE_NO_WARNINGS=1"] if os.name == "nt" else []),
                        "-Wall", "-Wextra", "-Wpedantic", "-Wshadow",
                        "-Wconversion", "-Wsign-conversion", "-Werror",
                        f"-I{stub}", f"-I{root / 'src'}", "-fsyntax-only",
                        str(source),
                    ], root)
                print("offload policy/CUDA info syntax: passed")

            if not args.skip_tokenizer:
                compiler = shutil.which("cc") or shutil.which("clang")
                pkg_config = shutil.which("pkg-config")
                if not compiler:
                    raise RuntimeError("cc or clang is required for tokenizer syntax lint")
                if os.name == "nt":
                    icu_root = Path(os.environ.get(
                        "H3CSPEED_ICU_ROOT", root / "third_party" / "icu"))
                    header = icu_root / "include" / "unicode" / "uchar.h"
                    if not header.is_file():
                        raise RuntimeError(f"bundled ICU headers not found at {icu_root}")
                    icu = [f"-I{icu_root / 'include'}"]
                else:
                    if not pkg_config:
                        raise RuntimeError("pkg-config is required for tokenizer syntax lint")
                    icu = subprocess.run(
                        [pkg_config, "--cflags", "icu-uc"], check=True,
                        text=True, stdout=subprocess.PIPE,
                    ).stdout.split()
                run([
                    compiler, "-std=c11", "-D_GNU_SOURCE",
                    *(["-D_CRT_SECURE_NO_WARNINGS=1"] if os.name == "nt" else []),
                    "-Wall", "-Wextra",
                    "-Wpedantic", "-Wshadow", "-Wno-conversion",
                    f"-I{stub}", *icu, "-fsyntax-only",
                    str(root / "src/h3_tokenizer_portable.c"),
                ], root)
                print("portable tokenizer syntax: passed")
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"source syntax lint failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
