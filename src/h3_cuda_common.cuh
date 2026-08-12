#ifndef H3CSPEED_CUDA_COMMON_CUH
#define H3CSPEED_CUDA_COMMON_CUH

#include "h3_gpu.h"
#include "h3_offload_policy.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>

#if defined(_WIN32)
#include "h3_msvc_compat.h"
#endif

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

struct h3_gpu;

struct h3_gpu_tensor {
    h3_gpu *gpu;
    /* Device pointer. It is NULL while a read-only weight is offloaded. */
    void *data;
    /* Optional system-RAM copy for read-only file weights. */
    void *host_data;
    size_t elements;
    size_t bytes;
    size_t source_bytes;
    uint64_t source_offset;
    char *source_path;
    h3_gpu_dtype dtype;
    uint32_t u32_max;
    int u32_max_valid;
    int offloadable;
    int host_valid;
    int host_pinned;
    int source_streaming;
    /* Internal quantized-weight marker.  ConvRot weights were rotated offline
     * with a normalized regular Hadamard matrix; the matching activation
     * rotation is applied immediately before INT8 quantization.  This stays
     * outside h3_gpu.h so the public backend ABI is unchanged. */
    int convrot;
    uint32_t convrot_group_size;
    int in_lru;
    int in_host_lru;
    uint64_t pin_epoch;
    cudaEvent_t ready;
    int ready_valid;
    cudaEvent_t last_use;
    int last_use_valid;
    h3_gpu_tensor *lru_previous;
    h3_gpu_tensor *lru_next;
    h3_gpu_tensor *host_lru_previous;
    h3_gpu_tensor *host_lru_next;
    pthread_mutex_t lock;
};

struct h3_gpu {
    int device;
    cudaDeviceProp properties;
    cudaStream_t compute_stream;
    cudaStream_t upload_stream;
    cublasHandle_t blas;
    cublasLtHandle_t blas_lt;
    pthread_mutex_t lock;
    pthread_mutex_t scratch_lock;
    pthread_mutex_t offload_lock;
    pthread_mutex_t staging_lock;
    void *scratch;
    size_t scratch_bytes;
    void *staging;
    size_t staging_bytes;
    int staging_pinned;
    size_t device_live_bytes;
    size_t resident_weight_bytes;
    size_t host_cache_live_bytes;
    size_t pinned_host_live_bytes;
    size_t peak_resident_weight_bytes;
    size_t peak_host_cache_bytes;
    uint64_t operation_epoch;
    uint64_t offload_uploads;
    uint64_t offload_upload_bytes;
    uint64_t offload_evictions;
    uint64_t offload_evicted_bytes;
    uint64_t file_fallback_reads;
    uint64_t file_fallback_bytes;
    uint64_t host_cache_evictions;
    uint64_t host_cache_evicted_bytes;
    h3_gpu_tensor *lru_head;
    h3_gpu_tensor *lru_tail;
    h3_gpu_tensor *host_lru_head;
    h3_gpu_tensor *host_lru_tail;
    h3cspeed_offload_policy offload;
    char error[512];
    h3_gpu_stats stats;
    int profile_enabled;
    char profile_label[128];
    struct timespec profile_wall;
    cudaEvent_t profile_start;
    cudaEvent_t profile_mark;
};

static inline size_t h3cspeed_dtype_size(h3_gpu_dtype dtype) {
    switch (dtype) {
        case H3_GPU_F32: return sizeof(float);
        case H3_GPU_BF16: return sizeof(__nv_bfloat16);
        case H3_GPU_I8: return sizeof(int8_t);
        case H3_GPU_U32: return sizeof(uint32_t);
    }
    return 0;
}

static inline cudaDataType_t h3cspeed_cuda_dtype(h3_gpu_dtype dtype) {
    switch (dtype) {
        case H3_GPU_F32: return CUDA_R_32F;
        case H3_GPU_BF16: return CUDA_R_16BF;
        case H3_GPU_I8: return CUDA_R_8I;
        case H3_GPU_U32: return CUDA_R_32I;
    }
    return CUDA_R_32F;
}

static inline void h3cspeed_set_error(h3_gpu *gpu, const char *operation,
                                      const char *detail) {
    if (!gpu) return;
    pthread_mutex_lock(&gpu->lock);
    snprintf(gpu->error, sizeof(gpu->error), "%s%s%s",
             operation ? operation : "CUDA failure",
             detail ? ": " : "", detail ? detail : "");
    pthread_mutex_unlock(&gpu->lock);
}

static inline int h3cspeed_cuda_ok(h3_gpu *gpu, cudaError_t status,
                                   const char *operation) {
    if (status == cudaSuccess) return 1;
    h3cspeed_set_error(gpu, operation, cudaGetErrorString(status));
    return 0;
}

static inline int h3cspeed_cublas_ok(h3_gpu *gpu, cublasStatus_t status,
                                     const char *operation) {
    if (status == CUBLAS_STATUS_SUCCESS) return 1;
    const char *message = "unknown cuBLAS error";
    switch (status) {
        case CUBLAS_STATUS_NOT_INITIALIZED: message = "not initialized"; break;
        case CUBLAS_STATUS_ALLOC_FAILED: message = "allocation failed"; break;
        case CUBLAS_STATUS_INVALID_VALUE: message = "invalid value"; break;
        case CUBLAS_STATUS_ARCH_MISMATCH: message = "architecture mismatch"; break;
        case CUBLAS_STATUS_MAPPING_ERROR: message = "mapping error"; break;
        case CUBLAS_STATUS_EXECUTION_FAILED: message = "execution failed"; break;
        case CUBLAS_STATUS_INTERNAL_ERROR: message = "internal error"; break;
        case CUBLAS_STATUS_NOT_SUPPORTED: message = "not supported"; break;
        default: break;
    }
    h3cspeed_set_error(gpu, operation, message);
    return 0;
}

int h3cspeed_tensor_prepare(h3_gpu *gpu, h3_gpu_tensor *tensor);
void h3cspeed_operation_complete(h3_gpu *gpu);

static inline int h3cspeed_tensor_wait(h3_gpu *gpu, const h3_gpu_tensor *tensor) {
    return h3cspeed_tensor_prepare(
        gpu, const_cast<h3_gpu_tensor *>(tensor));
}

static inline int h3cspeed_tensor_record_upload(h3_gpu_tensor *tensor) {
    if (!tensor || !tensor->gpu) return 0;
    pthread_mutex_lock(&tensor->lock);
    cudaError_t status = cudaEventRecord(tensor->ready, tensor->gpu->upload_stream);
    if (status == cudaSuccess) tensor->ready_valid = 1;
    pthread_mutex_unlock(&tensor->lock);
    return h3cspeed_cuda_ok(tensor->gpu, status, "cudaEventRecord(upload)");
}

static inline int h3cspeed_launch_ok(h3_gpu *gpu, const char *operation) {
    return h3cspeed_cuda_ok(gpu, cudaGetLastError(), operation);
}

static inline unsigned h3cspeed_blocks(size_t elements, unsigned threads = 256) {
    if (!threads) return 1;
    size_t blocks = elements / threads + (elements % threads != 0);
    if (!blocks) blocks = 1;
    return (unsigned)(blocks > 65535 ? 65535 : blocks);
}

static inline int h3cspeed_size_mul(size_t left, size_t right,
                                    size_t *result) {
    if (!result || (left && right > SIZE_MAX / left)) return 0;
    *result = left * right;
    return 1;
}

static inline int h3cspeed_size_mul3(size_t first, size_t second,
                                     size_t third, size_t *result) {
    size_t intermediate = 0;
    return h3cspeed_size_mul(first, second, &intermediate) &&
           h3cspeed_size_mul(intermediate, third, result);
}

static inline int h3cspeed_size_mul5(size_t first, size_t second,
                                     size_t third, size_t fourth,
                                     size_t fifth, size_t *result) {
    size_t value = 0;
    return h3cspeed_size_mul(first, second, &value) &&
           h3cspeed_size_mul(value, third, &value) &&
           h3cspeed_size_mul(value, fourth, &value) &&
           h3cspeed_size_mul(value, fifth, result);
}

static inline void h3cspeed_count_direct(h3_gpu *gpu) {
    pthread_mutex_lock(&gpu->lock);
    gpu->stats.direct_dispatches++;
    pthread_mutex_unlock(&gpu->lock);
    h3cspeed_operation_complete(gpu);
}

static inline void h3cspeed_count_linear(h3_gpu *gpu) {
    pthread_mutex_lock(&gpu->lock);
    gpu->stats.mps_linear_dispatches++;
    pthread_mutex_unlock(&gpu->lock);
    h3cspeed_operation_complete(gpu);
}

static inline void h3cspeed_count_conv(h3_gpu *gpu) {
    pthread_mutex_lock(&gpu->lock);
    gpu->stats.mps_conv_dispatches++;
    pthread_mutex_unlock(&gpu->lock);
    h3cspeed_operation_complete(gpu);
}

static inline void h3cspeed_count_sdpa(h3_gpu *gpu) {
    pthread_mutex_lock(&gpu->lock);
    gpu->stats.mps_sdpa_dispatches++;
    pthread_mutex_unlock(&gpu->lock);
    h3cspeed_operation_complete(gpu);
}

static inline __device__ float h3cspeed_device_load(const void *data,
                                                     h3_gpu_dtype dtype,
                                                     size_t index) {
    switch (dtype) {
        case H3_GPU_F32: return static_cast<const float *>(data)[index];
        case H3_GPU_BF16: return __bfloat162float(
            static_cast<const __nv_bfloat16 *>(data)[index]);
        case H3_GPU_I8: return static_cast<float>(
            static_cast<const int8_t *>(data)[index]);
        case H3_GPU_U32: return static_cast<float>(
            static_cast<const uint32_t *>(data)[index]);
    }
    return 0.0f;
}

static inline __device__ void h3cspeed_device_store(void *data,
                                                    h3_gpu_dtype dtype,
                                                    size_t index,
                                                    float value) {
    switch (dtype) {
        case H3_GPU_F32:
            static_cast<float *>(data)[index] = value;
            break;
        case H3_GPU_BF16:
            static_cast<__nv_bfloat16 *>(data)[index] = __float2bfloat16_rn(value);
            break;
        case H3_GPU_I8: {
            float rounded = nearbyintf(value);
            rounded = fminf(127.0f, fmaxf(-127.0f, rounded));
            static_cast<int8_t *>(data)[index] = static_cast<int8_t>(rounded);
            break;
        }
        case H3_GPU_U32:
            static_cast<uint32_t *>(data)[index] = value < 0.0f ? 0u :
                static_cast<uint32_t>(value);
            break;
    }
}

void *h3cspeed_scratch_reserve(h3_gpu *gpu, size_t bytes);
int h3cspeed_linear(h3_gpu *gpu, h3_gpu_tensor *output, size_t output_offset,
                    const h3_gpu_tensor *input, size_t input_offset,
                    const h3_gpu_tensor *weight, const h3_gpu_tensor *bias,
                    uint32_t rows, uint32_t input_dim, uint32_t output_dim);
int h3cspeed_quantize_rows(h3_gpu *gpu, h3_gpu_tensor *quantized,
                          h3_gpu_tensor *scales, const h3_gpu_tensor *input,
                          uint32_t rows, uint32_t width, int head_major,
                          uint32_t heads, uint32_t head_dim);
int h3cspeed_sdpa(h3_gpu *gpu, h3_gpu_tensor *output,
                  const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                  const h3_gpu_tensor *value, uint32_t batch,
                  uint32_t sequence, uint32_t query_heads,
                  uint32_t kv_heads, uint32_t head_dim, float scale,
                  int causal, int output_head_major, int input_head_major,
                  int scale_query_bf16);

#endif
