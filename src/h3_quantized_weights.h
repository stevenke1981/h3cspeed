#ifndef H3CSPEED_QUANTIZED_WEIGHTS_H
#define H3CSPEED_QUANTIZED_WEIGHTS_H

#include "h3_gpu.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Internal overlay API. These functions deliberately stay outside h3_gpu.h so
 * the pinned public H3 backend ABI remains unchanged. */
h3_gpu_tensor *h3cspeed_gpu_tensor_load_i8_convrot(
    h3_gpu *gpu, const char *path, uint64_t file_offset, size_t elements,
    uint32_t group_size);

h3_gpu_tensor *h3cspeed_gpu_tensor_load_f16_as_bf16(
    h3_gpu *gpu, const char *path, uint64_t file_offset, size_t elements);
h3_gpu_tensor *h3cspeed_gpu_tensor_load_f16_as_f32(
    h3_gpu *gpu, const char *path, uint64_t file_offset, size_t elements);
int h3cspeed_gpu_tensor_read_f16_as_bf16(
    h3_gpu_tensor *destination, const char *path, uint64_t file_offset,
    size_t elements, char *error, size_t error_size);

int h3cspeed_gpu_tensor_read_i8_as_bf16(
    h3_gpu_tensor *destination, const char *weight_path,
    uint64_t weight_offset, const char *scale_path, uint64_t scale_offset,
    uint32_t rows, uint32_t columns, char *error, size_t error_size);

int h3cspeed_gpu_tensor_read_nvfp4_as_bf16(
    h3_gpu_tensor *destination, const char *packed_path,
    uint64_t packed_offset, const char *scale_path, uint64_t scale_offset,
    float tensor_scale, const char *pre_scale_path,
    uint64_t pre_scale_offset, uint32_t rows, uint32_t columns,
    char *error, size_t error_size);
int h3cspeed_gpu_mul_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *column_scale,
                          uint32_t rows, uint32_t columns);

#ifdef __cplusplus
}
#endif

#endif
