#include "h3_gpu.h"
#include "h3_quantized_weights.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", \
                __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static float bf16_to_f32(uint16_t bits) {
    uint32_t value = (uint32_t)bits << 16;
    float result;
    memcpy(&result, &value, sizeof(result));
    return result;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static int close_f32(float actual, float expected) {
    /* Readers promise BF16 rounding at the conversion boundary.  Compare to
     * the independently rounded oracle rather than using a fixed absolute
     * tolerance that would be too strict at larger exponents. */
    return fabsf(actual - bf16_to_f32(f32_to_bf16(expected))) <= 0.001f;
}

static int no_cuda_device(const char *error) {
    char normalized[512];
    size_t length = 0;
    if (!error) return 0;
    while (error[length] && length + 1 < sizeof(normalized)) {
        normalized[length] = (char)tolower((unsigned char)error[length]);
        length++;
    }
    normalized[length] = '\0';
    return strstr(normalized, "no cuda-capable device") != NULL ||
           strstr(normalized, "cuda_error_no_device") != NULL ||
           strstr(normalized, "cudaerrornodevice") != NULL;
}

static int write_file(const char *path, const void *data, size_t bytes) {
    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    int ok = fwrite(data, 1, bytes, file) == bytes;
    if (fclose(file) != 0) ok = 0;
    if (!ok) remove(path);
    return ok;
}

static float e4m3_value(uint8_t bits) {
    uint32_t sign = bits >> 7;
    uint32_t exponent = (bits >> 3) & 15u;
    uint32_t mantissa = bits & 7u;
    float value;
    if (exponent == 0)
        value = (float)mantissa * (1.0f / 512.0f);
    else if (exponent == 15)
        value = mantissa == 7u ? 0.0f :
            (1.0f + (float)mantissa / 8.0f) * 256.0f;
    else
        value = (1.0f + (float)mantissa / 8.0f) *
                ldexpf(1.0f, (int)exponent - 7);
    return sign ? -value : value;
}

static float e2m1_value(uint8_t value) {
    static const float table[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    return table[value & 15u];
}

static uint16_t comfy_nvfp4_bf16(float e2m1, float tensor_scale,
                                 float block_scale) {
    float global_bf16 = bf16_to_f32(f32_to_bf16(tensor_scale));
    float block_bf16 = bf16_to_f32(f32_to_bf16(block_scale));
    float total_scale = bf16_to_f32(
        f32_to_bf16(global_bf16 * block_bf16));
    return f32_to_bf16(e2m1 * total_scale);
}

static size_t blocked_scale_index(uint32_t row, uint32_t block,
                                  uint32_t blocks_per_row) {
    uint32_t row_group = row / 128u;
    uint32_t row_in_group = row & 127u;
    uint32_t column_group = block / 4u;
    uint32_t column_in_group = block & 3u;
    uint32_t row32 = row_in_group & 31u;
    uint32_t column4 = row_in_group / 32u;
    uint32_t groups = (blocks_per_row + 3u) / 4u;
    return ((size_t)row_group * groups + column_group) * 512u +
           (size_t)row32 * 16u + (size_t)column4 * 4u + column_in_group;
}

static int test_nvfp4(h3_gpu *gpu) {
    enum { ROWS = 128, COLUMNS = 64, BLOCKS = COLUMNS / 16 };
    const float tensor_scale = 0.0005958193796686828f;
    const char *packed_path = "h3cspeed_test_nvfp4_packed.bin";
    const char *scale_path = "h3cspeed_test_nvfp4_scale.bin";
    const char *pre_path = "h3cspeed_test_nvfp4_pre.bin";
    const size_t packed_bytes = (size_t)ROWS * COLUMNS / 2u;
    const size_t scale_bytes = (size_t)ROWS * BLOCKS;
    const size_t pre_bytes = (size_t)COLUMNS * sizeof(uint16_t);
    uint8_t *packed = (uint8_t *)malloc(packed_bytes);
    uint8_t *logical_scales = (uint8_t *)malloc(scale_bytes);
    uint8_t *blocked_scales = (uint8_t *)calloc(1, scale_bytes);
    uint16_t *pre = (uint16_t *)malloc(COLUMNS * sizeof(uint16_t));
    uint16_t *actual = (uint16_t *)malloc((size_t)ROWS * COLUMNS * sizeof(uint16_t));
    CHECK(packed && logical_scales && blocked_scales && pre && actual);
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t column = 0; column < COLUMNS; column += 2) {
            uint8_t high = (uint8_t)((row + column) & 15u);
            uint8_t low = (uint8_t)((row * 3u + column + 1u) & 15u);
            packed[(size_t)row * (COLUMNS / 2u) + column / 2u] =
                (uint8_t)((high << 4) | low);
        }
        for (uint32_t block = 0; block < BLOCKS; block++) {
            /* Include non-power-of-two E4M3FN scales.  These are the values
             * that distinguish Comfy-kitchen's BF16 scale-product boundary
             * from a single FP32 product followed by BF16 conversion. */
            static const uint8_t values[4] = {0x3bu, 0x43u, 0x4bu, 0x56u};
            logical_scales[(size_t)row * BLOCKS + block] =
                values[(row + block) & 3u];
        }
    }
    for (uint32_t row = 0; row < ROWS; row++)
        for (uint32_t block = 0; block < BLOCKS; block++)
            blocked_scales[blocked_scale_index(row, block, BLOCKS)] =
                logical_scales[(size_t)row * BLOCKS + block];
    for (uint32_t column = 0; column < COLUMNS; column++) {
        float value = column % 3u == 0 ? 0.5f :
                      (column % 3u == 1 ? 1.0f : 1.5f);
        pre[column] = f32_to_bf16(value);
    }
    CHECK(write_file(packed_path, packed, packed_bytes));
    CHECK(write_file(scale_path, blocked_scales, scale_bytes));
    CHECK(write_file(pre_path, pre, pre_bytes));
    h3_gpu_tensor *destination = h3_gpu_tensor_new_bf16(
        gpu, (size_t)ROWS * COLUMNS);
    CHECK(destination);
    char error[512] = {0};
    CHECK(h3cspeed_gpu_tensor_read_nvfp4_as_bf16(
        destination, packed_path, 0, scale_path, 0, tensor_scale,
        pre_path, 0, ROWS, COLUMNS, error, sizeof(error)));
    CHECK(h3_gpu_tensor_read_bf16(
        destination, actual, (size_t)ROWS * COLUMNS));
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t column = 0; column < COLUMNS; column++) {
            uint8_t packed_value = packed[(size_t)row * (COLUMNS / 2u) +
                                          column / 2u];
            uint8_t nibble = (column & 1u) ? packed_value & 15u :
                                             packed_value >> 4;
            uint16_t expected = comfy_nvfp4_bf16(
                e2m1_value(nibble), tensor_scale,
                e4m3_value(logical_scales[
                    (size_t)row * BLOCKS + column / 16u]));
            CHECK(actual[(size_t)row * COLUMNS + column] == expected);
        }
    }
    h3_gpu_tensor_free(destination);
    free(actual); free(pre); free(blocked_scales); free(logical_scales); free(packed);
    remove(pre_path); remove(scale_path); remove(packed_path);
    return 0;
}

static int test_f16_and_i8(h3_gpu *gpu) {
    const char *f16_path = "h3cspeed_test_f16.bin";
    const char *i8_path = "h3cspeed_test_i8.bin";
    const char *scale_path = "h3cspeed_test_i8_scale.bin";
    const uint16_t f16_values[4] = {0x3c00u, 0xc000u, 0x3555u, 0x0000u};
    const int8_t i8_values[8] = {1, -2, 3, -4, -5, 6, 7, -8};
    const float scales[2] = {0.5f, -0.25f};
    uint16_t f16_actual[4] = {0};
    uint16_t i8_actual[8] = {0};
    CHECK(write_file(f16_path, f16_values, sizeof(f16_values)));
    CHECK(write_file(i8_path, i8_values, sizeof(i8_values)));
    CHECK(write_file(scale_path, scales, sizeof(scales)));
    h3_gpu_tensor *f16 = h3_gpu_tensor_new_bf16(gpu, 4);
    h3_gpu_tensor *i8 = h3_gpu_tensor_new_bf16(gpu, 8);
    CHECK(f16 && i8);
    char error[512] = {0};
    CHECK(h3cspeed_gpu_tensor_read_f16_as_bf16(
        f16, f16_path, 0, 4, error, sizeof(error)));
    CHECK(h3cspeed_gpu_tensor_read_i8_as_bf16(
        i8, i8_path, 0, scale_path, 0, 2, 4, error, sizeof(error)));
    CHECK(h3_gpu_tensor_read_bf16(f16, f16_actual, 4));
    CHECK(h3_gpu_tensor_read_bf16(i8, i8_actual, 8));
    for (size_t index = 0; index < 4; index++) {
        float expected = index == 0 ? 1.0f : index == 1 ? -2.0f :
                         index == 2 ? 0.33325195f : 0.0f;
        CHECK(close_f32(bf16_to_f32(f16_actual[index]), expected));
    }
    for (size_t index = 0; index < 8; index++) {
        float expected = (float)i8_values[index] * scales[index / 4];
        CHECK(close_f32(bf16_to_f32(i8_actual[index]), expected));
    }
    h3_gpu_tensor_free(i8); h3_gpu_tensor_free(f16);
    remove(scale_path); remove(i8_path); remove(f16_path);
    return 0;
}

static int test_repeated_converted_weight_eviction(h3_gpu *gpu) {
    enum { ROWS = 4096, COLUMNS = 2048, DESTINATIONS = 9 };
    const char *weight_path = "h3cspeed_test_i8_eviction.bin";
    const char *scale_path = "h3cspeed_test_i8_eviction_scale.bin";
    const size_t elements = (size_t)ROWS * COLUMNS;
    int8_t *weights = (int8_t *)malloc(elements);
    float *scales = (float *)malloc((size_t)ROWS * sizeof(float));
    uint16_t *actual = (uint16_t *)malloc(elements * sizeof(uint16_t));
    h3_gpu_tensor *destinations[DESTINATIONS] = {0};
    CHECK(weights && scales && actual);
    for (size_t index = 0; index < elements; index++)
        weights[index] = (int8_t)((index % 31u) - 15);
    for (uint32_t row = 0; row < ROWS; row++)
        scales[row] = 0.015625f * (float)((row % 7u) + 1u);
    CHECK(write_file(weight_path, weights, elements));
    CHECK(write_file(scale_path, scales, (size_t)ROWS * sizeof(float)));
    char error[512] = {0};
    for (size_t index = 0; index < DESTINATIONS; index++) {
        destinations[index] = h3_gpu_tensor_new_bf16(gpu, elements);
        CHECK(destinations[index]);
        CHECK(h3cspeed_gpu_tensor_read_i8_as_bf16(
            destinations[index], weight_path, 0, scale_path, 0,
            ROWS, COLUMNS, error, sizeof(error)));
    }

    /* Nine 16 MiB converted weights exceed the deliberately configured
     * 128 MiB cache, so the oldest destination must be evicted. Rewriting it
     * exercises non-resident -> resident accounting; reading it afterwards
     * proves that the refreshed authoritative RAM backing can be reloaded. */
    CHECK(h3cspeed_gpu_tensor_read_i8_as_bf16(
        destinations[0], weight_path, 0, scale_path, 0,
        ROWS, COLUMNS, error, sizeof(error)));
    CHECK(h3_gpu_tensor_read_bf16(destinations[0], actual, elements));
    const size_t probes[] = {0u, 1u, COLUMNS - 1u,
                             (size_t)COLUMNS * (ROWS - 1u) + 17u};
    for (size_t index = 0; index < sizeof(probes) / sizeof(probes[0]); index++) {
        size_t element = probes[index];
        float expected = (float)weights[element] * scales[element / COLUMNS];
        CHECK(close_f32(bf16_to_f32(actual[element]), expected));
    }

    for (size_t index = 0; index < DESTINATIONS; index++)
        h3_gpu_tensor_free(destinations[index]);
    free(actual); free(scales); free(weights);
    remove(scale_path); remove(weight_path);
    return 0;
}

int main(void) {
    char error[512] = {0};
#if defined(_WIN32)
    CHECK(_putenv_s("H3_CUDA_OFFLOAD", "ram+file") == 0);
    CHECK(_putenv_s("H3_CUDA_LOW_VRAM", "1") == 0);
    CHECK(_putenv_s("H3_CUDA_VRAM_BUDGET_MIB", "512") == 0);
    CHECK(_putenv_s("H3_CUDA_WEIGHT_CACHE_MIB", "128") == 0);
    CHECK(_putenv_s("H3_CUDA_HOST_CACHE_MIB", "256") == 0);
#else
    CHECK(setenv("H3_CUDA_OFFLOAD", "ram+file", 1) == 0);
    CHECK(setenv("H3_CUDA_LOW_VRAM", "1", 1) == 0);
    CHECK(setenv("H3_CUDA_VRAM_BUDGET_MIB", "512", 1) == 0);
    CHECK(setenv("H3_CUDA_WEIGHT_CACHE_MIB", "128", 1) == 0);
    CHECK(setenv("H3_CUDA_HOST_CACHE_MIB", "256", 1) == 0);
#endif
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    if (!gpu) {
        if (no_cuda_device(error)) {
            fprintf(stderr, "CUDA NVFP4 numeric test skipped: %s\n",
                    error[0] ? error : "CUDA device unavailable");
            return 77;
        }
        fprintf(stderr, "CUDA NVFP4 numeric test failed to initialize: %s\n",
                error[0] ? error : "unknown CUDA initialization error");
        return 1;
    }
    int result = test_nvfp4(gpu) || test_f16_and_i8(gpu) ||
                 test_repeated_converted_weight_eviction(gpu);
    h3_gpu_free(gpu);
    if (!result) puts("CUDA NVFP4 numeric test passed");
    return result;
}
