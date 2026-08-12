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

static int is_no_cuda_device_error(const char *error) {
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
           strstr(normalized, "cudaerrornodevice") != NULL ||
           strstr(normalized, "cuda driver version is insufficient") != NULL ||
           strstr(normalized, "cuda_error_insufficient_driver") != NULL ||
           strstr(normalized, "cudaerrorinsufficientdriver") != NULL;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float bf16_to_f32(uint16_t bits) {
    uint32_t value = (uint32_t)bits << 16;
    float result;
    memcpy(&result, &value, sizeof(result));
    return result;
}

static int close_f32(float actual, float expected) {
    float tolerance = 0.02f * fmaxf(1.0f, fabsf(expected));
    return fabsf(actual - expected) <= tolerance;
}

/* CPU oracle for comfy-kitchen's regular H4 Kronecker expansion. */
static void rotate_group_cpu(const float *input, float *output,
                             uint32_t group_size) {
    float current[256] = {0.0f};
    float next[256] = {0.0f};
    for (uint32_t index = 0; index < group_size; index++)
        current[index] = input[index];
    for (uint32_t stride = 1; stride < group_size; stride *= 4) {
        for (uint32_t block = 0; block < group_size;
             block += 4u * stride) {
            for (uint32_t offset = 0; offset < stride; offset++) {
                uint32_t base = block + offset;
                float a = current[base];
                float b = current[base + stride];
                float c = current[base + 2u * stride];
                float d = current[base + 3u * stride];
                next[base] = (a + b + c - d) * 0.5f;
                next[base + stride] = (a + b - c + d) * 0.5f;
                next[base + 2u * stride] = (a - b + c + d) * 0.5f;
                next[base + 3u * stride] = (-a + b + c + d) * 0.5f;
            }
        }
        memcpy(current, next, group_size * sizeof(float));
    }
    memcpy(output, current, group_size * sizeof(float));
}

static void cpu_convrot_quantize(const float *input, int8_t *quantized,
                                 float *row_scales, uint32_t rows,
                                 uint32_t width, uint32_t group_size) {
    float rotated[256];
    float maximum;
    for (uint32_t row = 0; row < rows; row++) {
        maximum = 0.0f;
        for (uint32_t group = 0; group < width / group_size; group++) {
            rotate_group_cpu(input + (size_t)row * width + group * group_size,
                             rotated, group_size);
            for (uint32_t lane = 0; lane < group_size; lane++)
                maximum = fmaxf(maximum, fabsf(rotated[lane]));
        }
        float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
        row_scales[row] = scale;
        float inverse = 1.0f / scale;
        for (uint32_t group = 0; group < width / group_size; group++) {
            rotate_group_cpu(input + (size_t)row * width + group * group_size,
                             rotated, group_size);
            for (uint32_t lane = 0; lane < group_size; lane++) {
                float value = nearbyintf(rotated[lane] * inverse);
                value = fminf(127.0f, fmaxf(-127.0f, value));
                quantized[(size_t)row * width + group * group_size + lane] =
                    (int8_t)value;
            }
        }
    }
}

static int run_case(h3_gpu *gpu, const char *path, uint32_t rows,
                    uint32_t width, uint32_t output_dim) {
    size_t weight_elements = (size_t)output_dim * width;
    int8_t *weight_values = (int8_t *)malloc(weight_elements);
    float *weight_scales = (float *)malloc(output_dim * sizeof(float));
    float *input_values = (float *)malloc((size_t)rows * width * sizeof(float));
    int8_t *expected_q = (int8_t *)malloc((size_t)rows * width);
    float *expected_input_scales = (float *)malloc(rows * sizeof(float));
    uint16_t *actual = (uint16_t *)calloc((size_t)rows * output_dim,
                                           sizeof(uint16_t));
    CHECK(weight_values && weight_scales && input_values && expected_q &&
          expected_input_scales && actual);

    for (uint32_t output = 0; output < output_dim; output++) {
        weight_scales[output] = 0.0078125f + 0.001f * (float)output;
        for (uint32_t column = 0; column < width; column++) {
            int value = (int)((output * 17u + column * 13u) % 31u) - 15;
            weight_values[(size_t)output * width + column] = (int8_t)value;
        }
    }
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t column = 0; column < width; column++) {
            float centered = (float)((int)((column * 7u + row * 11u) % 37u) - 18);
            input_values[(size_t)row * width + column] =
                centered * 0.03125f + (float)(row + 1u) * 0.0078125f;
        }
    }
    cpu_convrot_quantize(input_values, expected_q, expected_input_scales,
                         rows, width, 256);

    const char *weight_path = path;
    FILE *file = fopen(weight_path, "wb");
    CHECK(file != NULL);
    CHECK(fwrite(weight_values, 1, weight_elements, file) == weight_elements);
    CHECK(fclose(file) == 0);

    h3_gpu_tensor *weight = h3cspeed_gpu_tensor_load_i8_convrot(
        gpu, weight_path, 0, weight_elements, 256);
    h3_gpu_tensor *weight_scale = h3_gpu_tensor_from_f32(
        gpu, weight_scales, output_dim);
    h3_gpu_tensor *input = h3_gpu_tensor_from_f32(
        gpu, input_values, (size_t)rows * width);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * output_dim);
    h3_gpu_tensor *quantized_input = h3_gpu_tensor_new_i8(
        gpu, (size_t)rows * width);
    h3_gpu_tensor *input_scales = h3_gpu_tensor_new_f32(gpu, rows);
    CHECK(weight && weight_scale && input && output && quantized_input &&
          input_scales);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_linear_int8_bf16(
        gpu, output, quantized_input, input_scales, input, weight,
        weight_scale, rows, width, output_dim, 0));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_bf16(output, actual,
                                  (size_t)rows * output_dim));

    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t out = 0; out < output_dim; out++) {
            int32_t accumulator = 0;
            for (uint32_t column = 0; column < width; column++) {
                accumulator += (int32_t)expected_q[(size_t)row * width + column] *
                    (int32_t)weight_values[(size_t)out * width + column];
            }
            float expected = (float)accumulator * expected_input_scales[row] *
                weight_scales[out];
            float got = bf16_to_f32(actual[(size_t)row * output_dim + out]);
            CHECK(close_f32(got, expected));
        }
    }

    h3_gpu_tensor_free(input_scales);
    h3_gpu_tensor_free(quantized_input);
    h3_gpu_tensor_free(output);
    h3_gpu_tensor_free(input);
    h3_gpu_tensor_free(weight_scale);
    h3_gpu_tensor_free(weight);
    free(actual);
    free(expected_input_scales);
    free(expected_q);
    free(input_values);
    free(weight_scales);
    free(weight_values);
    (void)remove(weight_path);
    return 0;
}

/* The Comfy H3 ConvRot checkpoint stores each fused projection as contiguous
 * [Q-all, K-all, V-all] rows.  Keep this input deliberately non-symmetric so
 * accidentally routing it through the legacy [head,Q,K,V] grouped decoder is
 * observable in every output stream. */
static int test_contiguous_qkv_rope(h3_gpu *gpu) {
    enum {
        SEQUENCE = 2,
        HEADS = 2,
        HEAD_DIM = 4,
        QKV_ELEMENTS = SEQUENCE * HEADS * HEAD_DIM * 3,
        OUTPUT_ELEMENTS = SEQUENCE * HEADS * HEAD_DIM,
    };
    uint16_t qkv[QKV_ELEMENTS];
    uint16_t q_norm_values[HEAD_DIM];
    uint16_t k_norm_values[HEAD_DIM];
    for (size_t index = 0; index < HEAD_DIM; index++) {
        q_norm_values[index] = f32_to_bf16(1.0f);
        k_norm_values[index] = f32_to_bf16(1.0f);
    }
    float expected_query[OUTPUT_ELEMENTS];
    float expected_key[OUTPUT_ELEMENTS];
    float expected_value[OUTPUT_ELEMENTS];
    memset(qkv, 0, sizeof(qkv));
    for (uint32_t row = 0; row < SEQUENCE; row++) {
        size_t row_base = (size_t)row * HEADS * HEAD_DIM * 3;
        for (uint32_t head = 0; head < HEADS; head++) {
            float q_values[HEAD_DIM];
            float k_values[HEAD_DIM];
            for (uint32_t dimension = 0; dimension < HEAD_DIM; dimension++) {
                q_values[dimension] = 1.0f + (float)(row * 100u + head * 10u + dimension);
                k_values[dimension] = 100.0f +
                    (float)(row * 100u + head * 10u + dimension);
                size_t q_index = row_base + (size_t)head * HEAD_DIM + dimension;
                size_t k_index = row_base + (size_t)HEADS * HEAD_DIM +
                    (size_t)head * HEAD_DIM + dimension;
                size_t v_index = row_base + (size_t)HEADS * 2u * HEAD_DIM +
                    (size_t)head * HEAD_DIM + dimension;
                qkv[q_index] = f32_to_bf16(q_values[dimension]);
                qkv[k_index] = f32_to_bf16(k_values[dimension]);
                qkv[v_index] = f32_to_bf16(
                    1000.0f + (float)(row * 100u + head * 10u + dimension));
            }
            float q_square = 0.0f;
            float k_square = 0.0f;
            for (uint32_t dimension = 0; dimension < HEAD_DIM; dimension++) {
                q_square += q_values[dimension] * q_values[dimension];
                k_square += k_values[dimension] * k_values[dimension];
            }
            float q_inverse = 1.0f / sqrtf(q_square / (float)HEAD_DIM + 1e-5f);
            float k_inverse = 1.0f / sqrtf(k_square / (float)HEAD_DIM + 1e-5f);
            size_t destination = ((size_t)head * SEQUENCE + row) * HEAD_DIM;
            for (uint32_t dimension = 0; dimension < HEAD_DIM; dimension++) {
                expected_query[destination + dimension] =
                    q_values[dimension] * q_inverse;
                expected_key[destination + dimension] =
                    k_values[dimension] * k_inverse;
                expected_value[destination + dimension] =
                    1000.0f + (float)(row * 100u + head * 10u + dimension);
            }
        }
    }
    float rope_cos[1] = {1.0f};
    float rope_sin[1] = {0.0f};
    h3_gpu_tensor *input = h3_gpu_tensor_from_bf16(
        gpu, qkv, QKV_ELEMENTS);
    h3_gpu_tensor *q_norm = h3_gpu_tensor_from_bf16(
        gpu, q_norm_values, HEAD_DIM);
    h3_gpu_tensor *k_norm = h3_gpu_tensor_from_bf16(
        gpu, k_norm_values, HEAD_DIM);
    h3_gpu_tensor *cosine = h3_gpu_tensor_from_f32(gpu, rope_cos, 1);
    h3_gpu_tensor *sine = h3_gpu_tensor_from_f32(gpu, rope_sin, 1);
    h3_gpu_tensor *query = h3_gpu_tensor_new_bf16(gpu, OUTPUT_ELEMENTS);
    h3_gpu_tensor *key = h3_gpu_tensor_new_bf16(gpu, OUTPUT_ELEMENTS);
    h3_gpu_tensor *value = h3_gpu_tensor_new_bf16(gpu, OUTPUT_ELEMENTS);
    CHECK(input && q_norm && k_norm && cosine && sine && query && key && value);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_qkv_rope_bf16(
        gpu, query, key, value, input, q_norm, k_norm, cosine, sine,
        SEQUENCE, HEADS, HEAD_DIM, 0, 1e-5f));
    CHECK(h3_gpu_submit(gpu));
    uint16_t actual_query_bits[OUTPUT_ELEMENTS];
    uint16_t actual_key_bits[OUTPUT_ELEMENTS];
    uint16_t actual_value_bits[OUTPUT_ELEMENTS];
    CHECK(h3_gpu_tensor_read_bf16(query, actual_query_bits, OUTPUT_ELEMENTS));
    CHECK(h3_gpu_tensor_read_bf16(key, actual_key_bits, OUTPUT_ELEMENTS));
    CHECK(h3_gpu_tensor_read_bf16(value, actual_value_bits, OUTPUT_ELEMENTS));
    for (size_t index = 0; index < OUTPUT_ELEMENTS; index++) {
        CHECK(close_f32(bf16_to_f32(actual_query_bits[index]),
                        expected_query[index]));
        CHECK(close_f32(bf16_to_f32(actual_key_bits[index]),
                        expected_key[index]));
        CHECK(close_f32(bf16_to_f32(actual_value_bits[index]),
                        expected_value[index]));
    }
    h3_gpu_tensor_free(value);
    h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(query);
    h3_gpu_tensor_free(sine);
    h3_gpu_tensor_free(cosine);
    h3_gpu_tensor_free(k_norm);
    h3_gpu_tensor_free(q_norm);
    h3_gpu_tensor_free(input);
    return 0;
}

int main(void) {
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    if (!gpu) {
        if (is_no_cuda_device_error(error)) return 77;
        fprintf(stderr, "CUDA setup failed: %s\n", error);
        return 1;
    }
    const char *path = "convrot_numeric_weights.bin";
    /* K=256 is the first CPU oracle; the following cases exercise multiple
     * ConvRot groups while keeping each group at the Comfy default size. */
    int result = test_contiguous_qkv_rope(gpu) ||
                 run_case(gpu, path, 2, 256, 3) ||
                 run_case(gpu, path, 3, 512, 3) ||
                 run_case(gpu, path, 4, 768, 3);
    h3_gpu_free(gpu);
    (void)remove(path);
    return result;
}
