#include "h3_gpu.h"

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

static int close_f32(float actual, float expected, float tolerance) {
    return fabsf(actual - expected) <= tolerance;
}

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

static int set_attention_backend(const char *value) {
#if defined(_WIN32)
    return _putenv_s("H3_CUDA_ATTENTION", value ? value : "") == 0;
#else
    return value ? setenv("H3_CUDA_ATTENTION", value, 1) == 0 :
                   unsetenv("H3_CUDA_ATTENTION") == 0;
#endif
}

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
           strstr(normalized, "cudaerrornodevice") != NULL;
}

static int test_layer_norm_centered(h3_gpu *gpu) {
    const float input_values[] = {
        10000.0f, 10001.0f, 9999.0f, 10000.0f,
        20000.0f, 20001.0f, 19999.0f, 20000.0f,
    };
    const float weight_values[] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float bias_values[] = {0.0f, 0.0f, 0.0f, 0.0f};
    h3_gpu_tensor *input = h3_gpu_tensor_from_f32(gpu, input_values, 8);
    h3_gpu_tensor *weight = h3_gpu_tensor_from_f32(gpu, weight_values, 4);
    h3_gpu_tensor *bias = h3_gpu_tensor_from_f32(gpu, bias_values, 4);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, 8);
    CHECK(input && weight && bias && output);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_layer_norm_f32(gpu, output, input, weight, bias,
                                2, 4, 1e-5f));
    CHECK(h3_gpu_submit(gpu));
    float actual[8] = {0.0f};
    CHECK(h3_gpu_tensor_read_f32(output, actual, 8));
    const float inverse = 1.0f / sqrtf(0.5f + 1e-5f);
    const float expected[] = {
        0.0f, inverse, -inverse, 0.0f,
        0.0f, inverse, -inverse, 0.0f,
    };
    for (size_t index = 0; index < 8; index++)
        CHECK(close_f32(actual[index], expected[index], 1e-3f));
    h3_gpu_tensor_free(output);
    h3_gpu_tensor_free(bias);
    h3_gpu_tensor_free(weight);
    h3_gpu_tensor_free(input);
    return 0;
}

static int test_video_qkv_rope(h3_gpu *gpu) {
    enum {
        SEQUENCE = 2, HEADS = 2, HEAD_DIM = 4, ROPE_HALF = 2,
        QKV_ELEMENTS = SEQUENCE * HEADS * HEAD_DIM * 3,
        OUTPUT_ELEMENTS = SEQUENCE * HEADS * HEAD_DIM
    };
    const size_t qkv_elements = QKV_ELEMENTS;
    const size_t output_elements = OUTPUT_ELEMENTS;
    float qkv[QKV_ELEMENTS];
    float cosine[SEQUENCE * ROPE_HALF] = {0.0f, 1.0f, 1.0f, 0.0f};
    float sine[SEQUENCE * ROPE_HALF] = {1.0f, 0.0f, 0.0f, 1.0f};
    float expected_query[OUTPUT_ELEMENTS];
    float expected_key[OUTPUT_ELEMENTS];
    float expected_value[OUTPUT_ELEMENTS];
    for (int row = 0; row < SEQUENCE; row++) {
        for (int head = 0; head < HEADS; head++) {
            size_t base = ((size_t)row * HEADS + (size_t)head) * HEAD_DIM * 3;
            float q[] = {
                3.0f + row + head, 4.0f + row + head, 0.0f, 0.0f
            };
            float k[] = {
                6.0f + 2.0f * row + head,
                8.0f + 2.0f * row + head, 0.0f, 0.0f
            };
            for (int dimension = 0; dimension < HEAD_DIM; dimension++) {
                qkv[base + (size_t)dimension] = q[dimension];
                qkv[base + HEAD_DIM + (size_t)dimension] = k[dimension];
                qkv[base + HEAD_DIM * 2 + (size_t)dimension] =
                    100.0f + 10.0f * row + 4.0f * head + dimension;
            }
            float q_inverse = 1.0f / sqrtf(
                (q[0] * q[0] + q[1] * q[1]) / HEAD_DIM + 1e-5f);
            float k_inverse = 1.0f / sqrtf(
                (k[0] * k[0] + k[1] * k[1]) / HEAD_DIM + 1e-5f);
            size_t destination = ((size_t)head * SEQUENCE + (size_t)row) *
                                 HEAD_DIM;
            for (int dimension = 0; dimension < HEAD_DIM; dimension++) {
                float q_value = q[dimension] * q_inverse;
                float k_value = k[dimension] * k_inverse;
                if (dimension < ROPE_HALF) {
                    int pair = dimension + ROPE_HALF;
                    float c = cosine[row * ROPE_HALF + dimension];
                    float s = sine[row * ROPE_HALF + dimension];
                    expected_query[destination + (size_t)dimension] =
                        q_value * c - q[pair] * q_inverse * s;
                    expected_key[destination + (size_t)dimension] =
                        k_value * c - k[pair] * k_inverse * s;
                } else if (dimension < ROPE_HALF * 2) {
                    int pair = dimension - ROPE_HALF;
                    float c = cosine[row * ROPE_HALF + pair];
                    float s = sine[row * ROPE_HALF + pair];
                    expected_query[destination + (size_t)dimension] =
                        q_value * c + q[pair] * q_inverse * s;
                    expected_key[destination + (size_t)dimension] =
                        k_value * c + k[pair] * k_inverse * s;
                } else {
                    expected_query[destination + (size_t)dimension] = q_value;
                    expected_key[destination + (size_t)dimension] = k_value;
                }
                expected_value[destination + (size_t)dimension] =
                    100.0f + 10.0f * row + 4.0f * head + dimension;
            }
        }
    }
    h3_gpu_tensor *input = h3_gpu_tensor_from_f32(gpu, qkv, qkv_elements);
    h3_gpu_tensor *cosine_tensor = h3_gpu_tensor_from_f32(
        gpu, cosine, SEQUENCE * ROPE_HALF);
    h3_gpu_tensor *sine_tensor = h3_gpu_tensor_from_f32(
        gpu, sine, SEQUENCE * ROPE_HALF);
    h3_gpu_tensor *query = h3_gpu_tensor_new_f32(gpu, output_elements);
    h3_gpu_tensor *key = h3_gpu_tensor_new_f32(gpu, output_elements);
    h3_gpu_tensor *value = h3_gpu_tensor_new_f32(gpu, output_elements);
    CHECK(input && cosine_tensor && sine_tensor && query && key && value);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_video_qkv_rope_f32(
        gpu, query, key, value, input, cosine_tensor, sine_tensor,
        SEQUENCE, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f));
    CHECK(h3_gpu_submit(gpu));
    float actual_query[OUTPUT_ELEMENTS];
    float actual_key[OUTPUT_ELEMENTS];
    float actual_value[OUTPUT_ELEMENTS];
    CHECK(h3_gpu_tensor_read_f32(query, actual_query, output_elements));
    CHECK(h3_gpu_tensor_read_f32(key, actual_key, output_elements));
    CHECK(h3_gpu_tensor_read_f32(value, actual_value, output_elements));
    for (size_t index = 0; index < output_elements; index++) {
        CHECK(close_f32(actual_query[index], expected_query[index], 1e-4f));
        CHECK(close_f32(actual_key[index], expected_key[index], 1e-4f));
        CHECK(close_f32(actual_value[index], expected_value[index], 1e-4f));
    }
    h3_gpu_tensor_free(value);
    h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(query);
    h3_gpu_tensor_free(sine_tensor);
    h3_gpu_tensor_free(cosine_tensor);
    h3_gpu_tensor_free(input);
    return 0;
}

static int test_fused_adaln_inverse(h3_gpu *gpu) {
    enum { ROWS = 2, WIDTH = 4, OUTPUT_DIM = 2, SLOTS = 2 };
    const uint16_t input_values[ROWS * WIDTH] = {
        0x4040, 0x4080, 0x0000, 0x0000,
        0x3f80, 0x0000, 0x0000, 0x0000,
    };
    const uint16_t norm_weight_values[WIDTH] = {
        0x3f80, 0x3f80, 0x3f80, 0x3f80,
    };
    /* Two modulation rows, each with shift slot 0 and zero scale slot 1. */
    const uint16_t modulation_values[ROWS * SLOTS * WIDTH] = {
        0x3f00, 0xbf80, 0x3f00, 0x0000,
        0x0000, 0x0000, 0x0000, 0x0000,
        0xbe80, 0x3f00, 0xbf00, 0x0000,
        0x0000, 0x0000, 0x0000, 0x0000,
    };
    const uint32_t row_map_values[ROWS] = {0, 1};
    const uint16_t weight_values[OUTPUT_DIM * WIDTH] = {
        0x3f80, 0x3f00, 0xbf80, 0x3e80,
        0x3e80, 0xbf80, 0x3f00, 0x3f80,
    };
    const uint16_t bias_values[OUTPUT_DIM] = {0x3f40, 0xbf00};
    h3_gpu_tensor *input = h3_gpu_tensor_from_bf16(
        gpu, input_values, ROWS * WIDTH);
    h3_gpu_tensor *norm_weight = h3_gpu_tensor_from_bf16(
        gpu, norm_weight_values, WIDTH);
    h3_gpu_tensor *modulation = h3_gpu_tensor_from_bf16(
        gpu, modulation_values, ROWS * SLOTS * WIDTH);
    h3_gpu_tensor *row_map = h3_gpu_tensor_from_u32(gpu, row_map_values, ROWS);
    h3_gpu_tensor *weight = h3_gpu_tensor_from_bf16(
        gpu, weight_values, OUTPUT_DIM * WIDTH);
    h3_gpu_tensor *bias = h3_gpu_tensor_from_bf16(gpu, bias_values, OUTPUT_DIM);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, ROWS * OUTPUT_DIM);
    h3_gpu_tensor *inverse = h3_gpu_tensor_new_f32(gpu, ROWS);
    CHECK(input && norm_weight && modulation && row_map && weight &&
          bias && output && inverse);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_adaln_linear_bf16(
        gpu, output, inverse, input, 0, norm_weight, modulation, row_map,
        weight, bias, ROWS, WIDTH, OUTPUT_DIM, SLOTS, 0, 1, 1e-5f));
    CHECK(h3_gpu_submit(gpu));
    float actual_inverse[ROWS] = {0.0f};
    uint16_t actual_output[ROWS * OUTPUT_DIM] = {0};
    CHECK(h3_gpu_tensor_read_f32(inverse, actual_inverse, ROWS));
    CHECK(h3_gpu_tensor_read_bf16(output, actual_output, ROWS * OUTPUT_DIM));
    CHECK(close_f32(actual_inverse[0], 1.0f / sqrtf(6.25f + 1e-5f), 1e-5f));
    CHECK(close_f32(actual_inverse[1], 1.0f / sqrtf(0.25f + 1e-5f), 1e-5f));
    const float inverse0 = 1.0f / sqrtf(6.25f + 1e-5f);
    const float inverse1 = 1.0f / sqrtf(0.25f + 1e-5f);
    const float normalized[] = {
        3.0f * inverse0 + 0.5f, 4.0f * inverse0 - 1.0f, 0.5f, 0.0f,
        inverse1 - 0.25f, 0.5f, -0.5f, 0.0f,
    };
    const float expected[] = {
        normalized[0] + 0.5f * normalized[1] - normalized[2] + 0.25f * normalized[3] + 0.75f,
        0.25f * normalized[0] - normalized[1] + 0.5f * normalized[2] + normalized[3] - 0.5f,
        normalized[4] + 0.5f * normalized[5] - normalized[6] + 0.25f * normalized[7] + 0.75f,
        0.25f * normalized[4] - normalized[5] + 0.5f * normalized[6] + normalized[7] - 0.5f,
    };
    for (size_t index = 0; index < ROWS * OUTPUT_DIM; index++) {
        CHECK(close_f32(bf16_to_f32(actual_output[index]), expected[index], 0.03f));
    }
    h3_gpu_tensor_free(inverse);
    h3_gpu_tensor_free(output);
    h3_gpu_tensor_free(bias);
    h3_gpu_tensor_free(weight);
    h3_gpu_tensor_free(row_map);
    h3_gpu_tensor_free(modulation);
    h3_gpu_tensor_free(norm_weight);
    h3_gpu_tensor_free(input);
    return 0;
}

static int test_text_rope_row_major(h3_gpu *gpu) {
    enum { SEQUENCE = 3, QUERY_HEADS = 2, KV_HEADS = 1, HEAD_DIM = 4,
           ROPE_HALF = 2 };
    const float query_values[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        11.0f, 12.0f, 13.0f, 14.0f,
        21.0f, 22.0f, 23.0f, 24.0f,
        31.0f, 32.0f, 33.0f, 34.0f,
        41.0f, 42.0f, 43.0f, 44.0f,
        51.0f, 52.0f, 53.0f, 54.0f,
    };
    const float key_values[] = {
        -1.0f, -2.0f, -3.0f, -4.0f,
        -11.0f, -12.0f, -13.0f, -14.0f,
        -21.0f, -22.0f, -23.0f, -24.0f,
    };
    const float cosine[] = {1.0f, 0.0f, 0.5f, -0.5f, -0.25f, 0.75f};
    const float sine[] = {0.0f, 1.0f, 0.25f, 0.75f, 0.5f, -0.125f};
    uint16_t query_bits[sizeof(query_values) / sizeof(query_values[0])];
    uint16_t key_bits[sizeof(key_values) / sizeof(key_values[0])];
    for (size_t i = 0; i < sizeof(query_bits) / sizeof(query_bits[0]); i++)
        query_bits[i] = f32_to_bf16(query_values[i]);
    for (size_t i = 0; i < sizeof(key_bits) / sizeof(key_bits[0]); i++)
        key_bits[i] = f32_to_bf16(key_values[i]);
    h3_gpu_tensor *query = h3_gpu_tensor_from_bf16(
        gpu, query_bits, sizeof(query_bits) / sizeof(query_bits[0]));
    h3_gpu_tensor *key = h3_gpu_tensor_from_bf16(
        gpu, key_bits, sizeof(key_bits) / sizeof(key_bits[0]));
    h3_gpu_tensor *cosine_tensor = h3_gpu_tensor_from_f32(
        gpu, cosine, sizeof(cosine) / sizeof(cosine[0]));
    h3_gpu_tensor *sine_tensor = h3_gpu_tensor_from_f32(
        gpu, sine, sizeof(sine) / sizeof(sine[0]));
    CHECK(query && key && cosine_tensor && sine_tensor);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_rope_text_bf16(
        gpu, query, key, cosine_tensor, sine_tensor, SEQUENCE, QUERY_HEADS,
        KV_HEADS, HEAD_DIM));
    CHECK(h3_gpu_submit(gpu));
    uint16_t actual_query[sizeof(query_bits) / sizeof(query_bits[0])] = {0};
    uint16_t actual_key[sizeof(key_bits) / sizeof(key_bits[0])] = {0};
    CHECK(h3_gpu_tensor_read_bf16(
        query, actual_query, sizeof(actual_query) / sizeof(actual_query[0])));
    CHECK(h3_gpu_tensor_read_bf16(
        key, actual_key, sizeof(actual_key) / sizeof(actual_key[0])));
    for (int row = 0; row < SEQUENCE; row++) {
        const float *c = cosine + row * ROPE_HALF;
        const float *s = sine + row * ROPE_HALF;
        for (int head = 0; head < QUERY_HEADS; head++) {
            size_t base = (size_t)(row * QUERY_HEADS + head) * HEAD_DIM;
            for (int d = 0; d < ROPE_HALF; d++) {
                int pair = d + ROPE_HALF;
                float first = bf16_to_f32(query_bits[base + d]);
                float second = bf16_to_f32(query_bits[base + pair]);
                float expected_first = first * c[d] - second * s[d];
                float expected_second = second * c[d] + first * s[d];
                CHECK(close_f32(bf16_to_f32(actual_query[base + d]),
                                expected_first, 0.02f));
                CHECK(close_f32(bf16_to_f32(actual_query[base + pair]),
                                expected_second, 0.02f));
            }
        }
        size_t base = (size_t)row * KV_HEADS * HEAD_DIM;
        for (int d = 0; d < ROPE_HALF; d++) {
            int pair = d + ROPE_HALF;
            float first = bf16_to_f32(key_bits[base + d]);
            float second = bf16_to_f32(key_bits[base + pair]);
            float expected_first = first * c[d] - second * s[d];
            float expected_second = second * c[d] + first * s[d];
            CHECK(close_f32(bf16_to_f32(actual_key[base + d]),
                            expected_first, 0.02f));
            CHECK(close_f32(bf16_to_f32(actual_key[base + pair]),
                            expected_second, 0.02f));
        }
    }
    h3_gpu_tensor_free(sine_tensor);
    h3_gpu_tensor_free(cosine_tensor);
    h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(query);
    return 0;
}

static int test_text_gqa_row_major(h3_gpu *gpu) {
    enum { SEQUENCE = 2, QUERY_HEADS = 2, KV_HEADS = 1, HEAD_DIM = 4 };
    const float query_values[] = {
        1.5f, -0.75f, 2.25f, -1.125f,
        3.25f, 0.5f, -2.75f, 1.75f,
        -2.5f, 1.25f, 0.875f, 3.5f,
        1.75f, -3.0f, 2.5f, -0.625f,
    };
    const float key_values[] = {
        0.5f, -1.25f, 2.0f, 0.75f,
        -1.5f, 0.25f, 1.25f, 2.5f,
    };
    const float value_values[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 7.0f, 11.0f, 13.0f,
    };
    uint16_t query_bits[sizeof(query_values) / sizeof(query_values[0])];
    uint16_t key_bits[sizeof(key_values) / sizeof(key_values[0])];
    uint16_t value_bits[sizeof(value_values) / sizeof(value_values[0])];
    for (size_t i = 0; i < sizeof(query_bits) / sizeof(query_bits[0]); i++)
        query_bits[i] = f32_to_bf16(query_values[i]);
    for (size_t i = 0; i < sizeof(key_bits) / sizeof(key_bits[0]); i++)
        key_bits[i] = f32_to_bf16(key_values[i]);
    for (size_t i = 0; i < sizeof(value_bits) / sizeof(value_bits[0]); i++)
        value_bits[i] = f32_to_bf16(value_values[i]);
    h3_gpu_tensor *query = h3_gpu_tensor_from_bf16(
        gpu, query_bits, sizeof(query_bits) / sizeof(query_bits[0]));
    h3_gpu_tensor *key = h3_gpu_tensor_from_bf16(
        gpu, key_bits, sizeof(key_bits) / sizeof(key_bits[0]));
    h3_gpu_tensor *value = h3_gpu_tensor_from_bf16(
        gpu, value_bits, sizeof(value_bits) / sizeof(value_bits[0]));
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(
        gpu, SEQUENCE * QUERY_HEADS * HEAD_DIM);
    CHECK(query && key && value && output);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_gqa_causal_bf16(
        gpu, output, query, key, value, SEQUENCE, QUERY_HEADS, KV_HEADS,
        HEAD_DIM, 0.5f));
    CHECK(h3_gpu_submit(gpu));
    uint16_t actual[SEQUENCE * QUERY_HEADS * HEAD_DIM] = {0};
    CHECK(h3_gpu_tensor_read_bf16(
        output, actual, sizeof(actual) / sizeof(actual[0])));

    /* Metal's text path stores row-major Q/K/V and rounds Q*scale to BF16
     * before the dot product.  Keep this reference deliberately independent
     * of the CUDA layout flags so a head/token transpose cannot pass. */
    float expected[SEQUENCE * QUERY_HEADS * HEAD_DIM] = {0.0f};
    for (int row = 0; row < SEQUENCE; row++) {
        for (int head = 0; head < QUERY_HEADS; head++) {
            float scores[SEQUENCE];
            float maximum = -INFINITY;
            int count = row + 1;
            for (int key_row = 0; key_row < count; key_row++) {
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) {
                    float q = bf16_to_f32(query_bits[
                        ((row * QUERY_HEADS + head) * HEAD_DIM) + d]);
                    q = bf16_to_f32(f32_to_bf16(q * 0.5f));
                    float k = bf16_to_f32(key_bits[key_row * HEAD_DIM + d]);
                    dot += q * k;
                }
                scores[key_row] = dot;
                if (dot > maximum) maximum = dot;
            }
            float denominator = 0.0f;
            for (int key_row = 0; key_row < count; key_row++) {
                scores[key_row] = expf(scores[key_row] - maximum);
                denominator += scores[key_row];
            }
            for (int d = 0; d < HEAD_DIM; d++) {
                float sum = 0.0f;
                for (int key_row = 0; key_row < count; key_row++)
                    sum += scores[key_row] / denominator *
                           bf16_to_f32(value_bits[key_row * HEAD_DIM + d]);
                expected[(row * QUERY_HEADS + head) * HEAD_DIM + d] = sum;
            }
        }
    }
    for (size_t i = 0; i < sizeof(actual) / sizeof(actual[0]); i++)
        CHECK(close_f32(bf16_to_f32(actual[i]), expected[i], 0.06f));
    h3_gpu_tensor_free(output);
    h3_gpu_tensor_free(value);
    h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(query);
    return 0;
}

static int test_sage_video_attention(h3_gpu *gpu) {
    enum { SEQUENCE = 3, HEADS = 1, HEAD_DIM = 8, ELEMENTS = SEQUENCE * HEAD_DIM };
    uint16_t query_bits[ELEMENTS], key_bits[ELEMENTS], value_bits[ELEMENTS];
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    for (int row = 0; row < SEQUENCE; row++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            size_t index = (size_t)row * HEAD_DIM + d;
            query_bits[index] = f32_to_bf16(
                sinf((float)(index + 1) * 0.37f) * 1.5f);
            key_bits[index] = f32_to_bf16(
                cosf((float)(index + 3) * 0.23f) * 1.25f);
            value_bits[index] = f32_to_bf16(
                ((float)(row + 1) * 0.5f) + (float)d * 0.125f);
        }
    }
    h3_gpu_tensor *query = h3_gpu_tensor_from_bf16(gpu, query_bits, ELEMENTS);
    h3_gpu_tensor *key = h3_gpu_tensor_from_bf16(gpu, key_bits, ELEMENTS);
    h3_gpu_tensor *value = h3_gpu_tensor_from_bf16(gpu, value_bits, ELEMENTS);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, ELEMENTS);
    CHECK(query && key && value && output);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_sdpa_bf16_head_major_output(
        gpu, output, query, key, value, SEQUENCE, HEADS, HEAD_DIM, scale));
    CHECK(h3_gpu_submit(gpu));
    uint16_t actual[ELEMENTS];
    CHECK(h3_gpu_tensor_read_bf16(output, actual, ELEMENTS));
    for (int row = 0; row < SEQUENCE; row++) {
        float scores[SEQUENCE], maximum = -INFINITY, denominator = 0.0f;
        for (int key_row = 0; key_row < SEQUENCE; key_row++) {
            float dot = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++)
                dot += bf16_to_f32(query_bits[row * HEAD_DIM + d]) *
                       bf16_to_f32(key_bits[key_row * HEAD_DIM + d]);
            scores[key_row] = dot * scale;
            if (scores[key_row] > maximum) maximum = scores[key_row];
        }
        for (int key_row = 0; key_row < SEQUENCE; key_row++) {
            scores[key_row] = expf(scores[key_row] - maximum);
            denominator += scores[key_row];
        }
        for (int d = 0; d < HEAD_DIM; d++) {
            float expected = 0.0f;
            for (int key_row = 0; key_row < SEQUENCE; key_row++)
                expected += scores[key_row] / denominator *
                    bf16_to_f32(value_bits[key_row * HEAD_DIM + d]);
            CHECK(close_f32(
                bf16_to_f32(actual[row * HEAD_DIM + d]), expected, 0.04f));
        }
    }
    h3_gpu_tensor_free(output);
    h3_gpu_tensor_free(value);
    h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(query);
    return 0;
}

static int test_sage_f32_native_fallback(h3_gpu *gpu) {
    enum { SEQUENCE = 2, HEADS = 1, HEAD_DIM = 4, ELEMENTS = 8 };
    const float query_values[ELEMENTS] = {
        1.0f, -0.5f, 0.25f, 2.0f, -1.0f, 0.75f, 1.5f, -0.25f};
    const float key_values[ELEMENTS] = {
        0.5f, 1.0f, -0.5f, 0.25f, 1.25f, -0.75f, 0.5f, 1.0f};
    const float value_values[ELEMENTS] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    h3_gpu_tensor *query = h3_gpu_tensor_from_f32(gpu, query_values, ELEMENTS);
    h3_gpu_tensor *key = h3_gpu_tensor_from_f32(gpu, key_values, ELEMENTS);
    h3_gpu_tensor *value = h3_gpu_tensor_from_f32(gpu, value_values, ELEMENTS);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, ELEMENTS);
    CHECK(query && key && value && output);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_sdpa_f32(
        gpu, output, query, key, value, SEQUENCE, HEADS, HEAD_DIM, 0.5f));
    CHECK(h3_gpu_submit(gpu));
    float actual[ELEMENTS];
    CHECK(h3_gpu_tensor_read_f32(output, actual, ELEMENTS));
    for (size_t index = 0; index < ELEMENTS; index++) CHECK(isfinite(actual[index]));
    h3_gpu_tensor_free(output);
    h3_gpu_tensor_free(value);
    h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(query);
    return 0;
}

int main(void) {
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    if (!gpu) {
        if (is_no_cuda_device_error(error)) {
            fprintf(stderr, "CUDA numeric regression skipped: %s\n",
                    error[0] ? error : "CUDA device unavailable");
            return 77;
        }
        fprintf(stderr, "CUDA numeric regression failed to initialize: %s\n",
                error[0] ? error : "unknown CUDA initialization error");
        return 1;
    }
    int result = test_layer_norm_centered(gpu) ||
                 test_video_qkv_rope(gpu) ||
                 test_fused_adaln_inverse(gpu) ||
                 test_text_rope_row_major(gpu) ||
                 test_text_gqa_row_major(gpu);
    if (!result && (!set_attention_backend("sage") ||
                    test_text_gqa_row_major(gpu) ||
                    test_sage_video_attention(gpu) ||
                    test_sage_f32_native_fallback(gpu))) result = 1;
    if (!set_attention_backend(NULL)) result = 1;
    h3_gpu_free(gpu);
    if (!result) puts("CUDA numeric regression passed");
    return result;
}
