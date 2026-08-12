#include "h3_gpu.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    BENCH_HEADS = 56,
    BENCH_SEQUENCE = 800,
    BENCH_HEAD_DIM = 128,
    BENCH_WARMUP = 2,
    BENCH_ITERATIONS = 10,
};

typedef struct attention_metrics {
    double mae;
    double max_abs;
    double cosine;
} attention_metrics;

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

static uint32_t next_random(uint32_t *state) {
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return *state;
}

static float random_symmetric(uint32_t *state) {
    uint32_t sample = next_random(state) >> 8;
    return ((float)(sample & UINT32_C(0xffff)) / 32767.5f) - 1.0f;
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

static double wall_seconds(void) {
    struct timespec timestamp;
    if (timespec_get(&timestamp, TIME_UTC) != TIME_UTC) return 0.0;
    return (double)timestamp.tv_sec +
           (double)timestamp.tv_nsec * 1.0e-9;
}

static int submit_attention(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *query,
                            const h3_gpu_tensor *key,
                            const h3_gpu_tensor *value, float scale) {
    if (!h3_gpu_begin(gpu) ||
        !h3_gpu_sdpa_bf16(gpu, output, query, key, value,
                          BENCH_SEQUENCE, BENCH_HEADS, BENCH_HEAD_DIM,
                          scale) ||
        !h3_gpu_submit(gpu)) {
        fprintf(stderr, "attention submission failed: %s\n",
                h3_gpu_error(gpu) ? h3_gpu_error(gpu) : "unknown CUDA error");
        return 0;
    }
    return 1;
}

static int run_backend(h3_gpu *gpu, const char *backend,
                       h3_gpu_tensor *output,
                       const h3_gpu_tensor *query,
                       const h3_gpu_tensor *key,
                       const h3_gpu_tensor *value,
                       float scale, uint16_t *host_output,
                       size_t elements, double *mean_ms) {
    if (!set_attention_backend(backend)) {
        fprintf(stderr, "cannot select H3_CUDA_ATTENTION=%s\n", backend);
        return 0;
    }
    for (int iteration = 0; iteration < BENCH_WARMUP; iteration++) {
        if (!submit_attention(gpu, output, query, key, value, scale))
            return 0;
    }
    double elapsed = 0.0;
    for (int iteration = 0; iteration < BENCH_ITERATIONS; iteration++) {
        double start = wall_seconds();
        if (!submit_attention(gpu, output, query, key, value, scale))
            return 0;
        elapsed += wall_seconds() - start;
    }
    if (!h3_gpu_tensor_read_bf16(output, host_output, elements)) {
        fprintf(stderr, "cannot read %s attention output: %s\n", backend,
                h3_gpu_error(gpu) ? h3_gpu_error(gpu) : "unknown CUDA error");
        return 0;
    }
    *mean_ms = elapsed * 1000.0 / (double)BENCH_ITERATIONS;
    return 1;
}

static attention_metrics compare_outputs(const uint16_t *native_output,
                                         const uint16_t *sage_output,
                                         size_t elements) {
    attention_metrics metrics = {0.0, 0.0, 0.0};
    double dot = 0.0;
    double native_norm = 0.0;
    double sage_norm = 0.0;
    for (size_t index = 0; index < elements; index++) {
        double native_value = (double)bf16_to_f32(native_output[index]);
        double sage_value = (double)bf16_to_f32(sage_output[index]);
        double difference = fabs(native_value - sage_value);
        metrics.mae += difference;
        if (difference > metrics.max_abs) metrics.max_abs = difference;
        dot += native_value * sage_value;
        native_norm += native_value * native_value;
        sage_norm += sage_value * sage_value;
    }
    metrics.mae /= (double)elements;
    if (native_norm > 0.0 && sage_norm > 0.0)
        metrics.cosine = dot / sqrt(native_norm * sage_norm);
    return metrics;
}

int main(void) {
    const size_t elements = (size_t)BENCH_HEADS * (size_t)BENCH_SEQUENCE *
                            (size_t)BENCH_HEAD_DIM;
    const float scale = 1.0f / sqrtf((float)BENCH_HEAD_DIM);
    uint16_t *query_bits = (uint16_t *)malloc(elements * sizeof(uint16_t));
    uint16_t *key_bits = (uint16_t *)malloc(elements * sizeof(uint16_t));
    uint16_t *value_bits = (uint16_t *)malloc(elements * sizeof(uint16_t));
    uint16_t *native_output = (uint16_t *)malloc(elements * sizeof(uint16_t));
    uint16_t *sage_output = (uint16_t *)malloc(elements * sizeof(uint16_t));
    if (!query_bits || !key_bits || !value_bits || !native_output ||
        !sage_output) {
        fprintf(stderr, "cannot allocate %zu BF16 elements for benchmark\n",
                elements);
        free(sage_output);
        free(native_output);
        free(value_bits);
        free(key_bits);
        free(query_bits);
        return 1;
    }
    uint32_t query_state = UINT32_C(0x12345678);
    uint32_t key_state = UINT32_C(0x31415926);
    uint32_t value_state = UINT32_C(0xdeadbeef);
    for (size_t index = 0; index < elements; index++) {
        query_bits[index] = f32_to_bf16(0.5f * random_symmetric(&query_state));
        key_bits[index] = f32_to_bf16(0.5f * random_symmetric(&key_state));
        value_bits[index] =
            f32_to_bf16(0.75f * random_symmetric(&value_state));
    }

    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    if (!gpu) {
        if (is_no_cuda_device_error(error)) {
            fprintf(stderr, "CUDA attention benchmark skipped: %s\n",
                    error[0] ? error : "CUDA device unavailable");
            free(sage_output);
            free(native_output);
            free(value_bits);
            free(key_bits);
            free(query_bits);
            return 77;
        }
        fprintf(stderr, "CUDA attention benchmark failed to initialize: %s\n",
                error[0] ? error : "unknown CUDA initialization error");
        free(sage_output);
        free(native_output);
        free(value_bits);
        free(key_bits);
        free(query_bits);
        return 1;
    }

    h3_gpu_tensor *query = h3_gpu_tensor_from_bf16(gpu, query_bits, elements);
    h3_gpu_tensor *key = h3_gpu_tensor_from_bf16(gpu, key_bits, elements);
    h3_gpu_tensor *value = h3_gpu_tensor_from_bf16(gpu, value_bits, elements);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, elements);
    int result = 1;
    if (!query || !key || !value || !output) {
        fprintf(stderr, "cannot allocate CUDA attention tensors: %s\n",
                h3_gpu_error(gpu) ? h3_gpu_error(gpu) : "unknown CUDA error");
        goto cleanup;
    }

    double native_ms = 0.0;
    double sage_ms = 0.0;
    if (!run_backend(gpu, "native", output, query, key, value, scale,
                     native_output, elements, &native_ms) ||
        !run_backend(gpu, "sage", output, query, key, value, scale,
                     sage_output, elements, &sage_ms)) {
        goto cleanup;
    }
    attention_metrics metrics =
        compare_outputs(native_output, sage_output, elements);
    double speedup = native_ms > 0.0 ? native_ms / sage_ms : 0.0;
    printf("CUDA attention B=1 H=%d N=%d D=%d warmup=%d iterations=%d\n",
           BENCH_HEADS, BENCH_SEQUENCE, BENCH_HEAD_DIM, BENCH_WARMUP,
           BENCH_ITERATIONS);
    printf("native_ms=%.3f sage_ms=%.3f speedup=%.3fx\n",
           native_ms, sage_ms, speedup);
    printf("output mae=%.6g max_abs=%.6g cosine=%.6f "
           "(thresholds mae<=0.10 max_abs<=1.0 cosine>=0.95)\n",
           metrics.mae, metrics.max_abs, metrics.cosine);
    if (sage_ms > native_ms) {
        fprintf(stderr, "SageAttention is slower than native attention\n");
        goto cleanup;
    }
    if (metrics.mae > 0.10 || metrics.max_abs > 1.0 ||
        metrics.cosine < 0.95) {
        fprintf(stderr, "SageAttention output parity is outside thresholds\n");
        goto cleanup;
    }
    result = 0;

cleanup:
    h3_gpu_tensor_free(output);
    h3_gpu_tensor_free(value);
    h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(query);
    h3_gpu_free(gpu);
    free(sage_output);
    free(native_output);
    free(value_bits);
    free(key_bits);
    free(query_bits);
    (void)set_attention_backend(NULL);
    return result;
}
