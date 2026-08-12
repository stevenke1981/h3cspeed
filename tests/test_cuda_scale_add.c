#include "h3_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", \
                __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

int main(void) {
    enum { ROWS = 2, WIDTH = 4, ELEMENTS = ROWS * WIDTH };
    const float residual_values[ELEMENTS] = {0.0f};
    const float branch_values[ELEMENTS] = {
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
    };
    const float scale_values[WIDTH] = {10.0f, 20.0f, 30.0f, 40.0f};
    const float expected[ELEMENTS] = {
        10.0f, 20.0f, 30.0f, 40.0f,
        10.0f, 20.0f, 30.0f, 40.0f,
    };
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "CUDA scale-add test skipped: %s\n",
                error[0] ? error : "CUDA device unavailable");
        return 77;
    }
    h3_gpu_tensor *residual = h3_gpu_tensor_from_f32(
        gpu, residual_values, ELEMENTS);
    h3_gpu_tensor *branch = h3_gpu_tensor_from_f32(
        gpu, branch_values, ELEMENTS);
    h3_gpu_tensor *scale = h3_gpu_tensor_from_f32(gpu, scale_values, WIDTH);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, ELEMENTS);
    CHECK(residual && branch && scale && output);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_scale_add_f32(gpu, output, residual, branch, scale,
                               ROWS, WIDTH));
    CHECK(h3_gpu_submit(gpu));
    float actual[ELEMENTS] = {0.0f};
    CHECK(h3_gpu_tensor_read_f32(output, actual, ELEMENTS));
    for (size_t index = 0; index < ELEMENTS; index++)
        CHECK(fabsf(actual[index] - expected[index]) < 1e-6f);
    h3_gpu_tensor_free(output);
    h3_gpu_tensor_free(scale);
    h3_gpu_tensor_free(branch);
    h3_gpu_tensor_free(residual);
    h3_gpu_free(gpu);
    puts("CUDA scale-add test passed");
    return 0;
}
