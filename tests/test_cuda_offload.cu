/* Focused CUDA offload/refill acceptance test.  It deliberately uses a
 * file-backed tensor larger than the staging buffer, then exceeds the weight
 * cache and reloads the first tensor.  CTest runs this once with
 * H3_CUDA_ASYNC_REFILL=0 and once with =1; both paths must return the same
 * deterministic canaries. */

#include "h3_cuda_common.cuh"
#include "h3_gpu.h"

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

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
           strstr(normalized, "insufficient driver") != NULL;
}

static float canary_value(size_t tensor, size_t chunk, size_t value) {
    return (float)(100000u + tensor * 10000u + chunk * 100u + value);
}

static int write_fixture(const char *path, size_t tensor_bytes,
                         size_t tensor_count, size_t chunk_bytes) {
    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    float *payload = (float *)calloc(tensor_bytes / sizeof(float),
                                     sizeof(float));
    if (!payload) {
        fclose(file);
        return 0;
    }
    for (size_t tensor = 0; tensor < tensor_count; tensor++) {
        const size_t chunks = tensor_bytes / chunk_bytes;
        for (size_t chunk = 0; chunk < chunks; chunk++) {
            size_t element = chunk * chunk_bytes / sizeof(float);
            for (size_t value = 0; value < 4; value++)
                payload[element + value] = canary_value(tensor, chunk, value);
        }
        const size_t tail = tensor_bytes / sizeof(float) - 4;
        for (size_t value = 0; value < 4; value++)
            payload[tail + value] = canary_value(tensor, chunks, value);
        if (fwrite(payload, 1, tensor_bytes, file) != tensor_bytes) {
            free(payload);
            fclose(file);
            return 0;
        }
        memset(payload, 0, tensor_bytes);
    }
    free(payload);
    return fclose(file) == 0;
}

static int check_tensor(h3_gpu_tensor *tensor, size_t tensor_index,
                        size_t tensor_bytes, size_t chunk_bytes) {
    const size_t chunks = tensor_bytes / chunk_bytes;
    for (size_t chunk = 0; chunk <= chunks; chunk++) {
        const size_t element = chunk < chunks ?
            chunk * chunk_bytes / sizeof(float) :
            tensor_bytes / sizeof(float) - 4;
        float values[4] = {0};
        if (!h3_gpu_tensor_read_f32_range(tensor, element, values, 4))
            return 0;
        for (size_t value = 0; value < 4; value++) {
            float expected = canary_value(tensor_index, chunk, value);
            if (values[value] != expected) {
                fprintf(stderr,
                        "canary mismatch tensor=%zu chunk=%zu value=%zu got=%g expected=%g\n",
                        tensor_index, chunk, value, (double)values[value],
                        (double)expected);
                return 0;
            }
        }
    }
    return 1;
}

int main(void) {
    const size_t tensor_bytes = 32u * 1024u * 1024u;
    const size_t tensor_elements = tensor_bytes / sizeof(float);
    const size_t tensor_count = 5;
    const size_t chunk_bytes = 2u * 1024u * 1024u;
    char fixture[96];
#if defined(_WIN32)
    int process_id = _getpid();
#else
    int process_id = (int)getpid();
#endif
    CHECK(snprintf(fixture, sizeof(fixture),
                   "h3cspeed-test-cuda-offload-%d.bin", process_id) > 0);
    CHECK(write_fixture(fixture, tensor_bytes, tensor_count, chunk_bytes));

    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    if (!gpu) {
        remove(fixture);
        if (is_no_cuda_device_error(error)) {
            fprintf(stderr, "CUDA offload test skipped: %s\n", error);
            return 77;
        }
        fprintf(stderr, "CUDA offload context creation failed: %s\n", error);
        return 1;
    }

    h3_gpu_tensor *tensors[tensor_count] = {};
    int ok = 1;
    for (size_t index = 0; index < tensor_count; index++) {
        tensors[index] = h3_gpu_tensor_load_f32(
            gpu, fixture, (uint64_t)(index * tensor_bytes), tensor_elements);
        if (!tensors[index] || !check_tensor(tensors[index], index,
                                              tensor_bytes, chunk_bytes)) {
            fprintf(stderr, "tensor %zu failed: %s\n", index,
                    h3_gpu_error(gpu));
            ok = 0;
            break;
        }
        if (index == 0) {
            const char *async_value = getenv("H3_CUDA_ASYNC_REFILL");
            const int requested = async_value && strcmp(async_value, "1") == 0;
            if (requested) {
                CHECK(gpu->async_refill_enabled == 1);
                CHECK(gpu->staging_pinned == 1);
                CHECK(gpu->staging_slot_bytes > 0);
                CHECK(gpu->staging_slots[0] != nullptr &&
                      gpu->staging_slots[1] != nullptr);
                CHECK(gpu->staging_done[0] != nullptr &&
                      gpu->staging_done[1] != nullptr);
                CHECK(gpu->staging_done_valid[0] == 1 &&
                      gpu->staging_done_valid[1] == 1);
            } else {
                CHECK(gpu->async_refill_enabled == 0);
                CHECK(gpu->staging_slot_bytes == 0);
                CHECK(gpu->staging_slots[0] == nullptr &&
                      gpu->staging_slots[1] == nullptr);
            }
        }
    }
    /* The fifth tensor exceeds the configured 128 MiB weight cache.  Reading
     * tensor zero again must rebuild it from its authoritative file source. */
    if (ok) ok = check_tensor(tensors[0], 0, tensor_bytes, chunk_bytes);
    if (ok) {
        CHECK(gpu->offload_evictions >= 1);
        CHECK(gpu->offload_uploads >= 6);
        CHECK(gpu->file_fallback_reads >= 6);
    }
    for (size_t index = 0; index < tensor_count; index++)
        h3_gpu_tensor_free(tensors[index]);
    h3_gpu_free(gpu);
    remove(fixture);
    return ok ? 0 : 1;
}
