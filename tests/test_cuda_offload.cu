/* Focused CUDA offload/refill acceptance test.  It deliberately uses a
 * file-backed tensor larger than the staging buffer, then exceeds the weight
 * cache and reloads the first tensor.  CTest runs this with
 * H3_CUDA_ASYNC_REFILL=0, =1, and a non-zero host-cache promote path.  All
 * three must return the same deterministic canaries. */

#include "h3_cuda_common.cuh"
#include "h3_gpu.h"

#include <ctype.h>
#include <algorithm>
#include <inttypes.h>
#include <thread>
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

static __global__ void timeline_delay_kernel(unsigned long long cycles,
                                             int *started) {
    if (blockIdx.x || threadIdx.x) return;
    if (started) {
        *started = 1;
        __threadfence_system();
    }
    const unsigned long long cycle_started = clock64();
    while (clock64() - cycle_started < cycles) { }
}

static int file_fnv1a(const char *path, uint64_t *size, uint64_t *hash) {
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    uint64_t total = 0;
    uint64_t value = UINT64_C(1469598103934665603);
    unsigned char buffer[64u * 1024u];
    size_t got = 0;
    while ((got = fread(buffer, 1, sizeof(buffer), file)) != 0) {
        total += (uint64_t)got;
        for (size_t index = 0; index < got; index++) {
            value ^= (uint64_t)buffer[index];
            value *= UINT64_C(1099511628211);
        }
    }
    const int ok = ferror(file) == 0 && fclose(file) == 0;
    if (!ok) return 0;
    if (size) *size = total;
    if (hash) *hash = value;
    return 1;
}

static uint64_t host_now_ns(void) {
    const double seconds = h3cspeed_profile_now_seconds();
    return seconds > 0.0 ? (uint64_t)(seconds * 1000000000.0) : 0;
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
    uint64_t fixture_size_before = 0, fixture_hash_before = 0;
    CHECK(file_fnv1a(fixture, &fixture_size_before, &fixture_hash_before));

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
    const int trace_requested = getenv("H3_CUDA_REFILL_TRACE") &&
                                strcmp(getenv("H3_CUDA_REFILL_TRACE"), "1") == 0;
    int timeline_pass = 0;
    double timeline_h2d_overlap_ms = 0.0;
    double timeline_host_read_ms = 0.0;
    uint64_t timeline_chunks = 0;
    int timeline_host_read_nested = 0;
    for (size_t index = 0; index < tensor_count; index++) {
        tensors[index] = h3_gpu_tensor_load_f32(
            gpu, fixture, (uint64_t)(index * tensor_bytes), tensor_elements);
        if (!tensors[index]) {
            fprintf(stderr, "tensor %zu failed: %s\n", index,
                    h3_gpu_error(gpu));
            ok = 0;
            break;
        }
        if (index == 1 && trace_requested) {
            cudaEvent_t origin = nullptr;
            cudaEvent_t compute_start = nullptr;
            cudaEvent_t compute_end = nullptr;
            CHECK(cudaEventCreate(&origin) == cudaSuccess);
            CHECK(cudaEventCreate(&compute_start) == cudaSuccess);
            CHECK(cudaEventCreate(&compute_end) == cudaSuccess);
            volatile int *compute_started_host = nullptr;
            int *compute_started_device = nullptr;
            CHECK(cudaHostAlloc((void **)&compute_started_host, sizeof(int),
                                cudaHostAllocMapped) == cudaSuccess);
            *compute_started_host = 0;
            CHECK(cudaHostGetDevicePointer((void **)&compute_started_device,
                                           (void *)compute_started_host, 0) ==
                  cudaSuccess);
            CHECK(cudaEventRecord(origin, gpu->compute_stream) == cudaSuccess);
            CHECK(cudaEventSynchronize(origin) == cudaSuccess);
            CHECK(cudaEventRecord(compute_start, gpu->compute_stream) == cudaSuccess);
            int clock_khz = 0;
            CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate,
                                         gpu->device) == cudaSuccess);
            const unsigned long long cycles =
                (unsigned long long)(clock_khz > 0 ? clock_khz : 1000000) * 20ull;
            for (size_t kernel = 0; kernel < 32; kernel++) {
                timeline_delay_kernel<<<1, 1, 0, gpu->compute_stream>>>(
                    cycles, kernel == 0 ? compute_started_device : nullptr);
                CHECK(cudaGetLastError() == cudaSuccess);
            }
            CHECK(cudaEventRecord(compute_end, gpu->compute_stream) == cudaSuccess);
            CHECK(cudaEventSynchronize(compute_start) == cudaSuccess);
            const double start_deadline =
                h3cspeed_profile_now_seconds() + 5.0;
            while (*compute_started_host == 0 &&
                   h3cspeed_profile_now_seconds() < start_deadline)
                std::this_thread::yield();
            CHECK(*compute_started_host == 1);
            const uint64_t expected_refill_id =
                gpu->refill_trace_next_refill_id + 1;
            const uint64_t prepare_host_start_ns = host_now_ns();
            CHECK(h3cspeed_tensor_prepare(gpu, tensors[index]) == 1);
            const uint64_t prepare_host_end_ns = host_now_ns();
            CHECK(gpu->refill_trace_enabled == 1);
            CHECK(cudaEventQuery(compute_end) == cudaErrorNotReady);
            CHECK(check_tensor(tensors[index], index, tensor_bytes, chunk_bytes));

            float compute_start_ms = 0.0f, compute_end_ms = 0.0f;
            CHECK(cudaEventElapsedTime(&compute_start_ms, origin,
                                       compute_start) == cudaSuccess);
            CHECK(cudaEventElapsedTime(&compute_end_ms, origin,
                                       compute_end) == cudaSuccess);
            CHECK(compute_end_ms > compute_start_ms);
            uint64_t first_refill_id = 0;
            int found_first_refill = 0;
            for (size_t trace_index = 0;
                 trace_index < H3CSPEED_REFILL_TRACE_CAPACITY; trace_index++) {
                const h3cspeed_refill_trace_entry *entry =
                    &gpu->refill_trace_entries[trace_index];
                if (entry->valid && entry->refill_id == expected_refill_id &&
                    entry->chunk_index == 0) {
                    first_refill_id = entry->refill_id;
                    found_first_refill = 1;
                    break;
                }
            }
            CHECK(found_first_refill == 1);
            CHECK(first_refill_id == expected_refill_id);
            uint64_t previous_slot_sequence[2] = {0, 0};
            for (size_t trace_index = 0;
                 trace_index < H3CSPEED_REFILL_TRACE_CAPACITY; trace_index++) {
                const h3cspeed_refill_trace_entry *entry =
                    &gpu->refill_trace_entries[trace_index];
                if (entry->valid && entry->sequence <
                        gpu->refill_trace_next_sequence &&
                    entry->refill_id != first_refill_id &&
                    entry->sequence > previous_slot_sequence[entry->slot])
                    previous_slot_sequence[entry->slot] = entry->sequence;
            }
            uint64_t previous_sequence = 0;
            timeline_host_read_nested = 1;
            for (size_t trace_index = 0;
                 trace_index < H3CSPEED_REFILL_TRACE_CAPACITY; trace_index++) {
                const h3cspeed_refill_trace_entry *entry =
                    &gpu->refill_trace_entries[trace_index];
                if (!entry->valid || entry->refill_id != first_refill_id) continue;
                CHECK(entry->sequence > previous_sequence);
                previous_sequence = entry->sequence;
                CHECK(entry->reuse_after_sequence ==
                      previous_slot_sequence[entry->slot]);
                previous_slot_sequence[entry->slot] = entry->sequence;
                CHECK(entry->host_read_end_ns >= entry->host_read_start_ns);
                if (entry->host_read_start_ns < prepare_host_start_ns ||
                    entry->host_read_end_ns > prepare_host_end_ns)
                    timeline_host_read_nested = 0;
                timeline_host_read_ms +=
                    (double)(entry->host_read_end_ns - entry->host_read_start_ns) /
                    1000000.0;
                CHECK(cudaEventQuery(entry->h2d_start) == cudaSuccess);
                CHECK(cudaEventQuery(entry->h2d_end) == cudaSuccess);
                float h2d_start_ms = 0.0f, h2d_end_ms = 0.0f;
                CHECK(cudaEventElapsedTime(&h2d_start_ms, origin,
                                           entry->h2d_start) == cudaSuccess);
                CHECK(cudaEventElapsedTime(&h2d_end_ms, origin,
                                           entry->h2d_end) == cudaSuccess);
                const double overlap_start =
                    std::max((double)h2d_start_ms, (double)compute_start_ms);
                const double overlap_end =
                    std::min((double)h2d_end_ms, (double)compute_end_ms);
                if (overlap_end > overlap_start)
                    timeline_h2d_overlap_ms = std::max(
                        timeline_h2d_overlap_ms, overlap_end - overlap_start);
                timeline_chunks++;
            }
            CHECK(timeline_chunks == tensor_bytes / gpu->staging_slot_bytes);
            CHECK(timeline_host_read_ms > 0.0);
            CHECK(timeline_h2d_overlap_ms >= 0.05);
            CHECK(timeline_host_read_nested == 1);
            timeline_pass = 1;
            CHECK(cudaEventDestroy(origin) == cudaSuccess);
            CHECK(cudaEventDestroy(compute_start) == cudaSuccess);
            CHECK(cudaEventDestroy(compute_end) == cudaSuccess);
            CHECK(cudaFreeHost((void *)compute_started_host) == cudaSuccess);
        } else if (!check_tensor(tensors[index], index,
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
                CHECK(gpu->refill_trace_enabled == 0);
            }
        }
    }
    /* The fifth tensor exceeds the configured 128 MiB weight cache.  Reading
     * tensor zero again must rebuild it from its authoritative file source. */
    if (ok) ok = check_tensor(tensors[0], 0, tensor_bytes, chunk_bytes);
    if (ok) {
        const char *host_cache = getenv("H3_CUDA_HOST_CACHE_MIB");
        const int host_promote = host_cache && *host_cache &&
                                 strcmp(host_cache, "0") != 0;
        CHECK(gpu->offload_evictions >= 1);
        CHECK(gpu->fence_ready_evictions >= 1);
        CHECK(gpu->offload_uploads >= 6);
        if (host_promote) {
            CHECK(tensors[0]->host_data != nullptr);
            CHECK(tensors[0]->host_valid == 1);
            CHECK(gpu->host_cache_promotions >= 1);
            CHECK(gpu->file_fallback_reads == 0);
            const uint64_t fallback_before = gpu->file_fallback_reads;
            const uint64_t promotions_before = gpu->host_cache_promotions;
            CHECK(check_tensor(tensors[0], 0, tensor_bytes, chunk_bytes));
            CHECK(gpu->file_fallback_reads == fallback_before);
            CHECK(gpu->host_cache_promotions == promotions_before);
        } else {
            CHECK(gpu->file_fallback_reads >= 6);
        }
        CHECK(gpu->weight_reuse_stores >= 1);
        CHECK(gpu->weight_reuse_hits >= 1);
        CHECK(gpu->weight_reuse_pool_count <=
              H3CSPEED_WEIGHT_REUSE_POOL_CAPACITY);
        CHECK(gpu->weight_reuse_pool_bytes <=
              gpu->offload.weight_cache_bytes / 3u);
        CHECK(gpu->device_live_bytes <= gpu->offload.vram_budget_bytes);
        uint64_t fixture_size_after = 0, fixture_hash_after = 0;
        CHECK(file_fnv1a(fixture, &fixture_size_after, &fixture_hash_after));
        CHECK(fixture_size_after == fixture_size_before);
        CHECK(fixture_hash_after == fixture_hash_before);
        if (trace_requested) {
            CHECK(timeline_pass == 1);
            printf("{\"kind\":\"h3cspeed.cuda.refill_timeline\",\"h2d_overlap_ms\":%.6f,\"host_read_ms\":%.6f,\"chunks\":%" PRIu64 ",\"fence_ready_evictions\":%" PRIu64 ",\"h2d_overlap\":\"PASS\",\"host_read_nested_in_compute_window\":\"PASS\"}\n",
                   timeline_h2d_overlap_ms, timeline_host_read_ms,
                   timeline_chunks, gpu->fence_ready_evictions);
        }
    }
    for (size_t index = 0; index < tensor_count; index++)
        h3_gpu_tensor_free(tensors[index]);
    h3_gpu_free(gpu);
    remove(fixture);
    return ok ? 0 : 1;
}
