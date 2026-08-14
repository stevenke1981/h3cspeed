/* Focused PERF-006 proof for the private one-ahead DiT prefetch contract.
 * The enabled run starts a real compute kernel, joins the host-side file read
 * in the helper call, and verifies that its H2D interval overlaps the started
 * kernel.  It then evicts and reloads the same file-backed tensor.  The
 * disabled run proves the helper is a no-op and leaves the existing lazy
 * upload path responsible for the numeric canary. */

#include "h3_cuda_common.cuh"
#include "h3_gpu.h"
#include "h3_gpu_cuda_private.h"

#include <algorithm>
#include <chrono>
#include <ctype.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>

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

static int no_cuda_error(const char *error) {
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

static __global__ void prefetch_compute_kernel(unsigned long long cycles,
                                                int *started) {
    if (blockIdx.x || threadIdx.x) return;
    if (started) {
        *started = 1;
        __threadfence_system();
    }
    unsigned long long begin = clock64();
    while (clock64() - begin < cycles) { }
}

static uint64_t fnv1a_file(const char *path, uint64_t *size) {
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    uint64_t total = 0;
    uint64_t hash = UINT64_C(1469598103934665603);
    unsigned char buffer[64u * 1024u];
    size_t got = 0;
    while ((got = fread(buffer, 1, sizeof(buffer), file)) != 0) {
        total += (uint64_t)got;
        for (size_t index = 0; index < got; index++) {
            hash ^= (uint64_t)buffer[index];
            hash *= UINT64_C(1099511628211);
        }
    }
    if (size) *size = total;
    if (ferror(file) || fclose(file) != 0) return 0;
    return hash;
}

static int write_fixture(const char *path, size_t bytes, size_t tensors) {
    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    const size_t elements = bytes / sizeof(float);
    float *payload = (float *)malloc(bytes);
    if (!payload) {
        fclose(file);
        return 0;
    }
    for (size_t tensor = 0; tensor < tensors; tensor++) {
        for (size_t index = 0; index < elements; index++)
            payload[index] = (float)(1000u + tensor * 100u + index % 17u);
        if (fwrite(payload, 1, bytes, file) != bytes) {
            free(payload);
            fclose(file);
            return 0;
        }
    }
    free(payload);
    return fclose(file) == 0;
}

static int check_canary(const h3_gpu_tensor *tensor, size_t tensor_index,
                        size_t bytes, size_t chunk_bytes) {
    float values[4] = {0};
    const size_t elements = bytes / sizeof(float);
    const size_t chunk_elements = chunk_bytes / sizeof(float);
    if (!chunk_elements || elements % chunk_elements != 0) return 0;
    for (size_t chunk = 0; chunk < elements / chunk_elements; chunk++) {
        const size_t offsets[2] = {
            chunk * chunk_elements,
            (chunk + 1) * chunk_elements - 4,
        };
        for (size_t boundary = 0; boundary < 2; boundary++) {
            if (!h3_gpu_tensor_read_f32_range(
                    tensor, offsets[boundary], values, 4)) return 0;
            for (size_t index = 0; index < 4; index++) {
                size_t source = offsets[boundary] + index;
                float expected = (float)(
                    1000u + tensor_index * 100u + source % 17u);
                if (values[index] != expected) return 0;
            }
        }
    }
    return 1;
}

int main(void) {
    const size_t tensor_bytes = 64u * 1024u * 1024u;
    const size_t tensor_count = 3;
    char fixture[128];
#if defined(_WIN32)
    int process_id = _getpid();
#else
    int process_id = (int)getpid();
#endif
    CHECK(snprintf(fixture, sizeof(fixture),
                   "h3cspeed-test-cuda-dit-prefetch-%d.bin", process_id) > 0);
    CHECK(write_fixture(fixture, tensor_bytes, tensor_count));
    uint64_t fixture_size_before = 0;
    const uint64_t fixture_hash_before = fnv1a_file(
        fixture, &fixture_size_before);
    CHECK(fixture_hash_before != 0);

    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    if (!gpu) {
        remove(fixture);
        if (no_cuda_error(error)) {
            fprintf(stderr, "CUDA DiT prefetch test skipped: %s\n", error);
            return 77;
        }
        fprintf(stderr, "CUDA DiT prefetch context failed: %s\n", error);
        return 1;
    }
    const char *env = getenv("H3_CUDA_DIT_PREFETCH");
    const int requested = env && strcmp(env, "1") == 0;
    h3cspeed_cuda_profile_dit_scope_begin(gpu, 0, requested);
    h3_gpu_tensor *first = h3_gpu_tensor_load_f32(
        gpu, fixture, 0, tensor_bytes / sizeof(float));
    h3_gpu_tensor *second = h3_gpu_tensor_load_f32(
        gpu, fixture, tensor_bytes, tensor_bytes / sizeof(float));
    h3_gpu_tensor *third = h3_gpu_tensor_load_f32(
        gpu, fixture, tensor_bytes * 2, tensor_bytes / sizeof(float));
    CHECK(first && second && third);

    if (!requested) {
        CHECK(first->data == nullptr);
        CHECK(h3cspeed_cuda_reserve_prefetch_weight(gpu, first) == 1);
        CHECK(first->data == nullptr);
        CHECK(h3cspeed_cuda_prefetch_weight(gpu, first) == 1);
        CHECK(first->data == nullptr);
        CHECK(check_canary(first, 0, tensor_bytes, tensor_bytes));
    } else {
        cudaEvent_t origin = nullptr;
        cudaEvent_t compute_start = nullptr;
        cudaEvent_t compute_end = nullptr;
        CHECK(cudaEventCreate(&origin) == cudaSuccess);
        CHECK(cudaEventCreate(&compute_start) == cudaSuccess);
        CHECK(cudaEventCreate(&compute_end) == cudaSuccess);
        CHECK(first->data == nullptr);
        CHECK(h3cspeed_cuda_reserve_prefetch_weight(gpu, first) == 1);
        CHECK(first->data != nullptr);
        CHECK(first->pin_epoch != gpu->operation_epoch);
        CHECK(first->ready_valid == 0);
        volatile int *started_host = nullptr;
        int *started_device = nullptr;
        CHECK(cudaHostAlloc((void **)&started_host, sizeof(*started_host),
                            cudaHostAllocMapped) == cudaSuccess);
        *started_host = 0;
        CHECK(cudaHostGetDevicePointer((void **)&started_device,
                                       (void *)started_host, 0) == cudaSuccess);
        CHECK(cudaEventRecord(origin, gpu->compute_stream) == cudaSuccess);
        CHECK(cudaEventSynchronize(origin) == cudaSuccess);
        CHECK(cudaEventRecord(compute_start, gpu->compute_stream) ==
              cudaSuccess);
        int clock_khz = 0;
        CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate,
                                     gpu->device) == cudaSuccess);
        const unsigned long long cycles =
            (unsigned long long)(clock_khz > 0 ? clock_khz : 1000000) * 20ull;
        for (size_t kernel = 0; kernel < 64; kernel++) {
            prefetch_compute_kernel<<<1, 1, 0, gpu->compute_stream>>>(
                cycles, kernel == 0 ? started_device : nullptr);
            CHECK(cudaGetLastError() == cudaSuccess);
        }
        CHECK(cudaEventRecord(compute_end, gpu->compute_stream) ==
              cudaSuccess);
        CHECK(cudaEventSynchronize(compute_start) == cudaSuccess);
        const auto started_deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!*started_host &&
               std::chrono::steady_clock::now() < started_deadline)
            std::this_thread::yield();
        CHECK(*started_host == 1);
        const uint64_t expected_refill = gpu->refill_trace_next_refill_id + 1;
        CHECK(first->pin_epoch != gpu->operation_epoch);
        CHECK(h3cspeed_cuda_prefetch_weight(gpu, first) == 1);
        CHECK(first->data != nullptr);
        CHECK(first->pin_epoch != gpu->operation_epoch);
        CHECK(h3_gpu_submit(gpu) == 1);
        CHECK(gpu->staging_slot_bytes > 0);
        CHECK(check_canary(first, 0, tensor_bytes,
                           gpu->staging_slot_bytes));

        float compute_start_ms = 0.0f, compute_end_ms = 0.0f;
        CHECK(cudaEventElapsedTime(&compute_start_ms, origin,
                                   compute_start) == cudaSuccess);
        CHECK(cudaEventElapsedTime(&compute_end_ms, origin,
                                   compute_end) == cudaSuccess);
        CHECK(compute_end_ms > compute_start_ms);
        double overlap_ms = 0.0;
        uint64_t chunks = 0;
        uint64_t overlapping_chunks = 0;
        for (size_t index = 0; index < H3CSPEED_REFILL_TRACE_CAPACITY;
             index++) {
            const h3cspeed_refill_trace_entry *entry =
                &gpu->refill_trace_entries[index];
            if (!entry->valid || entry->refill_id != expected_refill) continue;
            float start_ms = 0.0f, end_ms = 0.0f;
            CHECK(cudaEventElapsedTime(&start_ms, origin,
                                       entry->h2d_start) == cudaSuccess);
            CHECK(cudaEventElapsedTime(&end_ms, origin,
                                       entry->h2d_end) == cudaSuccess);
            const double start = std::max((double)start_ms,
                                          (double)compute_start_ms);
            const double end = std::min((double)end_ms,
                                        (double)compute_end_ms);
            if (end > start) {
                overlap_ms = std::max(overlap_ms, end - start);
                overlapping_chunks++;
            }
            chunks++;
        }
        CHECK(tensor_bytes % gpu->staging_slot_bytes == 0);
        CHECK(chunks == tensor_bytes / gpu->staging_slot_bytes);
        fprintf(stderr, "DiT prefetch overlap: %" PRIu64 "/%" PRIu64
                        " chunks, max %.6f ms\n",
                overlapping_chunks, chunks, overlap_ms);
        CHECK(overlapping_chunks > 0);
        CHECK(overlap_ms >= 0.05);
        printf("{\"kind\":\"h3cspeed.cuda.dit_prefetch\",\"h2d_overlap_ms\":%.6f,\"chunks\":%" PRIu64 ",\"overlapping_chunks\":%" PRIu64 ",\"overlap\":\"PASS\"}\n",
               overlap_ms, chunks, overlapping_chunks);
        CHECK(cudaFreeHost((void *)started_host) == cudaSuccess);
        CHECK(cudaEventDestroy(origin) == cudaSuccess);
        CHECK(cudaEventDestroy(compute_start) == cudaSuccess);
        CHECK(cudaEventDestroy(compute_end) == cudaSuccess);

        /* A future-block reservation is an atomic ownership boundary: the
         * allocator may not evict an earlier member of the same batch. */
        CHECK(h3cspeed_cuda_reserve_prefetch_weight(gpu, first) == 1);
        CHECK(first->prefetch_reserved == 1);
        CHECK(h3cspeed_cuda_reserve_prefetch_weight(gpu, second) == 1);
        CHECK(second->prefetch_reserved == 1);
        CHECK(h3cspeed_cuda_reserve_prefetch_weight(gpu, third) == 0);
        CHECK(third->data == nullptr);

        /* Future tensors are deliberately not pinned by the helper.  A tiny
         * weight cache must therefore evict and rebuild the first source. */
        h3cspeed_cuda_cancel_prefetch_weight(gpu, first);
        CHECK(h3cspeed_cuda_prefetch_weight(gpu, second) == 1);
        CHECK(h3_gpu_submit(gpu) == 1);
        h3cspeed_cuda_cancel_prefetch_weight(gpu, second);
        CHECK(h3cspeed_cuda_prefetch_weight(gpu, third) == 1);
        CHECK(h3_gpu_submit(gpu) == 1);
        h3cspeed_cuda_cancel_prefetch_weight(gpu, third);
        CHECK(first->data == nullptr);
        CHECK(h3cspeed_cuda_reserve_prefetch_weight(gpu, first) == 1);
        CHECK(first->data != nullptr);
        CHECK(first->ready_valid == 0);
        CHECK(h3cspeed_cuda_prefetch_weight(gpu, first) == 1);
        CHECK(h3_gpu_submit(gpu) == 1);
        CHECK(check_canary(first, 0, tensor_bytes,
                           gpu->staging_slot_bytes));
        CHECK(gpu->offload_evictions == 2);
        CHECK(gpu->offload_uploads == 4);
        CHECK(gpu->file_fallback_reads == 4);

        char bad_fixture[128];
        CHECK(snprintf(bad_fixture, sizeof(bad_fixture),
                       "h3cspeed-test-cuda-dit-prefetch-bad-%d.bin",
                       process_id) > 0);
        CHECK(write_fixture(bad_fixture, tensor_bytes, 1));
        h3_gpu_tensor *failed = h3_gpu_tensor_load_f32(
            gpu, bad_fixture, 0, tensor_bytes / sizeof(float));
        CHECK(failed != nullptr);
        CHECK(h3cspeed_cuda_reserve_prefetch_weight(gpu, failed) == 1);
        CHECK(failed->data != nullptr);
        CHECK(remove(bad_fixture) == 0);
        CHECK(h3cspeed_cuda_prefetch_weight(gpu, failed) == 0);
        CHECK(failed->data == nullptr);
        h3_gpu_tensor_free(failed);
    }

    uint64_t fixture_size_after = 0;
    CHECK(fnv1a_file(fixture, &fixture_size_after) == fixture_hash_before);
    CHECK(fixture_size_after == fixture_size_before);
    CHECK(h3_gpu_submit(gpu) == 1);
    const char *trace_env = getenv("H3_CUDA_UPLOAD_WAIT_TRACE");
    const int trace_requested = trace_env && strcmp(trace_env, "1") == 0;
    if (trace_requested) {
        CHECK(gpu->upload_wait_trace_requested == 1);
        CHECK(gpu->upload_wait_trace_complete == 1);
        CHECK(gpu->upload_ready_wait_count > 0);
        CHECK(gpu->upload_ready_wait_seconds >= 0.0);
        CHECK(gpu->upload_wait_trace_union_valid == 1);
    } else {
        CHECK(gpu->upload_wait_trace_requested == 0);
        CHECK(gpu->upload_ready_wait_count == 0);
        CHECK(gpu->upload_wait_trace_complete == 0);
    }
    h3cspeed_cuda_profile_dit_scope_end(gpu);
    h3_gpu_tensor_free(first);
    h3_gpu_tensor_free(second);
    h3_gpu_tensor_free(third);
    h3_gpu_free(gpu);

    if (requested) {
        char pin_error[512] = {0};
        h3_gpu *pin_gpu = h3_gpu_create(NULL, pin_error, sizeof(pin_error));
        CHECK(pin_gpu != nullptr);
        h3_gpu_tensor *pinned = h3_gpu_tensor_load_f32(
            pin_gpu, fixture, 0, tensor_bytes / sizeof(float));
        CHECK(pinned != nullptr);
        CHECK(h3cspeed_tensor_prepare(pin_gpu, pinned) == 1);
        CHECK(pinned->pin_epoch == pin_gpu->operation_epoch);
        CHECK(h3cspeed_cuda_reserve_prefetch_weight(pin_gpu, pinned) == 0);
        CHECK(h3cspeed_cuda_prime_prefetch_weight(pin_gpu, pinned) == 1);
        CHECK(pinned->pin_epoch == pin_gpu->operation_epoch);
        CHECK(pinned->prefetch_reserved == 1);
        /* Sampler windows may reuse the command chain without submit.  Each
         * completed kernel still advances the private operation epoch, so a
         * weight consumed by the prior evaluation becomes a valid future
         * reservation in the next evaluation. */
        h3cspeed_operation_complete(pin_gpu);
        CHECK(pinned->pin_epoch != pin_gpu->operation_epoch);
        CHECK(h3cspeed_cuda_reserve_prefetch_weight(pin_gpu, pinned) == 1);
        CHECK(pinned->prefetch_reserved == 1);
        h3cspeed_cuda_cancel_prefetch_weight(pin_gpu, pinned);
        CHECK(h3_gpu_submit(pin_gpu) == 1);
        h3_gpu_tensor_free(pinned);
        h3_gpu_free(pin_gpu);
    }
    remove(fixture);
    return 0;
}
