#include "h3_cuda_common.cuh"
#include "h3_quantized_weights.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cuda_fp16.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits>
#include <stdarg.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static double elapsed_seconds(const struct timespec *start,
                              const struct timespec *stop) {
    return (double)(stop->tv_sec - start->tv_sec) +
           (double)(stop->tv_nsec - start->tv_nsec) / 1e9;
}

static int environment_flag_enabled(const char *name) {
    const char *value = name ? getenv(name) : nullptr;
    return value && *value && strcmp(value, "0") != 0 &&
           strcasecmp(value, "off") != 0 &&
           strcasecmp(value, "false") != 0 &&
           strcasecmp(value, "no") != 0;
}

static int async_refill_requested(void) {
    const char *value = getenv("H3_CUDA_ASYNC_REFILL");
    return value && strcmp(value, "1") == 0;
}

static int refill_trace_requested(void) {
    const char *value = getenv("H3_CUDA_REFILL_TRACE");
    return value && strcmp(value, "1") == 0;
}

static int dit_prefetch_requested(void);

static uint64_t refill_trace_now_ns(void) {
    const double seconds = h3cspeed_profile_now_seconds();
    if (!(seconds > 0.0)) return 0;
    const double nanos = seconds * 1000000000.0;
    if (nanos >= (double)UINT64_MAX) return UINT64_MAX;
    return (uint64_t)nanos;
}

enum h3cspeed_profile_stream {
    H3CSPEED_PROFILE_STREAM_COMPUTE = 0,
    H3CSPEED_PROFILE_STREAM_UPLOAD = 1
};

static std::atomic<uint64_t> profile_context_next{1};

static uint64_t next_profile_context_id(void) {
    uint64_t value = profile_context_next.fetch_add(1,
                                                     std::memory_order_relaxed);
    return value ? value : profile_context_next.fetch_add(
        1, std::memory_order_relaxed);
}

static void profile_update(h3_gpu *gpu,
                           void (*update)(h3cspeed_profile_metrics *,
                                          uint64_t, double),
                           uint64_t bytes, double seconds) {
    if (!gpu || !gpu->profile_enabled || !update) return;
    pthread_mutex_lock(&gpu->lock);
    update(&gpu->profile_metrics, bytes, seconds);
    pthread_mutex_unlock(&gpu->lock);
}

static void profile_file_read(h3cspeed_profile_metrics *metrics,
                              uint64_t bytes, double seconds) {
    metrics->file_read_calls++;
    metrics->file_read_bytes += bytes;
    metrics->file_read_seconds += seconds;
}

static void profile_pageable_copy(h3cspeed_profile_metrics *metrics,
                                  uint64_t bytes, double seconds) {
    metrics->pageable_copy_calls++;
    metrics->pageable_copy_bytes += bytes;
    metrics->pageable_copy_seconds += seconds;
}

static void profile_h2d_enqueue(h3cspeed_profile_metrics *metrics,
                                uint64_t bytes, double seconds) {
    metrics->h2d_enqueue_calls++;
    metrics->h2d_enqueue_bytes += bytes;
    metrics->h2d_enqueue_seconds += seconds;
}

static void profile_allocation(h3cspeed_profile_metrics *metrics,
                               uint64_t bytes, double seconds) {
    (void)bytes;
    metrics->allocation_calls++;
    metrics->allocation_seconds += seconds;
}

static cudaError_t profiled_stream_synchronize(
        h3_gpu *gpu, cudaStream_t stream, h3cspeed_profile_stream kind) {
    if (!gpu || !gpu->profile_enabled) return cudaStreamSynchronize(stream);
    double started = h3cspeed_profile_now_seconds();
    cudaError_t status = cudaStreamSynchronize(stream);
    double elapsed = h3cspeed_profile_now_seconds() - started;
    pthread_mutex_lock(&gpu->lock);
    if (kind == H3CSPEED_PROFILE_STREAM_COMPUTE) {
        gpu->profile_metrics.compute_stream_syncs++;
        gpu->profile_metrics.compute_stream_wait_seconds += elapsed;
    } else {
        gpu->profile_metrics.upload_stream_syncs++;
        gpu->profile_metrics.upload_stream_wait_seconds += elapsed;
    }
    pthread_mutex_unlock(&gpu->lock);
    return status;
}

static cudaError_t profiled_event_synchronize(h3_gpu *gpu, cudaEvent_t event) {
    if (!gpu || !gpu->profile_enabled) return cudaEventSynchronize(event);
    double started = h3cspeed_profile_now_seconds();
    cudaError_t status = cudaEventSynchronize(event);
    double elapsed = h3cspeed_profile_now_seconds() - started;
    pthread_mutex_lock(&gpu->lock);
    gpu->profile_metrics.event_syncs++;
    gpu->profile_metrics.event_wait_seconds += elapsed;
    pthread_mutex_unlock(&gpu->lock);
    return status;
}

static cudaError_t profiled_h2d_async(h3_gpu *gpu, void *destination,
                                      const void *source, size_t bytes,
                                      cudaStream_t stream) {
    if (!gpu || !gpu->profile_enabled)
        return cudaMemcpyAsync(destination, source, bytes,
                               cudaMemcpyHostToDevice, stream);
    double started = h3cspeed_profile_now_seconds();
    cudaError_t status = cudaMemcpyAsync(destination, source, bytes,
                                         cudaMemcpyHostToDevice, stream);
    profile_update(gpu, profile_h2d_enqueue, (uint64_t)bytes,
                   h3cspeed_profile_now_seconds() - started);
    return status;
}

static void upload_wait_trace_mark_failed(h3_gpu *gpu, const char *operation) {
    if (!gpu) return;
    pthread_mutex_lock(&gpu->lock);
    gpu->upload_wait_trace_complete = 0;
    gpu->upload_wait_trace_union_valid = 0;
    pthread_mutex_unlock(&gpu->lock);
    if (operation) h3cspeed_set_error(gpu, operation, "PERF-006 trace failure");
}

static void upload_wait_trace_clear_events(h3_gpu *gpu) {
    if (!gpu) return;
    for (size_t index = 0; index < H3CSPEED_UPLOAD_WAIT_TRACE_CAPACITY;
         index++) {
        h3cspeed_upload_wait_trace_entry *entry =
            &gpu->upload_wait_trace_entries[index];
        if (entry->start) (void)cudaEventDestroy(entry->start);
        if (entry->end) (void)cudaEventDestroy(entry->end);
        entry->start = nullptr;
        entry->end = nullptr;
        entry->valid = 0;
    }
    gpu->upload_wait_trace_count = 0;
    gpu->upload_wait_trace_initialized = 0;
}

static int upload_wait_trace_init(h3_gpu *gpu) {
    if (!gpu || !gpu->upload_wait_trace_requested) return 1;
    if (gpu->upload_wait_trace_initialized) return 1;
    if (!gpu->upload_wait_trace_complete ||
        !gpu->upload_wait_trace_union_valid) return 0;
    for (size_t index = 0; index < H3CSPEED_UPLOAD_WAIT_TRACE_CAPACITY;
         index++) {
        cudaError_t status = cudaEventCreate(
            &gpu->upload_wait_trace_entries[index].start);
        if (status != cudaSuccess) {
            upload_wait_trace_clear_events(gpu);
            upload_wait_trace_mark_failed(gpu, "cudaEventCreate(upload wait start)");
            return 0;
        }
        status = cudaEventCreate(&gpu->upload_wait_trace_entries[index].end);
        if (status != cudaSuccess) {
            upload_wait_trace_clear_events(gpu);
            upload_wait_trace_mark_failed(gpu, "cudaEventCreate(upload wait end)");
            return 0;
        }
        gpu->upload_wait_trace_entries[index].valid = 0;
    }
    gpu->upload_wait_trace_initialized = 1;
    return 1;
}

static void upload_wait_trace_destroy(h3_gpu *gpu) {
    upload_wait_trace_clear_events(gpu);
}

/* Return 1 when an event pair was armed, 0 when tracing is disabled/outside
 * the DiT scope, and -1 when the opt-in trace failed closed. */
static int upload_wait_trace_begin(h3_gpu *gpu, size_t *slot) {
    if (!gpu || !gpu->upload_wait_trace_requested ||
        !gpu->profile_dit_scope_active) return 0;
    pthread_mutex_lock(&gpu->lock);
    if (!gpu->upload_wait_trace_complete ||
        gpu->upload_wait_trace_count >= H3CSPEED_UPLOAD_WAIT_TRACE_CAPACITY) {
        if (gpu->upload_wait_trace_complete) {
            gpu->upload_wait_trace_overflow = 1;
            gpu->upload_wait_trace_complete = 0;
            gpu->upload_wait_trace_union_valid = 0;
        }
        pthread_mutex_unlock(&gpu->lock);
        return -1;
    }
    size_t index = gpu->upload_wait_trace_count;
    cudaError_t status = cudaEventRecord(
        gpu->upload_wait_trace_entries[index].start, gpu->compute_stream);
    if (status != cudaSuccess) {
        gpu->upload_wait_trace_complete = 0;
        gpu->upload_wait_trace_union_valid = 0;
        pthread_mutex_unlock(&gpu->lock);
        h3cspeed_set_error(gpu, "cudaEventRecord(upload wait start)",
                           cudaGetErrorString(status));
        return -1;
    }
    gpu->upload_wait_trace_entries[index].valid = 1;
    gpu->upload_wait_trace_count++;
    pthread_mutex_unlock(&gpu->lock);
    if (slot) *slot = index;
    return 1;
}

static int upload_wait_trace_end(h3_gpu *gpu, size_t slot) {
    if (!gpu || !gpu->upload_wait_trace_requested ||
        slot >= H3CSPEED_UPLOAD_WAIT_TRACE_CAPACITY) return 1;
    pthread_mutex_lock(&gpu->lock);
    if (!gpu->upload_wait_trace_entries[slot].valid) {
        pthread_mutex_unlock(&gpu->lock);
        return 1;
    }
    cudaError_t status = cudaEventRecord(
        gpu->upload_wait_trace_entries[slot].end, gpu->compute_stream);
    if (status != cudaSuccess) {
        gpu->upload_wait_trace_entries[slot].valid = 0;
        gpu->upload_wait_trace_complete = 0;
        gpu->upload_wait_trace_union_valid = 0;
        pthread_mutex_unlock(&gpu->lock);
        h3cspeed_set_error(gpu, "cudaEventRecord(upload wait end)",
                           cudaGetErrorString(status));
        return 0;
    }
    pthread_mutex_unlock(&gpu->lock);
    return 1;
}

static void upload_wait_trace_resolve(h3_gpu *gpu) {
    if (!gpu || !gpu->upload_wait_trace_requested) return;
    pthread_mutex_lock(&gpu->lock);
    size_t count = gpu->upload_wait_trace_count;
    pthread_mutex_unlock(&gpu->lock);
    for (size_t index = 0; index < count; index++) {
        pthread_mutex_lock(&gpu->lock);
        int valid = gpu->upload_wait_trace_entries[index].valid;
        cudaEvent_t start = gpu->upload_wait_trace_entries[index].start;
        cudaEvent_t end = gpu->upload_wait_trace_entries[index].end;
        pthread_mutex_unlock(&gpu->lock);
        if (!valid) continue;
        float milliseconds = 0.0f;
        cudaError_t status = cudaEventElapsedTime(&milliseconds, start, end);
        if (status != cudaSuccess || !std::isfinite(milliseconds) ||
            milliseconds < 0.0f) {
            upload_wait_trace_mark_failed(gpu,
                                          "cudaEventElapsedTime(upload wait)");
            continue;
        }
        pthread_mutex_lock(&gpu->lock);
        gpu->upload_ready_wait_seconds += (double)milliseconds / 1000.0;
        gpu->upload_ready_wait_count++;
        pthread_mutex_unlock(&gpu->lock);
    }
    pthread_mutex_lock(&gpu->lock);
    for (size_t index = 0; index < count; index++)
        gpu->upload_wait_trace_entries[index].valid = 0;
    gpu->upload_wait_trace_count = 0;
    pthread_mutex_unlock(&gpu->lock);
}

extern "C" void h3cspeed_cuda_profile_dit_scope_begin(
        h3_gpu *gpu, int ssd_streaming, int one_ahead_convrot) {
    if (!gpu) return;
    if (gpu->upload_wait_trace_requested &&
        !gpu->upload_wait_trace_initialized &&
        !upload_wait_trace_init(gpu)) return;
    pthread_mutex_lock(&gpu->lock);
    gpu->profile_dit_scope_seen = 1;
    gpu->profile_dit_scope_active =
        gpu->upload_wait_trace_requested &&
        gpu->upload_wait_trace_initialized ? 1 : 0;
    gpu->profile_dit_ssd_streaming = ssd_streaming ? 1 : 0;
    gpu->dit_prefetch_requested = dit_prefetch_requested();
    snprintf(gpu->dit_prefetch_mode, sizeof(gpu->dit_prefetch_mode), "%s",
             one_ahead_convrot ? "one_ahead_convrot" : "disabled");
    pthread_mutex_unlock(&gpu->lock);
}

extern "C" void h3cspeed_cuda_profile_dit_scope_end(h3_gpu *gpu) {
    if (!gpu) return;
    pthread_mutex_lock(&gpu->lock);
    gpu->profile_dit_scope_active = 0;
    pthread_mutex_unlock(&gpu->lock);
}

static void profile_prefetch_counter(h3_gpu *gpu, int kind) {
    if (!gpu || !gpu->profile_enabled) return;
    pthread_mutex_lock(&gpu->lock);
    uint64_t *counter = nullptr;
    switch (kind) {
        case 0: counter = &gpu->prefetch_reserve_count; break;
        case 1: counter = &gpu->prefetch_upload_count; break;
        case 2: counter = &gpu->prefetch_consume_count; break;
        case 3: counter = &gpu->prefetch_cancel_count; break;
        case 4: counter = &gpu->prefetch_error_count; break;
        case 5: counter = &gpu->prefetch_block_count; break;
        default: break;
    }
    if (counter) (*counter)++;
    pthread_mutex_unlock(&gpu->lock);
}

extern "C" void h3cspeed_cuda_profile_note_prefetch_error(h3_gpu *gpu) {
    profile_prefetch_counter(gpu, 4);
}

extern "C" void h3cspeed_cuda_profile_note_prefetch_block(h3_gpu *gpu) {
    profile_prefetch_counter(gpu, 5);
}

static int read_exact(h3_gpu *gpu, int descriptor, void *buffer, size_t bytes,
                      uint64_t offset, char *error, size_t error_size);
static int convrot_group_valid(uint32_t group_size);
static int refill_trace_init_locked(h3_gpu *gpu);
static int async_refill_active(const h3_gpu *gpu);

static void lru_remove_locked(h3_gpu *gpu, h3_gpu_tensor *tensor) {
    if (!gpu || !tensor || !tensor->in_lru) return;
    if (tensor->lru_previous) tensor->lru_previous->lru_next = tensor->lru_next;
    else gpu->lru_head = tensor->lru_next;
    if (tensor->lru_next) tensor->lru_next->lru_previous = tensor->lru_previous;
    else gpu->lru_tail = tensor->lru_previous;
    tensor->lru_previous = nullptr;
    tensor->lru_next = nullptr;
    tensor->in_lru = 0;
}

static void lru_append_locked(h3_gpu *gpu, h3_gpu_tensor *tensor) {
    if (!gpu || !tensor) return;
    lru_remove_locked(gpu, tensor);
    tensor->lru_previous = gpu->lru_tail;
    tensor->lru_next = nullptr;
    if (gpu->lru_tail) gpu->lru_tail->lru_next = tensor;
    else gpu->lru_head = tensor;
    gpu->lru_tail = tensor;
    tensor->in_lru = 1;
}

static void host_lru_remove_locked(h3_gpu *gpu, h3_gpu_tensor *tensor) {
    if (!gpu || !tensor || !tensor->in_host_lru) return;
    if (tensor->host_lru_previous)
        tensor->host_lru_previous->host_lru_next = tensor->host_lru_next;
    else gpu->host_lru_head = tensor->host_lru_next;
    if (tensor->host_lru_next)
        tensor->host_lru_next->host_lru_previous = tensor->host_lru_previous;
    else gpu->host_lru_tail = tensor->host_lru_previous;
    tensor->host_lru_previous = nullptr;
    tensor->host_lru_next = nullptr;
    tensor->in_host_lru = 0;
}

static void host_lru_append_locked(h3_gpu *gpu, h3_gpu_tensor *tensor) {
    if (!gpu || !tensor || !tensor->host_data) return;
    host_lru_remove_locked(gpu, tensor);
    tensor->host_lru_previous = gpu->host_lru_tail;
    tensor->host_lru_next = nullptr;
    if (gpu->host_lru_tail) gpu->host_lru_tail->host_lru_next = tensor;
    else gpu->host_lru_head = tensor;
    gpu->host_lru_tail = tensor;
    tensor->in_host_lru = 1;
}

static int tensor_event_synchronize(h3_gpu *gpu, h3_gpu_tensor *tensor,
                                    int use_event, const char *operation) {
    if (!gpu || !tensor) return 0;
    pthread_mutex_lock(&tensor->lock);
    int valid = use_event ? tensor->last_use_valid : tensor->ready_valid;
    cudaEvent_t event = use_event ? tensor->last_use : tensor->ready;
    pthread_mutex_unlock(&tensor->lock);
    if (!valid) return 1;
    return h3cspeed_cuda_ok(gpu, profiled_event_synchronize(gpu, event), operation);
}

static void track_device_allocation(h3_gpu *gpu, size_t bytes) {
    gpu->device_live_bytes += bytes;
    pthread_mutex_lock(&gpu->lock);
    gpu->stats.allocated_bytes += bytes;
    gpu->stats.live_bytes += bytes;
    gpu->stats.peak_live_bytes = std::max(gpu->stats.peak_live_bytes,
                                         gpu->stats.live_bytes);
    pthread_mutex_unlock(&gpu->lock);
}

static void track_device_release(h3_gpu *gpu, size_t bytes) {
    gpu->device_live_bytes = gpu->device_live_bytes >= bytes ?
        gpu->device_live_bytes - bytes : 0;
    pthread_mutex_lock(&gpu->lock);
    gpu->stats.live_bytes = gpu->stats.live_bytes >= bytes ?
        gpu->stats.live_bytes - bytes : 0;
    pthread_mutex_unlock(&gpu->lock);
}

static int release_tensor_device_locked(h3_gpu *gpu, h3_gpu_tensor *tensor,
                                        int wait_for_use,
                                        int legacy_count_eviction,
                                        h3cspeed_profile_eviction_reason reason) {
    if (!gpu || !tensor || !tensor->data) return 1;
    double started = gpu->profile_enabled ? h3cspeed_profile_now_seconds() : 0.0;
    if (wait_for_use) {
        if (!tensor_event_synchronize(gpu, tensor, 0,
                                      "wait for weight upload before eviction") ||
            !tensor_event_synchronize(gpu, tensor, 1,
                                      "wait for weight use before eviction"))
            return 0;
    }
    void *pointer = tensor->data;
    /* Do not mutate the residency/LRU bookkeeping until cudaFree succeeds.
     * Keeping the pointer reachable is safer than silently losing it if the
     * CUDA runtime reports an asynchronous failure at this synchronization
     * boundary. */
    if (!h3cspeed_cuda_ok(gpu, cudaFree(pointer), "cudaFree tensor")) return 0;
    tensor->data = nullptr;
    tensor->prefetch_reserved = 0;
    pthread_mutex_lock(&tensor->lock);
    tensor->ready_valid = 0;
    tensor->last_use_valid = 0;
    pthread_mutex_unlock(&tensor->lock);
    lru_remove_locked(gpu, tensor);
    if (tensor->offloadable) {
        gpu->resident_weight_bytes = gpu->resident_weight_bytes >= tensor->bytes ?
            gpu->resident_weight_bytes - tensor->bytes : 0;
        if (legacy_count_eviction) {
            gpu->offload_evictions++;
            gpu->offload_evicted_bytes += tensor->bytes;
        }
    }
    track_device_release(gpu, tensor->bytes);
    if (gpu->profile_enabled) {
        pthread_mutex_lock(&gpu->lock);
        (void)h3cspeed_profile_record_eviction(
            &gpu->profile_metrics, reason, (uint64_t)tensor->bytes,
            h3cspeed_profile_now_seconds() - started);
        pthread_mutex_unlock(&gpu->lock);
    }
    return 1;
}

static h3_gpu_tensor *eviction_candidate_locked(h3_gpu *gpu,
                                                 h3_gpu_tensor *protected_tensor) {
    for (h3_gpu_tensor *candidate = gpu->lru_head; candidate;
         candidate = candidate->lru_next) {
        if (candidate == protected_tensor || !candidate->data ||
            candidate->prefetch_reserved ||
            candidate->pin_epoch == gpu->operation_epoch)
            continue;
        return candidate;
    }
    return nullptr;
}

static int allocation_fits_locked(h3_gpu *gpu, size_t bytes,
                                  int weight_allocation) {
    if (!gpu->offload.enabled) return 1;
    if ((uint64_t)bytes > gpu->offload.vram_budget_bytes) return 0;
    if ((uint64_t)gpu->device_live_bytes + bytes >
        gpu->offload.vram_budget_bytes) return 0;
    if (weight_allocation &&
        (uint64_t)gpu->resident_weight_bytes + bytes >
        gpu->offload.weight_cache_bytes) return 0;
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        const size_t runtime_headroom = 64u * 1024u * 1024u;
        if (free_bytes < bytes || free_bytes - bytes < runtime_headroom) return 0;
    } else {
        (void)cudaGetLastError();
    }
    return 1;
}

static int evict_until_fit_locked(h3_gpu *gpu, size_t bytes,
                                  int weight_allocation,
                                  h3_gpu_tensor *protected_tensor) {
    while (!allocation_fits_locked(gpu, bytes, weight_allocation)) {
        h3_gpu_tensor *candidate = eviction_candidate_locked(gpu, protected_tensor);
        if (!candidate) {
            char detail[320];
            snprintf(detail, sizeof(detail),
                     "request %.2f MiB exceeds the active %.2f MiB VRAM budget "
                     "or %.2f MiB weight cache; lower resolution/frames or "
                     "increase H3_CUDA_VRAM_BUDGET_MIB/H3_CUDA_WEIGHT_CACHE_MIB",
                     (double)bytes / (1024.0 * 1024.0),
                     (double)gpu->offload.vram_budget_bytes / (1024.0 * 1024.0),
                     (double)gpu->offload.weight_cache_bytes / (1024.0 * 1024.0));
            h3cspeed_set_error(gpu, "CUDA offload allocation", detail);
            return 0;
        }
        if (!release_tensor_device_locked(
                gpu, candidate, 1, 1,
                H3CSPEED_PROFILE_EVICTION_CAPACITY_LRU)) return 0;
    }
    return 1;
}

static int trim_offload_cache_locked(h3_gpu *gpu) {
    if (!gpu || !gpu->offload.enabled) return 1;
    while ((uint64_t)gpu->device_live_bytes > gpu->offload.vram_budget_bytes ||
           (uint64_t)gpu->resident_weight_bytes >
               gpu->offload.weight_cache_bytes) {
        h3_gpu_tensor *candidate = eviction_candidate_locked(gpu, nullptr);
        if (!candidate) {
            h3cspeed_set_error(
                gpu, "CUDA offload cache trim",
                "no evictable weight remains; the active operation or "
                "non-offloadable activations exceed the configured VRAM budget");
            return 0;
        }
        if (!release_tensor_device_locked(
                gpu, candidate, 1, 1,
                H3CSPEED_PROFILE_EVICTION_CAPACITY_LRU)) return 0;
    }
    return 1;
}

static int device_allocate_locked(h3_gpu *gpu, void **pointer, size_t bytes,
                                  int weight_allocation,
                                  h3_gpu_tensor *protected_tensor) {
    if (!gpu || !pointer) return 0;
    *pointer = nullptr;
    if (!bytes) return 1;
    if (!evict_until_fit_locked(gpu, bytes, weight_allocation,
                                protected_tensor)) return 0;
    double allocation_started = gpu->profile_enabled ?
        h3cspeed_profile_now_seconds() : 0.0;
    cudaError_t status = cudaMalloc(pointer, bytes);
    if (gpu->profile_enabled)
        profile_update(gpu, profile_allocation, (uint64_t)bytes,
                       h3cspeed_profile_now_seconds() - allocation_started);
    if (status != cudaSuccess && gpu->offload.enabled) {
        (void)cudaGetLastError();
        h3_gpu_tensor *candidate = nullptr;
        while ((candidate = eviction_candidate_locked(gpu, protected_tensor))) {
            if (!release_tensor_device_locked(
                    gpu, candidate, 1, 1,
                    H3CSPEED_PROFILE_EVICTION_CAPACITY_LRU)) return 0;
        }
        allocation_started = gpu->profile_enabled ?
            h3cspeed_profile_now_seconds() : 0.0;
        status = cudaMalloc(pointer, bytes);
        if (gpu->profile_enabled)
            profile_update(gpu, profile_allocation, (uint64_t)bytes,
                           h3cspeed_profile_now_seconds() - allocation_started);
    }
    if (!h3cspeed_cuda_ok(gpu, status, "cudaMalloc tensor")) return 0;
    track_device_allocation(gpu, bytes);
    if (weight_allocation) {
        gpu->resident_weight_bytes += bytes;
        gpu->peak_resident_weight_bytes = std::max(
            gpu->peak_resident_weight_bytes, gpu->resident_weight_bytes);
    }
    return 1;
}

static void host_backing_release_locked(h3_gpu *gpu, h3_gpu_tensor *tensor) {
    if (!gpu || !tensor || !tensor->host_data) return;
    host_lru_remove_locked(gpu, tensor);
    if (tensor->host_pinned) (void)cudaFreeHost(tensor->host_data);
#if defined(_WIN32)
    else _aligned_free(tensor->host_data);
#else
    else free(tensor->host_data);
#endif
    gpu->host_cache_live_bytes = gpu->host_cache_live_bytes >= tensor->bytes ?
        gpu->host_cache_live_bytes - tensor->bytes : 0;
    if (tensor->host_pinned)
        gpu->pinned_host_live_bytes =
            gpu->pinned_host_live_bytes >= tensor->bytes ?
            gpu->pinned_host_live_bytes - tensor->bytes : 0;
    tensor->host_data = nullptr;
    tensor->host_pinned = 0;
    tensor->host_valid = 0;
}

static h3_gpu_tensor *host_cache_candidate_locked(
        h3_gpu *gpu, h3_gpu_tensor *protected_tensor) {
    for (h3_gpu_tensor *candidate = gpu->host_lru_head; candidate;
         candidate = candidate->host_lru_next) {
        /* Generated INT8 weights have no disk source and therefore cannot be
         * dropped from RAM. File-backed BF16/F32 copies can always be reread. */
        if (candidate == protected_tensor || !candidate->host_data ||
            !candidate->source_path) continue;
        return candidate;
    }
    return nullptr;
}

static int host_cache_make_room_locked(h3_gpu *gpu, size_t bytes,
                                       h3_gpu_tensor *protected_tensor) {
    if (!gpu || !gpu->offload.host_cache_bytes) return 0;
    while ((uint64_t)gpu->host_cache_live_bytes + bytes >
           gpu->offload.host_cache_bytes) {
        h3_gpu_tensor *candidate = host_cache_candidate_locked(
            gpu, protected_tensor);
        if (!candidate) return 0;
        /* A pinned host copy may still be the source of an asynchronous DMA. */
        if (!tensor_event_synchronize(gpu, candidate, 0,
                                      "wait for host-cache DMA before eviction"))
            return 0;
        gpu->host_cache_evictions++;
        gpu->host_cache_evicted_bytes += candidate->bytes;
        host_backing_release_locked(gpu, candidate);
    }
    return 1;
}

static int host_backing_allocate_locked(h3_gpu *gpu, h3_gpu_tensor *tensor) {
    if (!gpu || !tensor || !tensor->bytes || !gpu->offload.enabled ||
        !gpu->offload.host_cache_bytes) return 0;
    if (tensor->host_data) {
        host_lru_append_locked(gpu, tensor);
        return 1;
    }
    if (!host_cache_make_room_locked(gpu, tensor->bytes, tensor)) return 0;

    void *memory = nullptr;
    int pinned = 0;
    if ((uint64_t)gpu->pinned_host_live_bytes + tensor->bytes <=
        gpu->offload.pinned_host_bytes) {
        cudaError_t status = cudaHostAlloc(&memory, tensor->bytes,
                                           cudaHostAllocPortable);
        if (status == cudaSuccess) pinned = 1;
        else {
            memory = nullptr;
            (void)cudaGetLastError();
        }
    }
    if (!memory) {
        if (posix_memalign(&memory, 4096, tensor->bytes) != 0) memory = nullptr;
    }
    if (!memory) return 0;

    tensor->host_data = memory;
    tensor->host_pinned = pinned;
    gpu->host_cache_live_bytes += tensor->bytes;
    if (pinned) gpu->pinned_host_live_bytes += tensor->bytes;
    gpu->peak_host_cache_bytes = std::max(gpu->peak_host_cache_bytes,
                                         gpu->host_cache_live_bytes);
    host_lru_append_locked(gpu, tensor);
    return 1;
}

static int staging_allocate_locked(h3_gpu *gpu) {
    if (gpu->refill_trace_requested && gpu->refill_trace_failed) return 0;
    if (gpu->staging) return 1;
    size_t requested = (size_t)gpu->offload.staging_bytes;
    if (!requested) requested = 64u * 1024u * 1024u;
    cudaError_t status = cudaHostAlloc(&gpu->staging, requested,
                                       cudaHostAllocPortable);
    if (status == cudaSuccess) {
        gpu->staging_pinned = 1;
    } else {
        (void)cudaGetLastError();
        gpu->staging = malloc(requested);
        gpu->staging_pinned = 0;
    }
    if (!gpu->staging) {
        h3cspeed_set_error(gpu, "offload staging allocation", "out of host memory");
        return 0;
    }
    gpu->staging_bytes = requested;
    gpu->staging_slot_bytes = 0;
    gpu->staging_slots[0] = nullptr;
    gpu->staging_slots[1] = nullptr;
    gpu->staging_done[0] = nullptr;
    gpu->staging_done[1] = nullptr;
    gpu->staging_done_valid[0] = 0;
    gpu->staging_done_valid[1] = 0;
    /* The opt-in path is deliberately restricted to pinned memory.  A
     * pageable fallback cannot safely overlap CPU refills with DMA. */
    if (gpu->async_refill_enabled && gpu->staging_pinned && requested >= 2) {
        size_t slot_bytes = requested / 2;
        if (slot_bytes) {
            cudaError_t first = cudaEventCreateWithFlags(
                &gpu->staging_done[0], cudaEventDisableTiming);
            cudaError_t second = first == cudaSuccess ?
                cudaEventCreateWithFlags(&gpu->staging_done[1],
                                         cudaEventDisableTiming) : first;
            if (first == cudaSuccess && second == cudaSuccess) {
                gpu->staging_slots[0] = gpu->staging;
                gpu->staging_slots[1] = static_cast<unsigned char *>(
                    gpu->staging) + slot_bytes;
                gpu->staging_slot_bytes = slot_bytes;
            } else {
                if (gpu->staging_done[0]) {
                    (void)cudaEventDestroy(gpu->staging_done[0]);
                    gpu->staging_done[0] = nullptr;
                }
                if (gpu->staging_done[1]) {
                    (void)cudaEventDestroy(gpu->staging_done[1]);
                    gpu->staging_done[1] = nullptr;
                }
                gpu->async_refill_enabled = 0;
                (void)cudaGetLastError();
            }
        }
    }
    if (gpu->refill_trace_requested && async_refill_active(gpu) &&
        !refill_trace_init_locked(gpu)) {
        /* The caller will drain the upload stream before returning the
         * failure.  Keep the explicit opt-in fail-closed rather than silently
         * running a partially instrumented refill. */
        return 0;
    }
    return 1;
}

static int async_refill_active(const h3_gpu *gpu) {
    return gpu && gpu->async_refill_enabled && gpu->staging_slot_bytes &&
           gpu->staging_slots[0] && gpu->staging_slots[1] &&
           gpu->staging_done[0] && gpu->staging_done[1];
}

static void refill_trace_destroy(h3_gpu *gpu) {
    if (!gpu) return;
    for (size_t index = 0; index < H3CSPEED_REFILL_TRACE_CAPACITY; index++) {
        h3cspeed_refill_trace_entry *entry = &gpu->refill_trace_entries[index];
        if (entry->h2d_start) {
            (void)cudaEventDestroy(entry->h2d_start);
            entry->h2d_start = nullptr;
        }
        if (entry->h2d_end) {
            (void)cudaEventDestroy(entry->h2d_end);
            entry->h2d_end = nullptr;
        }
        entry->valid = 0;
    }
    gpu->refill_trace_enabled = 0;
}

static int refill_trace_init_locked(h3_gpu *gpu) {
    if (!gpu || !gpu->refill_trace_requested ||
        !async_refill_active(gpu)) return 1;
    if (gpu->refill_trace_enabled) return 1;
    for (size_t index = 0; index < H3CSPEED_REFILL_TRACE_CAPACITY; index++) {
        h3cspeed_refill_trace_entry *entry = &gpu->refill_trace_entries[index];
        cudaError_t status = cudaEventCreate(&entry->h2d_start);
        if (status == cudaSuccess)
            status = cudaEventCreate(&entry->h2d_end);
        if (status != cudaSuccess) {
            (void)cudaGetLastError();
            refill_trace_destroy(gpu);
            gpu->refill_trace_failed = 1;
            h3cspeed_set_error(gpu, "CUDA refill trace event creation",
                               cudaGetErrorString(status));
            return 0;
        }
    }
    gpu->refill_trace_enabled = 1;
    return 1;
}

static h3cspeed_refill_trace_entry *refill_trace_begin(
        h3_gpu *gpu, uint64_t refill_id, size_t slot, uint64_t chunk_index,
        uint32_t source_kind, size_t bytes) {
    if (!gpu || !gpu->refill_trace_enabled || slot >= 2) return nullptr;
    const size_t index = (size_t)(gpu->refill_trace_next_sequence %
                                  H3CSPEED_REFILL_TRACE_CAPACITY);
    h3cspeed_refill_trace_entry *entry = &gpu->refill_trace_entries[index];
    /* A bounded ring may eventually recycle an event pair.  Drain only the
     * old pair at wrap-around so recording cannot race a prior DMA interval. */
    if (entry->valid) {
        cudaError_t status = cudaEventSynchronize(entry->h2d_end);
        if (status != cudaSuccess) {
            gpu->refill_trace_failed = 1;
            h3cspeed_set_error(gpu, "CUDA refill trace event synchronize",
                               cudaGetErrorString(status));
            return nullptr;
        }
    }
    const uint64_t sequence = ++gpu->refill_trace_next_sequence;
    entry->sequence = sequence;
    entry->refill_id = refill_id;
    entry->reuse_after_sequence = gpu->refill_trace_last_slot_sequence[slot];
    entry->chunk_index = chunk_index;
    entry->bytes = bytes;
    entry->slot = (uint32_t)slot;
    entry->source_kind = source_kind;
    entry->host_read_start_ns = refill_trace_now_ns();
    entry->host_read_end_ns = 0;
    entry->valid = 0;
    gpu->refill_trace_last_slot_sequence[slot] = sequence;
    return entry;
}

static int refill_trace_record_start(h3_gpu *gpu,
                                      h3cspeed_refill_trace_entry *entry) {
    if (!gpu || !entry || !gpu->refill_trace_enabled) return 1;
    cudaError_t status = cudaEventRecord(entry->h2d_start,
                                         gpu->upload_stream);
    if (status != cudaSuccess) {
        gpu->refill_trace_failed = 1;
        h3cspeed_set_error(gpu, "CUDA refill trace H2D start",
                           cudaGetErrorString(status));
        return 0;
    }
    return 1;
}

static int refill_trace_record_end(h3_gpu *gpu,
                                    h3cspeed_refill_trace_entry *entry) {
    if (!gpu || !entry || !gpu->refill_trace_enabled) return 1;
    cudaError_t status = cudaEventRecord(entry->h2d_end, gpu->upload_stream);
    if (status != cudaSuccess) {
        gpu->refill_trace_failed = 1;
        h3cspeed_set_error(gpu, "CUDA refill trace H2D end",
                           cudaGetErrorString(status));
        return 0;
    }
    entry->valid = 1;
    return 1;
}

static int drain_upload_stream(h3_gpu *gpu, const char *operation) {
    if (!gpu || !gpu->upload_stream) return 1;
    return h3cspeed_cuda_ok(
        gpu,
        profiled_stream_synchronize(gpu, gpu->upload_stream,
                                    H3CSPEED_PROFILE_STREAM_UPLOAD),
        operation ? operation : "drain upload stream after staging failure");
}

static int staging_slot_wait_locked(h3_gpu *gpu, size_t slot) {
    if (!async_refill_active(gpu) || slot >= 2 ||
        !gpu->staging_done_valid[slot]) return 1;
    return h3cspeed_cuda_ok(
        gpu, profiled_event_synchronize(gpu, gpu->staging_done[slot]),
        "wait for staging slot DMA before refill");
}

static int staging_slot_record_locked(h3_gpu *gpu, size_t slot) {
    if (!async_refill_active(gpu) || slot >= 2) return 1;
    cudaError_t status = cudaEventRecord(gpu->staging_done[slot],
                                         gpu->upload_stream);
    if (status == cudaSuccess) gpu->staging_done_valid[slot] = 1;
    return h3cspeed_cuda_ok(gpu, status, "record staging slot DMA");
}

static int upload_weight_locked(h3_gpu *gpu, h3_gpu_tensor *tensor) {
    if (!gpu || !tensor || !tensor->data || !tensor->source_bytes) return 0;
    uint64_t trace_refill_id = 0;
    if (tensor->host_data && tensor->host_valid)
        host_lru_append_locked(gpu, tensor);
    if (tensor->host_data && tensor->host_valid && tensor->host_pinned) {
        if (!h3cspeed_cuda_ok(gpu,
            profiled_h2d_async(gpu, tensor->data, tensor->host_data,
                               tensor->source_bytes, gpu->upload_stream),
            "upload pinned host weight")) {
            (void)drain_upload_stream(gpu,
                                       "drain upload stream after pinned upload failure");
            return 0;
        }
    } else {
        pthread_mutex_lock(&gpu->staging_lock);
        if (!staging_allocate_locked(gpu)) {
            (void)drain_upload_stream(gpu,
                                      "drain upload stream after refill trace failure");
            pthread_mutex_unlock(&gpu->staging_lock);
            return 0;
        }
        if (gpu->refill_trace_enabled)
            trace_refill_id = ++gpu->refill_trace_next_refill_id;
        int descriptor = -1;
        if (!tensor->host_data || !tensor->host_valid) {
            descriptor = open(tensor->source_path, O_RDONLY | O_CLOEXEC);
            if (descriptor < 0) {
                char detail[384];
                snprintf(detail, sizeof(detail), "cannot open %s: %s",
                         tensor->source_path ? tensor->source_path : "(null)",
                         strerror(errno));
                h3cspeed_set_error(gpu, "file-backed weight upload", detail);
                (void)drain_upload_stream(gpu,
                                           "drain upload stream after file open failure");
                pthread_mutex_unlock(&gpu->staging_lock);
                return 0;
            }
#ifdef POSIX_FADV_NOREUSE
            if (tensor->source_streaming)
                (void)posix_fadvise(descriptor,
                                    (off_t)tensor->source_offset,
                                    (off_t)tensor->source_bytes,
                                    POSIX_FADV_NOREUSE);
#endif
        }
        size_t done = 0;
        int ok = 1;
        while (done < tensor->source_bytes) {
            const int use_async_refill = async_refill_active(gpu);
            const size_t chunk_index = gpu->staging_slot_bytes ?
                done / gpu->staging_slot_bytes : 0;
            const size_t slot = chunk_index & 1u;
            if (use_async_refill && !staging_slot_wait_locked(gpu, slot)) {
                ok = 0;
                break;
            }
            void *staging = use_async_refill ? gpu->staging_slots[slot] :
                gpu->staging;
            const size_t staging_capacity = use_async_refill ?
                gpu->staging_slot_bytes : gpu->staging_bytes;
            size_t chunk = std::min(tensor->source_bytes - done,
                                    staging_capacity);
            h3cspeed_refill_trace_entry *trace_entry =
                use_async_refill && gpu->refill_trace_enabled ?
                refill_trace_begin(gpu, trace_refill_id, slot, chunk_index,
                                   descriptor >= 0 ? 1u : 2u, chunk) : nullptr;
            if (gpu->refill_trace_enabled && !trace_entry) {
                ok = 0;
                break;
            }
            if (descriptor >= 0) {
                char read_error[256] = {0};
                if (!read_exact(gpu, descriptor, staging, chunk,
                                tensor->source_offset + done,
                                read_error, sizeof(read_error))) {
                    h3cspeed_set_error(gpu, "file-backed weight read", read_error);
                    ok = 0;
                    break;
                }
            } else {
                double copy_started = gpu->profile_enabled ?
                    h3cspeed_profile_now_seconds() : 0.0;
                memcpy(staging,
                       static_cast<const unsigned char *>(tensor->host_data) + done,
                       chunk);
                if (gpu->profile_enabled)
                    profile_update(gpu, profile_pageable_copy, (uint64_t)chunk,
                                   h3cspeed_profile_now_seconds() - copy_started);
            }
            if (trace_entry) trace_entry->host_read_end_ns = refill_trace_now_ns();
            if ((trace_entry && !refill_trace_record_start(gpu, trace_entry)) ||
                !h3cspeed_cuda_ok(gpu,
                    profiled_h2d_async(
                        gpu, static_cast<unsigned char *>(tensor->data) + done,
                        staging, chunk, gpu->upload_stream),
                    "staged host weight upload") ||
                (trace_entry && !refill_trace_record_end(gpu, trace_entry)) ||
                (use_async_refill && !staging_slot_record_locked(gpu, slot))) {
                ok = 0;
                break;
            }
            if (!use_async_refill &&
                !h3cspeed_cuda_ok(gpu,
                    profiled_stream_synchronize(
                        gpu, gpu->upload_stream,
                        H3CSPEED_PROFILE_STREAM_UPLOAD),
                    "staged host weight synchronization")) {
                ok = 0;
                break;
            }
            done += chunk;
        }
        if (descriptor >= 0) close(descriptor);
        if (!ok) (void)drain_upload_stream(gpu,
                                           "drain upload stream after staging failure");
        pthread_mutex_unlock(&gpu->staging_lock);
        if (!ok) return 0;
        if (!tensor->host_data || !tensor->host_valid) {
            gpu->file_fallback_reads++;
            gpu->file_fallback_bytes += tensor->source_bytes;
        }
    }
    if (tensor->source_bytes < tensor->bytes &&
        !h3cspeed_cuda_ok(gpu,
            cudaMemsetAsync(static_cast<unsigned char *>(tensor->data) +
                                tensor->source_bytes,
                            0, tensor->bytes - tensor->source_bytes,
                            gpu->upload_stream),
            "zero unused weight slot tail")) {
        (void)drain_upload_stream(gpu, "drain upload stream after staging failure");
        return 0;
    }
    if (!h3cspeed_tensor_record_upload(tensor)) {
        (void)drain_upload_stream(gpu, "drain upload stream after upload event failure");
        return 0;
    }
    gpu->offload_uploads++;
    gpu->offload_upload_bytes += tensor->source_bytes;
    return 1;
}

int h3cspeed_tensor_prepare(h3_gpu *gpu, h3_gpu_tensor *tensor) {
    if (!gpu || !tensor || tensor->gpu != gpu) {
        if (gpu) h3cspeed_set_error(gpu, "tensor prepare", "invalid CUDA tensor");
        return 0;
    }
    int consumed_prefetch = 0;
    pthread_mutex_lock(&gpu->offload_lock);
    if (!tensor->data) {
        if (!tensor->offloadable || !tensor->source_bytes ||
            ((!tensor->host_data || !tensor->host_valid) &&
             !tensor->source_path)) {
            pthread_mutex_unlock(&gpu->offload_lock);
            h3cspeed_set_error(
                gpu, "tensor prepare",
                "tensor has no device allocation or valid RAM/file offload source");
            return 0;
        }
        if (!device_allocate_locked(gpu, &tensor->data, tensor->bytes, 1,
                                    tensor) ||
            !upload_weight_locked(gpu, tensor)) {
            if (tensor->data)
                (void)release_tensor_device_locked(
                    gpu, tensor, 0, 0,
                    H3CSPEED_PROFILE_EVICTION_ERROR_CLEANUP);
            pthread_mutex_unlock(&gpu->offload_lock);
            return 0;
        }
        lru_append_locked(gpu, tensor);
    } else if (tensor->offloadable) {
        lru_append_locked(gpu, tensor);
    }
    if (tensor->offloadable) {
        consumed_prefetch = tensor->prefetch_reserved;
        tensor->prefetch_reserved = 0;
        tensor->pin_epoch = gpu->operation_epoch;
    }
    pthread_mutex_unlock(&gpu->offload_lock);
    if (consumed_prefetch) profile_prefetch_counter(gpu, 2);

    pthread_mutex_lock(&tensor->lock);
    int ready_valid = tensor->ready_valid;
    cudaEvent_t ready = tensor->ready;
    pthread_mutex_unlock(&tensor->lock);
    if (!ready_valid) return 1;
    size_t trace_slot = 0;
    int trace_armed = upload_wait_trace_begin(gpu, &trace_slot);
    cudaError_t wait_status = cudaStreamWaitEvent(gpu->compute_stream, ready, 0);
    if (trace_armed > 0) (void)upload_wait_trace_end(gpu, trace_slot);
    return h3cspeed_cuda_ok(gpu, wait_status,
                            "cudaStreamWaitEvent(weight ready)");
}

static int dit_prefetch_requested(void) {
    const char *value = getenv("H3_CUDA_DIT_PREFETCH");
    return value && strcmp(value, "1") == 0;
}

extern "C" int h3cspeed_cuda_reserve_prefetch_weight(
        h3_gpu *gpu, h3_gpu_tensor *tensor) {
    if (!gpu || !tensor || tensor->gpu != gpu) {
        if (gpu) h3cspeed_set_error(gpu, "DiT weight prefetch reserve",
                                    "invalid CUDA tensor");
        return 0;
    }
    if (!dit_prefetch_requested() || !gpu->offload.enabled) return 1;

    pthread_mutex_lock(&gpu->offload_lock);
    if (!tensor->offloadable || !tensor->source_bytes ||
        ((!tensor->host_data || !tensor->host_valid) &&
         !tensor->source_path)) {
        pthread_mutex_unlock(&gpu->offload_lock);
        h3cspeed_set_error(
            gpu, "DiT weight prefetch reserve",
            "tensor has no valid RAM/file reconstruction source");
        return 0;
    }
    if (tensor->pin_epoch == gpu->operation_epoch) {
        pthread_mutex_unlock(&gpu->offload_lock);
        h3cspeed_set_error(
            gpu, "DiT weight prefetch reserve",
            "future tensor is pinned by the current operation");
        return 0;
    }
    if (tensor->data) {
        tensor->prefetch_reserved = 1;
        lru_append_locked(gpu, tensor);
        pthread_mutex_unlock(&gpu->offload_lock);
        profile_prefetch_counter(gpu, 0);
        return 1;
    }
    if (!device_allocate_locked(gpu, &tensor->data, tensor->bytes, 1,
                                tensor)) {
        pthread_mutex_unlock(&gpu->offload_lock);
        return 0;
    }
    pthread_mutex_lock(&tensor->lock);
    tensor->ready_valid = 0;
    tensor->last_use_valid = 0;
    pthread_mutex_unlock(&tensor->lock);
    /* Do not set pin_epoch: the scheduler has not consumed this future
     * weight.  The later upload helper records the existing ready event. */
    tensor->prefetch_reserved = 1;
    lru_append_locked(gpu, tensor);
    pthread_mutex_unlock(&gpu->offload_lock);
    profile_prefetch_counter(gpu, 0);
    return 1;
}

extern "C" int h3cspeed_cuda_prefetch_weight(h3_gpu *gpu,
                                                h3_gpu_tensor *tensor) {
    if (!gpu || !tensor || tensor->gpu != gpu) {
        if (gpu) h3cspeed_set_error(gpu, "DiT weight prefetch",
                                    "invalid CUDA tensor");
        return 0;
    }
    /* Keep the existing synchronous path byte-for-byte opt-in.  The caller
     * still fences its command chain after a disabled/no-offload no-op. */
    if (!dit_prefetch_requested() || !gpu->offload.enabled) return 1;

    pthread_mutex_lock(&gpu->offload_lock);
    if (!tensor->offloadable || !tensor->source_bytes ||
        ((!tensor->host_data || !tensor->host_valid) &&
         !tensor->source_path)) {
        pthread_mutex_unlock(&gpu->offload_lock);
        h3cspeed_set_error(
            gpu, "DiT weight prefetch",
            "tensor has no device allocation or valid RAM/file offload source");
        return 0;
    }
    if (tensor->data && tensor->pin_epoch == gpu->operation_epoch) {
        pthread_mutex_unlock(&gpu->offload_lock);
        h3cspeed_set_error(
            gpu, "DiT weight prefetch",
            "future tensor is pinned by the current operation");
        return 0;
    }

    pthread_mutex_lock(&tensor->lock);
    int ready_valid = tensor->ready_valid;
    pthread_mutex_unlock(&tensor->lock);
    if (tensor->data && ready_valid) {
        tensor->prefetch_reserved = 1;
        lru_append_locked(gpu, tensor);
        pthread_mutex_unlock(&gpu->offload_lock);
        profile_prefetch_counter(gpu, 1);
        return 1;
    }

    /* A stream-slot refill normally releases its old allocation before this
     * helper is called.  The event waits keep this helper safe for a direct
     * caller as well: no host or device storage is overwritten while a prior
     * upload or compute use is still in flight. */
    if (tensor->data &&
        (!tensor_event_synchronize(gpu, tensor, 0,
                                   "wait for future weight upload") ||
         !tensor_event_synchronize(gpu, tensor, 1,
                                   "wait for future weight use"))) {
        pthread_mutex_unlock(&gpu->offload_lock);
        return 0;
    }

    if (!tensor->data) {
        if (!device_allocate_locked(gpu, &tensor->data, tensor->bytes, 1,
                                    tensor)) {
            pthread_mutex_unlock(&gpu->offload_lock);
            return 0;
        }
    }
    if (!upload_weight_locked(gpu, tensor)) {
        if (tensor->data)
            (void)release_tensor_device_locked(
                gpu, tensor, 0, 0,
                H3CSPEED_PROFILE_EVICTION_ERROR_CLEANUP);
        pthread_mutex_unlock(&gpu->offload_lock);
        return 0;
    }
    /* Deliberately do not set pin_epoch: this is one-ahead future work. The
     * scheduler reservation protects it until consume or explicit cancel. */
    tensor->prefetch_reserved = 1;
    lru_append_locked(gpu, tensor);
    pthread_mutex_unlock(&gpu->offload_lock);
    profile_prefetch_counter(gpu, 1);
    return 1;
}

extern "C" void h3cspeed_cuda_cancel_prefetch_weight(
        h3_gpu *gpu, h3_gpu_tensor *tensor) {
    if (!gpu || !tensor || tensor->gpu != gpu) return;
    pthread_mutex_lock(&gpu->offload_lock);
    tensor->prefetch_reserved = 0;
    pthread_mutex_unlock(&gpu->offload_lock);
    profile_prefetch_counter(gpu, 3);
}

extern "C" int h3cspeed_cuda_prime_prefetch_weight(
        h3_gpu *gpu, h3_gpu_tensor *tensor) {
    if (!gpu || !tensor || tensor->gpu != gpu) {
        if (gpu) h3cspeed_set_error(gpu, "DiT weight prefetch prime",
                                    "invalid CUDA tensor");
        return 0;
    }
    if (!dit_prefetch_requested() || !gpu->offload.enabled) return 1;

    pthread_mutex_lock(&gpu->offload_lock);
    pthread_mutex_lock(&tensor->lock);
    int ready_valid = tensor->ready_valid;
    pthread_mutex_unlock(&tensor->lock);
    if (tensor->data && ready_valid &&
        tensor->pin_epoch == gpu->operation_epoch) {
        tensor->prefetch_reserved = 1;
        pthread_mutex_unlock(&gpu->offload_lock);
        profile_prefetch_counter(gpu, 0);
        return 1;
    }
    pthread_mutex_unlock(&gpu->offload_lock);
    return h3cspeed_cuda_reserve_prefetch_weight(gpu, tensor) &&
           h3cspeed_cuda_prefetch_weight(gpu, tensor);
}

void h3cspeed_operation_complete(h3_gpu *gpu) {
    if (!gpu || !gpu->offload.enabled) return;
    pthread_mutex_lock(&gpu->offload_lock);
    uint64_t completed_epoch = gpu->operation_epoch;
    int all_recorded = 1;
    uint64_t recorded_count = 0;
    for (h3_gpu_tensor *tensor = gpu->lru_head; tensor;
         tensor = tensor->lru_next) {
        if (!tensor->data || tensor->pin_epoch != completed_epoch) continue;
        pthread_mutex_lock(&tensor->lock);
        cudaError_t status = cudaEventRecord(tensor->last_use,
                                              gpu->compute_stream);
        if (status == cudaSuccess) {
            tensor->last_use_valid = 1;
            recorded_count++;
        }
        pthread_mutex_unlock(&tensor->lock);
        if (status != cudaSuccess) {
            all_recorded = 0;
            h3cspeed_set_error(gpu, "record offloaded weight use",
                               cudaGetErrorString(status));
        }
    }
    /* If an event could not be recorded, keep the epoch pinned. Advancing it
     * would make a still-in-use weight eligible for eviction without a valid
     * completion fence. A later successful completion can safely recover. */
    if (all_recorded) {
        gpu->operation_epoch++;
        if (!gpu->operation_epoch) gpu->operation_epoch = 1;
    }
    pthread_mutex_unlock(&gpu->offload_lock);
    if (gpu->profile_enabled && recorded_count) {
        pthread_mutex_lock(&gpu->lock);
        gpu->profile_metrics.last_use_fence_count += recorded_count;
        pthread_mutex_unlock(&gpu->lock);
    }
}

static h3_gpu_tensor *tensor_new_internal(h3_gpu *gpu, h3_gpu_dtype dtype,
                                          size_t elements,
                                          int allocate_device) {
    if (!gpu || !h3cspeed_dtype_size(dtype) ||
        elements > SIZE_MAX / h3cspeed_dtype_size(dtype)) return nullptr;
    h3_gpu_tensor *tensor = static_cast<h3_gpu_tensor *>(calloc(1, sizeof(*tensor)));
    if (!tensor) {
        h3cspeed_set_error(gpu, "tensor allocation", "out of host memory");
        return nullptr;
    }
    tensor->gpu = gpu;
    tensor->dtype = dtype;
    tensor->elements = elements;
    tensor->bytes = elements * h3cspeed_dtype_size(dtype);
    pthread_mutex_init(&tensor->lock, nullptr);
    if (!h3cspeed_cuda_ok(gpu,
            cudaEventCreateWithFlags(&tensor->ready, cudaEventDisableTiming),
            "cudaEventCreate(ready)") ||
        !h3cspeed_cuda_ok(gpu,
            cudaEventCreateWithFlags(&tensor->last_use, cudaEventDisableTiming),
            "cudaEventCreate(last use)")) {
        if (tensor->ready) cudaEventDestroy(tensor->ready);
        if (tensor->last_use) cudaEventDestroy(tensor->last_use);
        pthread_mutex_destroy(&tensor->lock);
        free(tensor);
        return nullptr;
    }
    if (allocate_device && tensor->bytes) {
        pthread_mutex_lock(&gpu->offload_lock);
        int ok = device_allocate_locked(gpu, &tensor->data, tensor->bytes, 0,
                                        nullptr);
        pthread_mutex_unlock(&gpu->offload_lock);
        if (!ok) {
            cudaEventDestroy(tensor->ready);
            cudaEventDestroy(tensor->last_use);
            pthread_mutex_destroy(&tensor->lock);
            free(tensor);
            return nullptr;
        }
    }
    pthread_mutex_lock(&gpu->lock);
    gpu->stats.tensor_allocations++;
    pthread_mutex_unlock(&gpu->lock);
    return tensor;
}

static h3_gpu_tensor *tensor_new(h3_gpu *gpu, h3_gpu_dtype dtype,
                                 size_t elements) {
    return tensor_new_internal(gpu, dtype, elements, 1);
}

extern "C" h3_gpu *h3_gpu_create(const char *shader_source_path,
                                    char *error, size_t error_size) {
    (void)shader_source_path;
    if (error && error_size) error[0] = '\0';
    h3_gpu *gpu = static_cast<h3_gpu *>(calloc(1, sizeof(*gpu)));
    if (!gpu) {
        if (error && error_size) snprintf(error, error_size, "out of host memory");
        return nullptr;
    }
    pthread_mutex_init(&gpu->lock, nullptr);
    pthread_mutex_init(&gpu->scratch_lock, nullptr);
    pthread_mutex_init(&gpu->offload_lock, nullptr);
    pthread_mutex_init(&gpu->staging_lock, nullptr);
    gpu->async_refill_requested = async_refill_requested();
    gpu->async_refill_enabled = gpu->async_refill_requested;
    gpu->refill_trace_requested = refill_trace_requested();
    const char *profile_json_dir = getenv("H3_PROFILE_JSON_DIR");
    gpu->profile_enabled = environment_flag_enabled("H3_PROFILE") ||
                           (profile_json_dir && *profile_json_dir);
    gpu->profile_context_id = next_profile_context_id();
    gpu->profile_context_start_seconds = h3cspeed_profile_now_seconds();
    h3cspeed_profile_metrics_init(&gpu->profile_metrics);
    gpu->upload_wait_trace_requested =
        getenv("H3_CUDA_UPLOAD_WAIT_TRACE") &&
        strcmp(getenv("H3_CUDA_UPLOAD_WAIT_TRACE"), "1") == 0;
    gpu->upload_wait_trace_complete = gpu->upload_wait_trace_requested ? 1 : 0;
    gpu->upload_wait_trace_union_valid = gpu->upload_wait_trace_requested ? 1 : 0;
    gpu->dit_prefetch_requested = dit_prefetch_requested();
    snprintf(gpu->dit_prefetch_mode, sizeof(gpu->dit_prefetch_mode),
             "%s", "disabled");
    if (profile_json_dir && strlen(profile_json_dir) >=
                                sizeof(gpu->profile_json_dir)) {
        h3cspeed_set_error(gpu, "H3_PROFILE_JSON_DIR", "path is too long");
        if (error && error_size) snprintf(error, error_size, "%s", gpu->error);
        h3_gpu_free(gpu);
        return nullptr;
    }
    if (profile_json_dir && *profile_json_dir)
        snprintf(gpu->profile_json_dir, sizeof(gpu->profile_json_dir), "%s",
                 profile_json_dir);
    gpu->operation_epoch = 1;
    const char *device_env = getenv("H3_CUDA_DEVICE");
    gpu->device = 0;
    if (device_env && *device_env) {
        char *end = nullptr;
        errno = 0;
        long parsed = strtol(device_env, &end, 10);
        if (errno || end == device_env || *end || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
            h3cspeed_set_error(gpu, "H3_CUDA_DEVICE",
                               "expected a non-negative CUDA device index");
            if (error && error_size) snprintf(error, error_size, "%s", gpu->error);
            h3_gpu_free(gpu);
            return nullptr;
        }
        gpu->device = (int)parsed;
    }
    if (!h3cspeed_cuda_ok(gpu, cudaSetDevice(gpu->device), "cudaSetDevice") ||
        !h3cspeed_cuda_ok(gpu,
            cudaGetDeviceProperties(&gpu->properties, gpu->device),
            "cudaGetDeviceProperties")) {
        if (error && error_size) snprintf(error, error_size, "%s", gpu->error);
        h3_gpu_free(gpu);
        return nullptr;
    }
    if (gpu->properties.major < 8) {
        h3cspeed_set_error(
            gpu, "unsupported CUDA device",
            "MiniMax-H3 BF16 execution requires compute capability 8.0 or newer");
        if (error && error_size) snprintf(error, error_size, "%s", gpu->error);
        h3_gpu_free(gpu);
        return nullptr;
    }
    size_t free_vram = 0, total_vram = gpu->properties.totalGlobalMem;
    if (cudaMemGetInfo(&free_vram, &total_vram) != cudaSuccess) {
        (void)cudaGetLastError();
        free_vram = total_vram;
    }
    char policy_error[256] = {0};
    if (!h3cspeed_offload_policy_from_env(
            (uint64_t)total_vram, (uint64_t)free_vram,
            h3cspeed_system_memory_available_bytes(), &gpu->offload,
            policy_error, sizeof(policy_error))) {
        h3cspeed_set_error(gpu, "CUDA offload configuration", policy_error);
        if (error && error_size) snprintf(error, error_size, "%s", gpu->error);
        h3_gpu_free(gpu);
        return nullptr;
    }
    if (gpu->offload.enabled ||
        environment_flag_enabled("H3_CUDA_OFFLOAD_VERBOSE")) {
        fprintf(stderr,
            "h3cspeed CUDA memory: %s%s, VRAM budget %.2f GiB, "
            "weight cache %.2f GiB, RAM cache %.2f GiB, "
            "pinned cap %.2f MiB, staging %.2f MiB\n",
            h3cspeed_offload_mode_name(gpu->offload.mode),
            gpu->offload.automatic ? " (auto)" : "",
            (double)gpu->offload.vram_budget_bytes / (1024.0 * 1024.0 * 1024.0),
            (double)gpu->offload.weight_cache_bytes / (1024.0 * 1024.0 * 1024.0),
            (double)gpu->offload.host_cache_bytes / (1024.0 * 1024.0 * 1024.0),
            (double)gpu->offload.pinned_host_bytes / (1024.0 * 1024.0),
            (double)gpu->offload.staging_bytes / (1024.0 * 1024.0));
    }
    if (!h3cspeed_cuda_ok(gpu,
            cudaStreamCreateWithFlags(&gpu->compute_stream, cudaStreamNonBlocking),
            "cudaStreamCreate(compute)") ||
        !h3cspeed_cuda_ok(gpu,
            cudaStreamCreateWithFlags(&gpu->upload_stream, cudaStreamNonBlocking),
            "cudaStreamCreate(upload)")) {
        if (error && error_size) snprintf(error, error_size, "%s", gpu->error);
        h3_gpu_free(gpu);
        return nullptr;
    }
    if (!h3cspeed_cublas_ok(gpu, cublasCreate(&gpu->blas), "cublasCreate") ||
        !h3cspeed_cublas_ok(gpu, cublasLtCreate(&gpu->blas_lt), "cublasLtCreate") ||
        !h3cspeed_cublas_ok(gpu, cublasSetStream(gpu->blas, gpu->compute_stream),
                            "cublasSetStream")) {
        if (error && error_size) snprintf(error, error_size, "%s", gpu->error);
        h3_gpu_free(gpu);
        return nullptr;
    }
    if (environment_flag_enabled("H3_CUDA_TF32") &&
        !h3cspeed_cublas_ok(
            gpu, cublasSetMathMode(gpu->blas, CUBLAS_TF32_TENSOR_OP_MATH),
            "cublasSetMathMode(TF32)")) {
        if (error && error_size) snprintf(error, error_size, "%s", gpu->error);
        h3_gpu_free(gpu);
        return nullptr;
    }
    (void)cudaEventCreate(&gpu->profile_start);
    (void)cudaEventCreate(&gpu->profile_mark);
    if (error && error_size) error[0] = '\0';
    return gpu;
}

static void profile_write_final_report(h3_gpu *gpu) {
    if (!gpu || !gpu->profile_enabled || !gpu->profile_json_dir[0] ||
        !gpu->profile_context_id) return;
    h3cspeed_profile_report report;
    memset(&report, 0, sizeof(report));
    pthread_mutex_lock(&gpu->lock);
    report.context_id = gpu->profile_context_id;
    report.device = gpu->device;
    report.sm_major = gpu->properties.major;
    report.sm_minor = gpu->properties.minor;
    report.label = h3cspeed_profile_safe_label(gpu->profile_label);
    report.complete = gpu->error[0] == '\0';
    report.wall_seconds = std::max(
        0.0, h3cspeed_profile_now_seconds() - gpu->profile_context_start_seconds);
    report.metrics = gpu->profile_metrics;
    report.device_peak_bytes = gpu->stats.peak_live_bytes;
    report.resident_weight_peak_bytes = gpu->peak_resident_weight_bytes;
    report.host_cache_peak_bytes = gpu->peak_host_cache_bytes;
    report.offload_uploads = gpu->offload_uploads;
    report.offload_upload_bytes = gpu->offload_upload_bytes;
    report.offload_evictions = gpu->offload_evictions;
    report.offload_evicted_bytes = gpu->offload_evicted_bytes;
    report.file_fallback_reads = gpu->file_fallback_reads;
    report.file_fallback_bytes = gpu->file_fallback_bytes;
    report.direct_dispatches = gpu->stats.direct_dispatches;
    report.linear_dispatches = gpu->stats.mps_linear_dispatches;
    report.convolution_dispatches = gpu->stats.mps_conv_dispatches;
    report.attention_dispatches = gpu->stats.mps_sdpa_dispatches;
    report.perf006.dit_prefetch_requested = gpu->dit_prefetch_requested;
    report.perf006.dit_prefetch_mode = gpu->dit_prefetch_mode;
    report.perf006.async_refill_requested = gpu->async_refill_requested;
    report.perf006.async_refill_active = async_refill_active(gpu);
    report.perf006.ssd_streaming = gpu->profile_dit_ssd_streaming;
    report.perf006.upload_wait_trace_requested =
        gpu->upload_wait_trace_requested;
    report.perf006.upload_wait_trace_complete =
        gpu->upload_wait_trace_requested &&
        gpu->upload_wait_trace_initialized &&
        gpu->upload_wait_trace_complete;
    report.perf006.upload_wait_trace_overflow = gpu->upload_wait_trace_overflow;
    report.perf006.upload_wait_trace_union_valid =
        gpu->upload_wait_trace_requested && gpu->upload_wait_trace_union_valid;
    report.perf006.scope = gpu->profile_dit_scope_seen ?
        "dit_denoise" : "none";
    report.perf006.upload_ready_wait_seconds = gpu->upload_ready_wait_seconds;
    report.perf006.upload_ready_wait_count = gpu->upload_ready_wait_count;
    report.perf006.prefetch_reserve_count = gpu->prefetch_reserve_count;
    report.perf006.prefetch_upload_count = gpu->prefetch_upload_count;
    report.perf006.prefetch_consume_count = gpu->prefetch_consume_count;
    report.perf006.prefetch_cancel_count = gpu->prefetch_cancel_count;
    report.perf006.prefetch_error_count = gpu->prefetch_error_count;
    report.perf006.prefetch_block_count = gpu->prefetch_block_count;
    pthread_mutex_unlock(&gpu->lock);

    char output_path[1024] = {0};
    char profile_error[256] = {0};
    if (!h3cspeed_profile_write_json_directory(
            gpu->profile_json_dir, &report, output_path, sizeof(output_path),
            profile_error, sizeof(profile_error))) {
        fprintf(stderr, "h3cspeed CUDA profile JSON warning: %s\n",
                profile_error[0] ? profile_error : "write failed");
        return;
    }
    fprintf(stderr, "h3cspeed CUDA profile JSON: %s\n", output_path);
}

extern "C" void h3_gpu_free(h3_gpu *gpu) {
    if (!gpu) return;
    (void)cudaSetDevice(gpu->device);
    if (gpu->compute_stream)
        (void)profiled_stream_synchronize(
            gpu, gpu->compute_stream, H3CSPEED_PROFILE_STREAM_COMPUTE);
    if (gpu->upload_stream)
        (void)profiled_stream_synchronize(
            gpu, gpu->upload_stream, H3CSPEED_PROFILE_STREAM_UPLOAD);
    upload_wait_trace_resolve(gpu);
    if (gpu->profile_enabled || gpu->offload.enabled) {
        const char *safe_label = h3cspeed_profile_safe_label(gpu->profile_label);
        const bool has_safe_label = strcmp(safe_label, "redacted") != 0;
        fprintf(stderr,
            "h3cspeed CUDA%s%s%s: device-live=%.2f MiB peak=%.2f MiB "
            "resident-weights=%.2f MiB peak-resident=%.2f MiB "
            "host-cache=%.2f MiB peak-host=%.2f MiB "
            "uploads=%" PRIu64 "/%.2f GiB evictions=%" PRIu64
            "/%.2f GiB host-evictions=%" PRIu64 "/%.2f GiB "
            "file-fallback=%" PRIu64 "/%.2f GiB linear=%" PRIu64
            " conv=%" PRIu64 " sdpa=%" PRIu64 "\n",
            has_safe_label ? " [" : "",
            has_safe_label ? safe_label : "",
            has_safe_label ? "]" : "",
            (double)gpu->device_live_bytes / (1024.0 * 1024.0),
            (double)gpu->stats.peak_live_bytes / (1024.0 * 1024.0),
            (double)gpu->resident_weight_bytes / (1024.0 * 1024.0),
            (double)gpu->peak_resident_weight_bytes / (1024.0 * 1024.0),
            (double)gpu->host_cache_live_bytes / (1024.0 * 1024.0),
            (double)gpu->peak_host_cache_bytes / (1024.0 * 1024.0),
            gpu->offload_uploads,
            (double)gpu->offload_upload_bytes / (1024.0 * 1024.0 * 1024.0),
            gpu->offload_evictions,
            (double)gpu->offload_evicted_bytes / (1024.0 * 1024.0 * 1024.0),
            gpu->host_cache_evictions,
            (double)gpu->host_cache_evicted_bytes /
                (1024.0 * 1024.0 * 1024.0),
            gpu->file_fallback_reads,
            (double)gpu->file_fallback_bytes / (1024.0 * 1024.0 * 1024.0),
            gpu->stats.mps_linear_dispatches,
            gpu->stats.mps_conv_dispatches,
            gpu->stats.mps_sdpa_dispatches);
    }
    profile_write_final_report(gpu);
    /* Both streams were synchronized above; only then is it safe to destroy
     * timing events that may still be referenced by the upload stream. */
    refill_trace_destroy(gpu);
    upload_wait_trace_destroy(gpu);
    if (gpu->scratch) cudaFree(gpu->scratch);
    if (gpu->staging_done[0]) cudaEventDestroy(gpu->staging_done[0]);
    if (gpu->staging_done[1]) cudaEventDestroy(gpu->staging_done[1]);
    if (gpu->staging) {
        if (gpu->staging_pinned) cudaFreeHost(gpu->staging);
        else free(gpu->staging);
    }
    if (gpu->blas) cublasDestroy(gpu->blas);
    if (gpu->blas_lt) cublasLtDestroy(gpu->blas_lt);
    if (gpu->profile_start) cudaEventDestroy(gpu->profile_start);
    if (gpu->profile_mark) cudaEventDestroy(gpu->profile_mark);
    if (gpu->compute_stream) cudaStreamDestroy(gpu->compute_stream);
    if (gpu->upload_stream) cudaStreamDestroy(gpu->upload_stream);
    pthread_mutex_destroy(&gpu->staging_lock);
    pthread_mutex_destroy(&gpu->offload_lock);
    pthread_mutex_destroy(&gpu->scratch_lock);
    pthread_mutex_destroy(&gpu->lock);
    free(gpu);
}

extern "C" int h3_gpu_is_m5(const h3_gpu *gpu) {
    (void)gpu;
    return 0;
}

extern "C" int h3_gpu_has_nax_mlp(const h3_gpu *gpu) {
    (void)gpu;
    return 0;
}

extern "C" int h3_gpu_has_int8_mlp(const h3_gpu *gpu) {
#if H3CSPEED_ENABLE_INT8
    return gpu && gpu->properties.major >= 8;
#else
    (void)gpu;
    return 0;
#endif
}

extern "C" h3_gpu_tensor *h3_gpu_tensor_new_f32(h3_gpu *gpu, size_t elements) {
    return tensor_new(gpu, H3_GPU_F32, elements);
}

extern "C" h3_gpu_tensor *h3_gpu_tensor_new_bf16(h3_gpu *gpu, size_t elements) {
    return tensor_new(gpu, H3_GPU_BF16, elements);
}

extern "C" h3_gpu_tensor *h3_gpu_tensor_new_i8(h3_gpu *gpu, size_t elements) {
    return tensor_new(gpu, H3_GPU_I8, elements);
}

static h3_gpu_tensor *tensor_from(h3_gpu *gpu, h3_gpu_dtype dtype,
                                  const void *values, size_t elements) {
    if (elements && !values) {
        h3cspeed_set_error(gpu, "tensor upload", "input values are null");
        return nullptr;
    }
    h3_gpu_tensor *tensor = tensor_new(gpu, dtype, elements);
    if (!tensor) return nullptr;
    if (dtype == H3_GPU_U32 && values && elements) {
        const uint32_t *items = static_cast<const uint32_t *>(values);
        uint32_t maximum = items[0];
        for (size_t index = 1; index < elements; index++)
            maximum = std::max(maximum, items[index]);
        tensor->u32_max = maximum;
        tensor->u32_max_valid = 1;
    }
    if (tensor->bytes && values) {
        if (!h3cspeed_cuda_ok(gpu,
            profiled_h2d_async(gpu, tensor->data, values, tensor->bytes,
                               gpu->upload_stream),
            "cudaMemcpyAsync tensor upload") ||
            !h3cspeed_tensor_record_upload(tensor)) {
            h3_gpu_tensor_free(tensor);
            return nullptr;
        }
    }
    return tensor;
}

extern "C" h3_gpu_tensor *h3_gpu_tensor_from_f32(h3_gpu *gpu,
                                                   const float *values,
                                                   size_t elements) {
    return tensor_from(gpu, H3_GPU_F32, values, elements);
}

extern "C" h3_gpu_tensor *h3_gpu_tensor_from_bf16(h3_gpu *gpu,
                                                    const uint16_t *values,
                                                    size_t elements) {
    return tensor_from(gpu, H3_GPU_BF16, values, elements);
}

extern "C" h3_gpu_tensor *h3_gpu_tensor_from_u32(h3_gpu *gpu,
                                                   const uint32_t *values,
                                                   size_t elements) {
    return tensor_from(gpu, H3_GPU_U32, values, elements);
}

static int read_exact(h3_gpu *gpu, int descriptor, void *buffer, size_t bytes,
                      uint64_t offset, char *error, size_t error_size) {
    double started = gpu && gpu->profile_enabled ?
        h3cspeed_profile_now_seconds() : 0.0;
    unsigned char *target = static_cast<unsigned char *>(buffer);
    size_t done = 0;
    while (done < bytes) {
        size_t request = std::min(bytes - done, (size_t)(64u * 1024u * 1024u));
        ssize_t got = pread(descriptor, target + done, request,
#if defined(_WIN32)
                            static_cast<int64_t>(offset + done));
#else
                            static_cast<off_t>(offset + done));
#endif
        if (got < 0) {
            if (errno == EINTR) continue;
            if (error && error_size) snprintf(error, error_size,
                "pread failed: %s", strerror(errno));
            if (gpu && gpu->profile_enabled)
                profile_update(gpu, profile_file_read, (uint64_t)done,
                               h3cspeed_profile_now_seconds() - started);
            return 0;
        }
        if (got == 0) {
            if (error && error_size) snprintf(error, error_size,
                "unexpected end of weight file");
            if (gpu && gpu->profile_enabled)
                profile_update(gpu, profile_file_read, (uint64_t)done,
                               h3cspeed_profile_now_seconds() - started);
            return 0;
        }
        done += (size_t)got;
    }
    if (gpu && gpu->profile_enabled)
        profile_update(gpu, profile_file_read, (uint64_t)done,
                       h3cspeed_profile_now_seconds() - started);
    return 1;
}

static int tensor_synchronize_before_host_overwrite_locked(
        h3_gpu_tensor *tensor, const char *operation) {
    if (!tensor || !tensor->gpu) return 0;
    h3_gpu *gpu = tensor->gpu;
    if (!gpu->offload.enabled) {
        return h3cspeed_cuda_ok(gpu,
                   profiled_stream_synchronize(
                       gpu, gpu->compute_stream,
                       H3CSPEED_PROFILE_STREAM_COMPUTE), operation) &&
               h3cspeed_cuda_ok(gpu,
                   profiled_stream_synchronize(
                       gpu, gpu->upload_stream,
                       H3CSPEED_PROFILE_STREAM_UPLOAD), operation);
    }
    if (tensor->data && tensor->pin_epoch == gpu->operation_epoch) {
        h3cspeed_set_error(gpu, operation,
                           "attempted to refill a weight pinned by the current operation");
        return 0;
    }
    if (!tensor_event_synchronize(gpu, tensor, 0,
                                  "wait for tensor upload before refill") ||
        !tensor_event_synchronize(gpu, tensor, 1,
                                  "wait for tensor use before refill")) return 0;
    if (tensor->data && !release_tensor_device_locked(
            gpu, tensor, 0, 1,
            H3CSPEED_PROFILE_EVICTION_PHASE_RETIRE)) return 0;
    return 1;
}

static int validate_file_range(int descriptor, const char *path,
                               uint64_t file_offset, size_t bytes,
                               char *error, size_t error_size) {
    struct stat information;
    if (fstat(descriptor, &information) != 0) {
        if (error && error_size)
            snprintf(error, error_size, "cannot stat %s: %s", path,
                     strerror(errno));
        return 0;
    }
    if (information.st_size < 0 ||
        file_offset > (uint64_t)information.st_size ||
        (uint64_t)bytes > (uint64_t)information.st_size - file_offset) {
        if (error && error_size)
            snprintf(error, error_size, "weight range is outside %s", path);
        return 0;
    }
    return 1;
}

static int tensor_read_file(h3_gpu_tensor *tensor, const char *path,
                            uint64_t file_offset, size_t elements,
                            int streaming, int overwrite_existing,
                            char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!tensor || !tensor->gpu || !path || !*path ||
        elements > tensor->elements) {
        if (error && error_size)
            snprintf(error, error_size, "invalid tensor file-read request");
        return 0;
    }
    h3_gpu *gpu = tensor->gpu;
    size_t item_size = h3cspeed_dtype_size(tensor->dtype);
    size_t bytes = 0;
    const uint64_t off_max =
#if defined(_WIN32)
        static_cast<uint64_t>(INT64_MAX);
#else
        static_cast<uint64_t>(std::numeric_limits<off_t>::max());
#endif
    if (!item_size || !h3cspeed_size_mul(elements, item_size, &bytes) ||
        file_offset > off_max || (uint64_t)bytes > off_max - file_offset) {
        if (error && error_size)
            snprintf(error, error_size, "tensor file-read range overflows");
        return 0;
    }
    int descriptor = open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        if (error && error_size)
            snprintf(error, error_size, "cannot open %s: %s", path,
                     strerror(errno));
        return 0;
    }
    if (!validate_file_range(descriptor, path, file_offset, bytes,
                             error, error_size)) {
        close(descriptor);
        return 0;
    }
#ifdef POSIX_FADV_NOREUSE
    if (streaming) (void)posix_fadvise(descriptor, (off_t)file_offset,
                                       (off_t)bytes, POSIX_FADV_NOREUSE);
#endif

    if (gpu->offload.enabled) {
        char *source_path = strdup(path);
        if (!source_path) {
            close(descriptor);
            if (error && error_size) snprintf(error, error_size,
                                               "out of host memory");
            return 0;
        }
        pthread_mutex_lock(&gpu->offload_lock);
        int ok = 1;
        if ((overwrite_existing || tensor->data) &&
            !tensor_synchronize_before_host_overwrite_locked(
                tensor, "synchronize tensor before offload refill")) ok = 0;
        if (ok) {
            free(tensor->source_path);
            tensor->source_path = source_path;
            source_path = nullptr;
            tensor->source_offset = file_offset;
            tensor->source_bytes = bytes;
            tensor->source_streaming = streaming;
            tensor->offloadable = 1;
            tensor->pin_epoch = 0;
            pthread_mutex_lock(&tensor->lock);
            tensor->ready_valid = 0;
            tensor->last_use_valid = 0;
            pthread_mutex_unlock(&tensor->lock);
            if (!tensor->host_data) (void)host_backing_allocate_locked(gpu, tensor);
            if (tensor->host_data) {
                ok = read_exact(gpu, descriptor, tensor->host_data, bytes,
                                file_offset, error, error_size);
                if (ok && bytes < tensor->bytes)
                    memset(static_cast<unsigned char *>(tensor->host_data) + bytes,
                           0, tensor->bytes - bytes);
                tensor->host_valid = ok;
                if (ok) host_lru_append_locked(gpu, tensor);
                else host_backing_release_locked(gpu, tensor);
            } else {
                /* The path+offset remains a valid third tier when the configured
                 * RAM cache is full. It will be read through pinned staging on use. */
                tensor->host_valid = 0;
            }
        }
        pthread_mutex_unlock(&gpu->offload_lock);
        free(source_path);
        close(descriptor);
        if (!ok && error && error_size && !error[0])
            snprintf(error, error_size, "%s", h3_gpu_error(gpu));
        return ok;
    }

    if (overwrite_existing &&
        !tensor_synchronize_before_host_overwrite_locked(
            tensor, "synchronize tensor before file refill")) {
        close(descriptor);
        if (error && error_size)
            snprintf(error, error_size, "%s", h3_gpu_error(gpu));
        return 0;
    }
    if (!tensor->data && tensor->bytes) {
        pthread_mutex_lock(&gpu->offload_lock);
        int allocated = device_allocate_locked(gpu, &tensor->data,
                                                tensor->bytes, 0, nullptr);
        pthread_mutex_unlock(&gpu->offload_lock);
        if (!allocated) {
            close(descriptor);
            if (error && error_size)
                snprintf(error, error_size, "%s", h3_gpu_error(gpu));
            return 0;
        }
    }

    pthread_mutex_lock(&gpu->staging_lock);
    int ok = staging_allocate_locked(gpu);
    size_t done = 0;
    while (ok && done < bytes) {
        size_t chunk = std::min(bytes - done, gpu->staging_bytes);
        if (!read_exact(gpu, descriptor, gpu->staging, chunk,
                        file_offset + done,
                        error, error_size) ||
            !h3cspeed_cuda_ok(gpu,
                profiled_h2d_async(
                    gpu, static_cast<unsigned char *>(tensor->data) + done,
                    gpu->staging, chunk, gpu->upload_stream),
                "stream weight upload") ||
            !h3cspeed_cuda_ok(gpu,
                profiled_stream_synchronize(
                    gpu, gpu->upload_stream, H3CSPEED_PROFILE_STREAM_UPLOAD),
                "weight upload synchronization")) {
            ok = 0;
            break;
        }
        done += chunk;
    }
    pthread_mutex_unlock(&gpu->staging_lock);
    close(descriptor);
    if (ok) ok = h3cspeed_tensor_record_upload(tensor);
    if (!ok && error && error_size && !error[0])
        snprintf(error, error_size, "%s", h3_gpu_error(gpu));
    return ok;
}

extern "C" h3_gpu_tensor *h3_gpu_tensor_load_bf16(h3_gpu *gpu,
                                                    const char *path,
                                                    uint64_t file_offset,
                                                    size_t elements) {
    h3_gpu_tensor *tensor = tensor_new_internal(
        gpu, H3_GPU_BF16, elements, gpu && !gpu->offload.enabled);
    char error[512] = {0};
    if (tensor && !tensor_read_file(tensor, path, file_offset, elements,
                                    0, 0, error, sizeof(error))) {
        h3cspeed_set_error(gpu, "load BF16 tensor", error);
        h3_gpu_tensor_free(tensor);
        tensor = nullptr;
    }
    return tensor;
}

extern "C" h3_gpu_tensor *h3_gpu_tensor_load_f32(h3_gpu *gpu,
                                                   const char *path,
                                                   uint64_t file_offset,
                                                   size_t elements) {
    h3_gpu_tensor *tensor = tensor_new_internal(
        gpu, H3_GPU_F32, elements, gpu && !gpu->offload.enabled);
    char error[512] = {0};
    if (tensor && !tensor_read_file(tensor, path, file_offset, elements,
                                    0, 0, error, sizeof(error))) {
        h3cspeed_set_error(gpu, "load F32 tensor", error);
        h3_gpu_tensor_free(tensor);
        tensor = nullptr;
    }
    return tensor;
}

extern "C" h3_gpu_tensor *h3cspeed_gpu_tensor_load_i8_convrot(
        h3_gpu *gpu, const char *path, uint64_t file_offset,
        size_t elements, uint32_t group_size) {
    if (!gpu || !path || !*path || !convrot_group_valid(group_size)) {
        h3cspeed_set_error(gpu, "load ConvRot INT8 tensor",
                           "invalid path or group size (expected 4^n <= 256)");
        return nullptr;
    }
    h3_gpu_tensor *tensor = tensor_new_internal(
        gpu, H3_GPU_I8, elements, gpu && !gpu->offload.enabled);
    char error[512] = {0};
    if (!tensor || !tensor_read_file(tensor, path, file_offset, elements,
                                     0, 0, error, sizeof(error))) {
        h3cspeed_set_error(gpu, "load ConvRot INT8 tensor",
                           error[0] ? error : "tensor allocation failed");
        h3_gpu_tensor_free(tensor);
        return nullptr;
    }
    tensor->convrot = 1;
    tensor->convrot_group_size = group_size;
    return tensor;
}

extern "C" int h3_gpu_tensor_read_file_bf16(h3_gpu_tensor *tensor,
                                              const char *path,
                                              uint64_t file_offset,
                                              size_t elements,
                                              char *error, size_t error_size) {
    if (!tensor || tensor->dtype != H3_GPU_BF16) return 0;
    return tensor_read_file(tensor, path, file_offset, elements,
                            0, 1, error, error_size);
}

extern "C" int h3_gpu_tensor_stream_file_bf16(h3_gpu_tensor *tensor,
                                                const char *path,
                                                uint64_t file_offset,
                                                size_t elements,
                                                char *error,
                                                size_t error_size) {
    if (!tensor || tensor->dtype != H3_GPU_BF16) return 0;
    return tensor_read_file(tensor, path, file_offset, elements,
                            1, 1, error, error_size);
}

extern "C" void h3_gpu_tensor_free(h3_gpu_tensor *tensor) {
    if (!tensor) return;
    h3_gpu *gpu = tensor->gpu;
    if (gpu) {
        pthread_mutex_lock(&gpu->offload_lock);
        (void)release_tensor_device_locked(
            gpu, tensor, tensor->offloadable ? 1 : 0, 0,
            H3CSPEED_PROFILE_EVICTION_PHASE_RETIRE);
        host_backing_release_locked(gpu, tensor);
        free(tensor->source_path);
        tensor->source_path = nullptr;
        pthread_mutex_unlock(&gpu->offload_lock);
    } else {
        free(tensor->source_path);
    }
    if (tensor->ready) cudaEventDestroy(tensor->ready);
    if (tensor->last_use) cudaEventDestroy(tensor->last_use);
    pthread_mutex_destroy(&tensor->lock);
    free(tensor);
}

extern "C" size_t h3_gpu_tensor_elements(const h3_gpu_tensor *tensor) {
    return tensor ? tensor->elements : 0;
}

extern "C" h3_gpu_dtype h3_gpu_tensor_dtype(const h3_gpu_tensor *tensor) {
    return tensor ? tensor->dtype : H3_GPU_F32;
}

static int tensor_read_range(const h3_gpu_tensor *tensor, size_t source_offset,
                             void *values, size_t elements,
                             h3_gpu_dtype expected) {
    if (!tensor || !values || tensor->dtype != expected ||
        source_offset > tensor->elements || elements > tensor->elements - source_offset)
        return 0;
    h3_gpu *gpu = tensor->gpu;
    if (!h3cspeed_tensor_wait(gpu, tensor) ||
        !h3cspeed_cuda_ok(gpu, profiled_stream_synchronize(
                               gpu, gpu->compute_stream,
                               H3CSPEED_PROFILE_STREAM_COMPUTE),
                           "read tensor compute synchronization") ||
        !h3cspeed_cuda_ok(gpu, profiled_stream_synchronize(
                               gpu, gpu->upload_stream,
                               H3CSPEED_PROFILE_STREAM_UPLOAD),
                           "read tensor upload synchronization")) return 0;
    int ok = h3cspeed_cuda_ok(gpu,
        cudaMemcpy(values,
                   static_cast<const unsigned char *>(tensor->data) +
                       source_offset * h3cspeed_dtype_size(expected),
                   elements * h3cspeed_dtype_size(expected),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy tensor read");
    h3cspeed_operation_complete(gpu);
    return ok;
}

extern "C" int h3_gpu_tensor_read_f32(const h3_gpu_tensor *tensor,
                                        float *values, size_t elements) {
    return tensor_read_range(tensor, 0, values, elements, H3_GPU_F32);
}

extern "C" int h3_gpu_tensor_read_f32_range(const h3_gpu_tensor *tensor,
                                              size_t source_offset,
                                              float *values, size_t elements) {
    return tensor_read_range(tensor, source_offset, values, elements, H3_GPU_F32);
}

extern "C" int h3_gpu_tensor_read_bf16(const h3_gpu_tensor *tensor,
                                         uint16_t *values, size_t elements) {
    return tensor_read_range(tensor, 0, values, elements, H3_GPU_BF16);
}

static int tensor_write_range(h3_gpu_tensor *tensor, size_t destination_offset,
                              const void *values, size_t elements,
                              h3_gpu_dtype expected) {
    if (!tensor || !values || tensor->dtype != expected ||
        destination_offset > tensor->elements ||
        elements > tensor->elements - destination_offset) return 0;
    h3_gpu *gpu = tensor->gpu;
    if (!h3cspeed_tensor_prepare(gpu, tensor) ||
        !h3cspeed_cuda_ok(gpu, profiled_stream_synchronize(
                               gpu, gpu->compute_stream,
                               H3CSPEED_PROFILE_STREAM_COMPUTE),
                           "synchronize tensor before host write") ||
        !h3cspeed_cuda_ok(gpu, profiled_stream_synchronize(
                               gpu, gpu->upload_stream,
                               H3CSPEED_PROFILE_STREAM_UPLOAD),
                           "synchronize upload before host write") ||
        !h3cspeed_cuda_ok(gpu,
            profiled_h2d_async(
                gpu, static_cast<unsigned char *>(tensor->data) +
                         destination_offset * h3cspeed_dtype_size(expected),
                values, elements * h3cspeed_dtype_size(expected),
                gpu->upload_stream),
            "cudaMemcpyAsync tensor write") ||
        !h3cspeed_tensor_record_upload(tensor)) return 0;

    /* File weights are read-only in the released model. If an embedding or
     * application mutates one through the public write API, keep that tensor
     * resident instead of later restoring stale bytes from its source file. */
    pthread_mutex_lock(&gpu->offload_lock);
    if (tensor->offloadable) {
        lru_remove_locked(gpu, tensor);
        gpu->resident_weight_bytes = gpu->resident_weight_bytes >= tensor->bytes ?
            gpu->resident_weight_bytes - tensor->bytes : 0;
        host_backing_release_locked(gpu, tensor);
        free(tensor->source_path);
        tensor->source_path = nullptr;
        tensor->source_bytes = 0;
        tensor->offloadable = 0;
        tensor->pin_epoch = 0;
    }
    pthread_mutex_unlock(&gpu->offload_lock);
    h3cspeed_operation_complete(gpu);
    return 1;
}

extern "C" int h3_gpu_tensor_write_f32(h3_gpu_tensor *tensor,
                                         const float *values, size_t elements) {
    return tensor_write_range(tensor, 0, values, elements, H3_GPU_F32);
}

extern "C" int h3_gpu_tensor_write_f32_range(h3_gpu_tensor *tensor,
                                               size_t destination_offset,
                                               const float *values,
                                               size_t elements) {
    return tensor_write_range(tensor, destination_offset, values, elements,
                              H3_GPU_F32);
}

extern "C" int h3_gpu_tensor_write_bf16(h3_gpu_tensor *tensor,
                                          const uint16_t *values,
                                          size_t elements) {
    return tensor_write_range(tensor, 0, values, elements, H3_GPU_BF16);
}

extern "C" int h3_gpu_tensor_write_bf16_range(h3_gpu_tensor *tensor,
                                                size_t destination_offset,
                                                const uint16_t *values,
                                                size_t elements) {
    return tensor_write_range(tensor, destination_offset, values, elements,
                              H3_GPU_BF16);
}

extern "C" int h3_gpu_begin(h3_gpu *gpu) {
    if (!gpu) return 0;
    if (gpu->profile_enabled) {
        clock_gettime(CLOCK_MONOTONIC, &gpu->profile_wall);
        cudaError_t status = cudaEventRecord(gpu->profile_start,
                                             gpu->compute_stream);
        pthread_mutex_lock(&gpu->lock);
        gpu->profile_metrics.begin_count++;
        gpu->profile_compute_span_active = status == cudaSuccess;
        pthread_mutex_unlock(&gpu->lock);
    }
    return 1;
}

extern "C" int h3_gpu_continue(h3_gpu *gpu) {
    if (!gpu) return 0;
    pthread_mutex_lock(&gpu->lock);
    gpu->stats.submissions++;
    if (gpu->profile_enabled) gpu->profile_metrics.continue_count++;
    pthread_mutex_unlock(&gpu->lock);
    return 1;
}

extern "C" int h3_gpu_submit(h3_gpu *gpu) {
    if (!gpu) return 0;
    h3cspeed_operation_complete(gpu);
    int device_span = 0;
    pthread_mutex_lock(&gpu->lock);
    int compute_span_active = gpu->profile_compute_span_active;
    pthread_mutex_unlock(&gpu->lock);
    if (gpu->profile_enabled && compute_span_active &&
        cudaEventRecord(gpu->profile_mark, gpu->compute_stream) == cudaSuccess)
        device_span = 1;
    struct timespec start, stop;
    clock_gettime(CLOCK_MONOTONIC, &start);
    /* Always drain both streams.  In particular, a compute failure must not
     * short-circuit the upload fence and leave prefetched storage in flight
     * while the caller unwinds and frees the DiT. */
    int compute_ok = h3cspeed_cuda_ok(
        gpu, profiled_stream_synchronize(
                 gpu, gpu->compute_stream, H3CSPEED_PROFILE_STREAM_COMPUTE),
        "cudaStreamSynchronize(compute)");
    char compute_error[512] = {0};
    if (!compute_ok)
        snprintf(compute_error, sizeof(compute_error), "%s",
                 h3_gpu_error(gpu));
    int upload_ok = h3cspeed_cuda_ok(
        gpu, profiled_stream_synchronize(
                 gpu, gpu->upload_stream, H3CSPEED_PROFILE_STREAM_UPLOAD),
        "cudaStreamSynchronize(upload)");
    upload_wait_trace_resolve(gpu);
    if (!compute_ok) {
        if (!upload_ok) {
            char submit_error[512];
            snprintf(submit_error, sizeof(submit_error),
                     "compute drain failed (%s); upload drain failed (%s)",
                     compute_error, h3_gpu_error(gpu));
            h3cspeed_set_error(gpu, "CUDA submit", submit_error);
        } else {
            h3cspeed_set_error(gpu, compute_error, nullptr);
        }
    }
    int ok = compute_ok && upload_ok;
    if (ok && gpu->offload.release_scratch_on_submit) {
        pthread_mutex_lock(&gpu->scratch_lock);
        pthread_mutex_lock(&gpu->offload_lock);
        if (gpu->scratch) {
            void *scratch = gpu->scratch;
            size_t scratch_bytes = gpu->scratch_bytes;
            if (h3cspeed_cuda_ok(gpu, cudaFree(scratch),
                                 "cudaFree low-VRAM scratch")) {
                gpu->scratch = nullptr;
                gpu->scratch_bytes = 0;
                track_device_release(gpu, scratch_bytes);
            } else {
                ok = 0;
            }
        }
        pthread_mutex_unlock(&gpu->offload_lock);
        pthread_mutex_unlock(&gpu->scratch_lock);
    }
    clock_gettime(CLOCK_MONOTONIC, &stop);
    pthread_mutex_lock(&gpu->lock);
    gpu->stats.submissions++;
    gpu->stats.command_wait_seconds += elapsed_seconds(&start, &stop);
    if (gpu->profile_enabled) {
        gpu->profile_metrics.submit_sync_count++;
        if (device_span) {
            float milliseconds = 0.0f;
            if (cudaEventElapsedTime(&milliseconds, gpu->profile_start,
                                     gpu->profile_mark) == cudaSuccess)
                gpu->profile_metrics.compute_device_seconds +=
                    (double)milliseconds / 1000.0;
        }
        gpu->profile_compute_span_active = 0;
    }
    pthread_mutex_unlock(&gpu->lock);
    return ok;
}

extern "C" const char *h3_gpu_error(const h3_gpu *gpu) {
    return gpu ? gpu->error : "CUDA context is null";
}

extern "C" int h3_gpu_get_stats(const h3_gpu *gpu, h3_gpu_stats *stats) {
    if (!gpu || !stats) return 0;
    pthread_mutex_lock(const_cast<pthread_mutex_t *>(&gpu->lock));
    *stats = gpu->stats;
    pthread_mutex_unlock(const_cast<pthread_mutex_t *>(&gpu->lock));
    return 1;
}

extern "C" void h3_gpu_profile_set_label(h3_gpu *gpu, const char *label) {
    if (!gpu) return;
    pthread_mutex_lock(&gpu->lock);
    snprintf(gpu->profile_label, sizeof(gpu->profile_label), "%s", label ? label : "");
    pthread_mutex_unlock(&gpu->lock);
}

extern "C" void h3_gpu_profile_mark(h3_gpu *gpu, const char *phase) {
    if (!gpu || !gpu->profile_enabled) return;
    (void)cudaEventRecord(gpu->profile_mark, gpu->compute_stream);
    (void)profiled_event_synchronize(gpu, gpu->profile_mark);
    float milliseconds = 0.0f;
    (void)cudaEventElapsedTime(&milliseconds, gpu->profile_start, gpu->profile_mark);
    char label_snapshot[sizeof(gpu->profile_label)];
    pthread_mutex_lock(&gpu->lock);
    snprintf(label_snapshot, sizeof(label_snapshot), "%s", gpu->profile_label);
    pthread_mutex_unlock(&gpu->lock);
    const char *safe_label = h3cspeed_profile_safe_label(label_snapshot);
    const bool has_safe_label = strcmp(safe_label, "redacted") != 0;
    fprintf(stderr, "h3cspeed CUDA%s%s%s: %s %.3f s\n",
            has_safe_label ? " [" : "",
            has_safe_label ? safe_label : "",
            has_safe_label ? "]" : "",
            h3cspeed_profile_safe_phase(phase), milliseconds / 1000.0f);
}

void *h3cspeed_scratch_reserve(h3_gpu *gpu, size_t bytes) {
    if (!gpu || !bytes) return nullptr;
    pthread_mutex_lock(&gpu->scratch_lock);
    if (bytes > gpu->scratch_bytes) {
        if (!h3cspeed_cuda_ok(gpu, profiled_stream_synchronize(
                                   gpu, gpu->compute_stream,
                                   H3CSPEED_PROFILE_STREAM_COMPUTE),
                               "synchronize before scratch resize")) {
            pthread_mutex_unlock(&gpu->scratch_lock);
            return nullptr;
        }
        pthread_mutex_lock(&gpu->offload_lock);
        if (gpu->scratch) {
            void *old = gpu->scratch;
            size_t old_bytes = gpu->scratch_bytes;
            if (!h3cspeed_cuda_ok(gpu, cudaFree(old), "cudaFree scratch")) {
                pthread_mutex_unlock(&gpu->offload_lock);
                pthread_mutex_unlock(&gpu->scratch_lock);
                return nullptr;
            }
            gpu->scratch = nullptr;
            gpu->scratch_bytes = 0;
            track_device_release(gpu, old_bytes);
        }
        if (!device_allocate_locked(gpu, &gpu->scratch, bytes, 0, nullptr)) {
            pthread_mutex_unlock(&gpu->offload_lock);
            pthread_mutex_unlock(&gpu->scratch_lock);
            return nullptr;
        }
        gpu->scratch_bytes = bytes;
        pthread_mutex_unlock(&gpu->offload_lock);
    }
    void *result = gpu->scratch;
    pthread_mutex_unlock(&gpu->scratch_lock);
    return result;
}

__global__ static void mixed_matmul_kernel(
        void *output, h3_gpu_dtype output_dtype, size_t output_offset,
        const void *input, h3_gpu_dtype input_dtype, size_t input_offset,
        const void *weight, h3_gpu_dtype weight_dtype,
        const void *bias, h3_gpu_dtype bias_dtype,
        uint32_t rows, uint32_t input_dim, uint32_t output_dim) {
    constexpr unsigned tile = 16;
    __shared__ float input_tile[tile][tile];
    __shared__ float weight_tile[tile][tile];
    uint32_t output_column = blockIdx.x * tile + threadIdx.x;
    uint32_t output_row = blockIdx.y * tile + threadIdx.y;
    float accumulator = 0.0f;
    for (uint32_t base = 0; base < input_dim; base += tile) {
        uint32_t input_column = base + threadIdx.x;
        uint32_t weight_column = base + threadIdx.y;
        input_tile[threadIdx.y][threadIdx.x] =
            output_row < rows && input_column < input_dim ?
            h3cspeed_device_load(input, input_dtype,
                input_offset + (size_t)output_row * input_dim + input_column) : 0.0f;
        weight_tile[threadIdx.y][threadIdx.x] =
            output_column < output_dim && weight_column < input_dim ?
            h3cspeed_device_load(weight, weight_dtype,
                (size_t)output_column * input_dim + weight_column) : 0.0f;
        __syncthreads();
        #pragma unroll
        for (unsigned inner = 0; inner < tile; inner++)
            accumulator += input_tile[threadIdx.y][inner] *
                           weight_tile[inner][threadIdx.x];
        __syncthreads();
    }
    if (output_row < rows && output_column < output_dim) {
        if (bias) accumulator += h3cspeed_device_load(bias, bias_dtype, output_column);
        h3cspeed_device_store(output, output_dtype,
            output_offset + (size_t)output_row * output_dim + output_column,
            accumulator);
    }
}

__global__ static void add_bias_kernel(void *output, h3_gpu_dtype output_dtype,
                                       size_t output_offset, const void *bias,
                                       h3_gpu_dtype bias_dtype, uint32_t rows,
                                       uint32_t width) {
    size_t total = (size_t)rows * width;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += (size_t)gridDim.x * blockDim.x) {
        uint32_t column = (uint32_t)(index % width);
        float value = h3cspeed_device_load(output, output_dtype,
                                            output_offset + index);
        value += h3cspeed_device_load(bias, bias_dtype, column);
        h3cspeed_device_store(output, output_dtype, output_offset + index, value);
    }
}

static const char *h3cspeed_dtype_name(h3_gpu_dtype dtype) {
    switch (dtype) {
        case H3_GPU_F32: return "F32";
        case H3_GPU_BF16: return "BF16";
        case H3_GPU_I8: return "I8";
        case H3_GPU_U32: return "U32";
    }
    return "unknown";
}

static int h3cspeed_cublas_linear_supported(h3_gpu_dtype input_dtype,
                                             h3_gpu_dtype weight_dtype,
                                             h3_gpu_dtype output_dtype,
                                             int has_bias) {
    if (input_dtype != weight_dtype ||
        (input_dtype != H3_GPU_F32 && input_dtype != H3_GPU_BF16)) return 0;
    /* CUDA 13.2/cuBLAS on Ampere does not implement the F32/F32 -> BF16 C
     * variant of GemmEx with COMPUTE_32F.  Also keep BF16 output with a bias
     * on the mixed kernel: cuBLAS would round the GEMM to BF16 before the
     * bias kernel adds and rounds again.  The mixed kernel applies bias in
     * FP32 and performs one explicit __float2bfloat16_rn store. */
    return output_dtype == H3_GPU_F32 ||
           (input_dtype == H3_GPU_BF16 && output_dtype == H3_GPU_BF16 &&
            !has_bias);
}

int h3cspeed_linear(h3_gpu *gpu, h3_gpu_tensor *output, size_t output_offset,
                    const h3_gpu_tensor *input, size_t input_offset,
                    const h3_gpu_tensor *weight, const h3_gpu_tensor *bias,
                    uint32_t rows, uint32_t input_dim, uint32_t output_dim) {
    size_t output_elements = 0;
    size_t input_elements = 0;
    size_t weight_elements = 0;
    if (!gpu || !output || !input || !weight || !rows || !input_dim ||
        !output_dim || rows > INT_MAX || input_dim > INT_MAX ||
        output_dim > INT_MAX ||
        !h3cspeed_size_mul(rows, output_dim, &output_elements) ||
        !h3cspeed_size_mul(rows, input_dim, &input_elements) ||
        !h3cspeed_size_mul(output_dim, input_dim, &weight_elements) ||
        output_offset > output->elements ||
        output_elements > output->elements - output_offset ||
        input_offset > input->elements ||
        input_elements > input->elements - input_offset ||
        weight_elements > weight->elements ||
        (bias && bias->elements < output_dim)) return 0;
    if (!h3cspeed_tensor_wait(gpu, input) || !h3cspeed_tensor_wait(gpu, weight) ||
        (bias && !h3cspeed_tensor_wait(gpu, bias))) return 0;

    const int cublas_compatible = h3cspeed_cublas_linear_supported(
        input->dtype, weight->dtype, output->dtype, bias != nullptr);
    if (cublas_compatible) {
        float alpha = 1.0f;
        float beta = 0.0f;
        const unsigned char *weight_pointer =
            static_cast<const unsigned char *>(weight->data);
        const unsigned char *input_pointer =
            static_cast<const unsigned char *>(input->data) +
            input_offset * h3cspeed_dtype_size(input->dtype);
        unsigned char *output_pointer = static_cast<unsigned char *>(output->data) +
            output_offset * h3cspeed_dtype_size(output->dtype);
        cublasStatus_t status = cublasGemmEx(
            gpu->blas, CUBLAS_OP_T, CUBLAS_OP_N,
            (int)output_dim, (int)rows, (int)input_dim,
            &alpha,
            weight_pointer, h3cspeed_cuda_dtype(weight->dtype), (int)input_dim,
            input_pointer, h3cspeed_cuda_dtype(input->dtype), (int)input_dim,
            &beta,
            output_pointer, h3cspeed_cuda_dtype(output->dtype), (int)output_dim,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (!h3cspeed_cublas_ok(gpu, status, "cublasGemmEx linear")) {
            char detail[320];
            snprintf(detail, sizeof(detail),
                     "status=%d M=%u N=%u K=%u opA=T opB=N lda=%u ldb=%u "
                     "ldc=%u A=%s B=%s C=%s compute=32F",
                     (int)status, output_dim, rows, input_dim, input_dim,
                     input_dim, output_dim, h3cspeed_dtype_name(weight->dtype),
                     h3cspeed_dtype_name(input->dtype),
                     h3cspeed_dtype_name(output->dtype));
            h3cspeed_set_error(gpu, "cublasGemmEx linear", detail);
            return 0;
        }
        if (bias) {
            add_bias_kernel<<<h3cspeed_blocks((size_t)rows * output_dim), 256, 0,
                              gpu->compute_stream>>>(
                output->data, output->dtype, output_offset, bias->data, bias->dtype,
                rows, output_dim);
            if (!h3cspeed_launch_ok(gpu, "add bias kernel")) return 0;
        }
    } else {
        dim3 block(16, 16);
        dim3 grid((output_dim + 15) / 16, (rows + 15) / 16);
        mixed_matmul_kernel<<<grid, block, 0, gpu->compute_stream>>>(
            output->data, output->dtype, output_offset,
            input->data, input->dtype, input_offset,
            weight->data, weight->dtype,
            bias ? bias->data : nullptr, bias ? bias->dtype : H3_GPU_F32,
            rows, input_dim, output_dim);
        if (!h3cspeed_launch_ok(gpu, "mixed matmul kernel")) return 0;
    }
    h3cspeed_count_linear(gpu);
    return 1;
}

extern "C" int h3_gpu_linear_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                   const h3_gpu_tensor *input,
                                   const h3_gpu_tensor *weight,
                                   const h3_gpu_tensor *bias, uint32_t rows,
                                   uint32_t input_dim, uint32_t output_dim) {
    return h3cspeed_linear(gpu, output, 0, input, 0, weight, bias,
                          rows, input_dim, output_dim);
}

extern "C" int h3_gpu_patch_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                          const h3_gpu_tensor *input,
                                          const h3_gpu_tensor *weight,
                                          const h3_gpu_tensor *bias,
                                          uint32_t rows, uint32_t input_dim,
                                          uint32_t output_dim) {
    return h3cspeed_linear(gpu, output, 0, input, 0, weight, bias,
                          rows, input_dim, output_dim);
}

extern "C" int h3_gpu_patch_linear_bf16_offset(
        h3_gpu *gpu, h3_gpu_tensor *output, size_t output_offset,
        const h3_gpu_tensor *input, size_t input_offset,
        const h3_gpu_tensor *weight, const h3_gpu_tensor *bias,
        uint32_t rows, uint32_t input_dim, uint32_t output_dim) {
    return h3cspeed_linear(gpu, output, output_offset, input, input_offset,
                          weight, bias, rows, input_dim, output_dim);
}

__global__ static void scatter_rows_kernel(void *output, h3_gpu_dtype output_dtype,
                                           const void *input,
                                           h3_gpu_dtype input_dtype,
                                           const uint32_t *map,
                                           uint32_t rows, uint32_t width,
                                           uint32_t output_rows) {
    size_t total = (size_t)rows * width;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += (size_t)gridDim.x * blockDim.x) {
        uint32_t row = (uint32_t)(index / width);
        uint32_t column = (uint32_t)(index % width);
        uint32_t destination = map[row];
        if (destination < output_rows) {
            float value = h3cspeed_device_load(input, input_dtype, index);
            h3cspeed_device_store(output, output_dtype,
                (size_t)destination * width + column, value);
        }
    }
}

extern "C" int h3_gpu_patch_linear_bf16_map(
        h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
        const h3_gpu_tensor *weight, const h3_gpu_tensor *bias,
        const h3_gpu_tensor *row_map, uint32_t output_rows, uint32_t rows,
        uint32_t input_dim, uint32_t output_dim) {
    size_t output_elements = 0;
    size_t temporary_elements = 0;
    if (!gpu || !output || !input || !weight || !row_map || !output_rows ||
        !rows || !input_dim || !output_dim ||
        row_map->dtype != H3_GPU_U32 || row_map->elements < rows ||
        !h3cspeed_size_mul(output_rows, output_dim, &output_elements) ||
        !h3cspeed_size_mul(rows, output_dim, &temporary_elements) ||
        output->elements < output_elements ||
        (row_map->u32_max_valid && row_map->u32_max >= output_rows)) return 0;
    h3_gpu_tensor *temporary = output->dtype == H3_GPU_BF16 ?
        h3_gpu_tensor_new_bf16(gpu, temporary_elements) :
        h3_gpu_tensor_new_f32(gpu, temporary_elements);
    if (!temporary) return 0;
    int ok = h3cspeed_linear(gpu, temporary, 0, input, 0, weight, bias,
                            rows, input_dim, output_dim) &&
             h3cspeed_tensor_wait(gpu, row_map);
    if (ok) {
        scatter_rows_kernel<<<h3cspeed_blocks(temporary_elements), 256, 0,
                              gpu->compute_stream>>>(
            output->data, output->dtype, temporary->data, temporary->dtype,
            static_cast<const uint32_t *>(row_map->data), rows, output_dim,
            output_rows);
        ok = h3cspeed_launch_ok(gpu, "scatter patch projection");
        if (ok) h3cspeed_count_direct(gpu);
    }
    h3_gpu_tensor_free(temporary);
    return ok;
}

__global__ static void unary_kernel(void *output, h3_gpu_dtype output_dtype,
                                    const void *input, h3_gpu_dtype input_dtype,
                                    size_t elements, int operation, int approximate,
                                    float minimum, float maximum) {
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += (size_t)gridDim.x * blockDim.x) {
        float value = h3cspeed_device_load(input, input_dtype, index);
        switch (operation) {
            case 0: value = value / (1.0f + expf(-value)); break; /* SiLU */
            case 1: {
                if (approximate) {
                    const float coefficient = 0.7978845608028654f;
                    value = 0.5f * value * (1.0f + tanhf(
                        coefficient * (value + 0.044715f * value * value * value)));
                } else {
                    value = 0.5f * value * (1.0f + erff(value * 0.7071067811865475f));
                }
                break;
            }
            case 2: value = fminf(maximum, fmaxf(minimum, value)); break;
            default: break;
        }
        h3cspeed_device_store(output, output_dtype, index, value);
    }
}

static int unary(h3_gpu *gpu, h3_gpu_tensor *output,
                 const h3_gpu_tensor *input, uint32_t elements,
                 int operation, int approximate = 0,
                 float minimum = 0.0f, float maximum = 0.0f) {
    if (!gpu || !output || !input || output->elements < elements ||
        input->elements < elements || !h3cspeed_tensor_wait(gpu, input)) return 0;
    unary_kernel<<<h3cspeed_blocks(elements), 256, 0, gpu->compute_stream>>>(
        output->data, output->dtype, input->data, input->dtype, elements,
        operation, approximate, minimum, maximum);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "unary CUDA kernel");
}

extern "C" int h3_gpu_silu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                 const h3_gpu_tensor *input,
                                 uint32_t elements) {
    return unary(gpu, output, input, elements, 0);
}

extern "C" int h3_gpu_silu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *input,
                                  uint32_t elements) {
    return unary(gpu, output, input, elements, 0);
}

extern "C" int h3_gpu_gelu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *input,
                                  uint32_t elements, int approximate) {
    return unary(gpu, output, input, elements, 1, approximate);
}

extern "C" int h3_gpu_clip_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                 const h3_gpu_tensor *input, uint32_t elements,
                                 float minimum, float maximum) {
    return unary(gpu, output, input, elements, 2, 0, minimum, maximum);
}

__global__ static void copy_cast_kernel(void *output, h3_gpu_dtype output_dtype,
                                        size_t output_offset, const void *input,
                                        h3_gpu_dtype input_dtype,
                                        size_t input_offset, size_t elements) {
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += (size_t)gridDim.x * blockDim.x) {
        h3cspeed_device_store(output, output_dtype, output_offset + index,
            h3cspeed_device_load(input, input_dtype, input_offset + index));
    }
}

static int copy_cast(h3_gpu *gpu, h3_gpu_tensor *output, size_t output_offset,
                     const h3_gpu_tensor *input, size_t input_offset,
                     size_t elements) {
    if (!gpu || !output || !input || output_offset > output->elements ||
        input_offset > input->elements || elements > output->elements - output_offset ||
        elements > input->elements - input_offset ||
        !h3cspeed_tensor_wait(gpu, input)) return 0;
    if (output->dtype == input->dtype) {
        if (!h3cspeed_cuda_ok(gpu,
            cudaMemcpyAsync(static_cast<unsigned char *>(output->data) +
                                output_offset * h3cspeed_dtype_size(output->dtype),
                            static_cast<const unsigned char *>(input->data) +
                                input_offset * h3cspeed_dtype_size(input->dtype),
                            elements * h3cspeed_dtype_size(input->dtype),
                            cudaMemcpyDeviceToDevice, gpu->compute_stream),
            "cudaMemcpyAsync tensor copy")) return 0;
        pthread_mutex_lock(&gpu->lock);
        gpu->stats.blit_copies++;
        pthread_mutex_unlock(&gpu->lock);
        return 1;
    }
    copy_cast_kernel<<<h3cspeed_blocks(elements), 256, 0, gpu->compute_stream>>>(
        output->data, output->dtype, output_offset,
        input->data, input->dtype, input_offset, elements);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "copy/cast CUDA kernel");
}

extern "C" int h3_gpu_cast_f32_to_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                         const h3_gpu_tensor *input,
                                         uint32_t elements) {
    return copy_cast(gpu, output, 0, input, 0, elements);
}

extern "C" int h3_gpu_cast_bf16_to_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                         const h3_gpu_tensor *input,
                                         uint32_t elements) {
    return copy_cast(gpu, output, 0, input, 0, elements);
}

extern "C" int h3_gpu_copy_bf16(h3_gpu *gpu, h3_gpu_tensor *destination,
                                  size_t destination_offset,
                                  const h3_gpu_tensor *source,
                                  size_t source_offset, size_t elements) {
    return copy_cast(gpu, destination, destination_offset, source, source_offset,
                     elements);
}

extern "C" int h3_gpu_copy_f32(h3_gpu *gpu, h3_gpu_tensor *destination,
                                 size_t destination_offset,
                                 const h3_gpu_tensor *source,
                                 size_t source_offset, size_t elements) {
    return copy_cast(gpu, destination, destination_offset, source, source_offset,
                     elements);
}

__global__ static void binary_kernel(void *output, h3_gpu_dtype output_dtype,
                                     const void *left, h3_gpu_dtype left_dtype,
                                     const void *right, h3_gpu_dtype right_dtype,
                                     size_t elements, float left_scale,
                                     float right_scale) {
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += (size_t)gridDim.x * blockDim.x) {
        float value = left_scale * h3cspeed_device_load(left, left_dtype, index) +
                      right_scale * h3cspeed_device_load(right, right_dtype, index);
        h3cspeed_device_store(output, output_dtype, index, value);
    }
}

static int binary(h3_gpu *gpu, h3_gpu_tensor *output,
                  const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                  size_t elements, float left_scale, float right_scale) {
    if (!gpu || !output || !left || !right || output->elements < elements ||
        left->elements < elements || right->elements < elements ||
        !h3cspeed_tensor_wait(gpu, left) || !h3cspeed_tensor_wait(gpu, right)) return 0;
    binary_kernel<<<h3cspeed_blocks(elements), 256, 0, gpu->compute_stream>>>(
        output->data, output->dtype, left->data, left->dtype,
        right->data, right->dtype, elements, left_scale, right_scale);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "binary CUDA kernel");
}

extern "C" int h3_gpu_add_scaled_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                       const h3_gpu_tensor *left,
                                       const h3_gpu_tensor *right,
                                       float left_scale, float right_scale,
                                       uint32_t elements) {
    return binary(gpu, output, left, right, elements, left_scale, right_scale);
}

extern "C" int h3_gpu_add_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                 const h3_gpu_tensor *left,
                                 const h3_gpu_tensor *right,
                                 uint32_t elements) {
    return binary(gpu, output, left, right, elements, 1.0f, 1.0f);
}

extern "C" int h3_gpu_sub_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                 const h3_gpu_tensor *left,
                                 const h3_gpu_tensor *right,
                                 uint32_t elements) {
    return binary(gpu, output, left, right, elements, 1.0f, -1.0f);
}

__global__ static void swiglu_kernel(void *output, h3_gpu_dtype output_dtype,
                                     const void *fused, h3_gpu_dtype fused_dtype,
                                     uint32_t rows, uint32_t width) {
    size_t total = (size_t)rows * width;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += (size_t)gridDim.x * blockDim.x) {
        uint32_t row = (uint32_t)(index / width);
        uint32_t column = (uint32_t)(index % width);
        size_t base = (size_t)row * width * 2;
        float gate = h3cspeed_device_load(fused, fused_dtype, base + column);
        float up = h3cspeed_device_load(fused, fused_dtype, base + width + column);
        float value = (gate / (1.0f + expf(-gate))) * up;
        h3cspeed_device_store(output, output_dtype, index, value);
    }
}

extern "C" int h3_gpu_swiglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                   const h3_gpu_tensor *fused, uint32_t rows,
                                   uint32_t width) {
    if (!gpu || !output || !fused || output->elements < (size_t)rows * width ||
        fused->elements < (size_t)rows * width * 2 ||
        !h3cspeed_tensor_wait(gpu, fused)) return 0;
    swiglu_kernel<<<h3cspeed_blocks((size_t)rows * width), 256, 0,
                     gpu->compute_stream>>>(output->data, output->dtype,
                                            fused->data, fused->dtype,
                                            rows, width);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "SwiGLU CUDA kernel");
}

extern "C" int h3_gpu_swiglu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                    const h3_gpu_tensor *fused, uint32_t rows,
                                    uint32_t width) {
    return h3_gpu_swiglu_f32(gpu, output, fused, rows, width);
}

__global__ static void silu_mul_kernel(void *output, h3_gpu_dtype output_dtype,
                                       const void *gate, h3_gpu_dtype gate_dtype,
                                       const void *up, h3_gpu_dtype up_dtype,
                                       size_t elements) {
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += (size_t)gridDim.x * blockDim.x) {
        float g = h3cspeed_device_load(gate, gate_dtype, index);
        float value = (g / (1.0f + expf(-g))) *
                      h3cspeed_device_load(up, up_dtype, index);
        h3cspeed_device_store(output, output_dtype, index, value);
    }
}

extern "C" int h3_gpu_silu_mul_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                      const h3_gpu_tensor *gate,
                                      const h3_gpu_tensor *up,
                                      uint32_t elements) {
    if (!gpu || !output || !gate || !up || output->elements < elements ||
        gate->elements < elements || up->elements < elements ||
        !h3cspeed_tensor_wait(gpu, gate) || !h3cspeed_tensor_wait(gpu, up)) return 0;
    silu_mul_kernel<<<h3cspeed_blocks(elements), 256, 0, gpu->compute_stream>>>(
        output->data, output->dtype, gate->data, gate->dtype,
        up->data, up->dtype, elements);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "SiLU multiply CUDA kernel");
}

__global__ static void geglu_kernel(void *output, h3_gpu_dtype output_dtype,
                                    const void *gate, h3_gpu_dtype gate_dtype,
                                    const void *linear, h3_gpu_dtype linear_dtype,
                                    size_t elements) {
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += (size_t)gridDim.x * blockDim.x) {
        float value = h3cspeed_device_load(gate, gate_dtype, index);
        value = 0.5f * value * (1.0f + erff(value * 0.7071067811865475f));
        value *= h3cspeed_device_load(linear, linear_dtype, index);
        h3cspeed_device_store(output, output_dtype, index, value);
    }
}

extern "C" int h3_gpu_geglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *gate,
                                  const h3_gpu_tensor *linear,
                                  uint32_t elements) {
    if (!gpu || !output || !gate || !linear || output->elements < elements ||
        gate->elements < elements || linear->elements < elements ||
        !h3cspeed_tensor_wait(gpu, gate) || !h3cspeed_tensor_wait(gpu, linear)) return 0;
    geglu_kernel<<<h3cspeed_blocks(elements), 256, 0, gpu->compute_stream>>>(
        output->data, output->dtype, gate->data, gate->dtype,
        linear->data, linear->dtype, elements);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "GEGLU CUDA kernel");
}

extern "C" int h3_gpu_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                    const h3_gpu_tensor *input,
                                    const h3_gpu_tensor *weight,
                                    const h3_gpu_tensor *bias, uint32_t rows,
                                    uint32_t input_dim, uint32_t output_dim) {
    return h3cspeed_linear(gpu, output, 0, input, 0, weight, bias,
                          rows, input_dim, output_dim);
}

extern "C" int h3_gpu_mlp_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *fc1_weight,
                                 const h3_gpu_tensor *fc2_weight,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t hidden_dim, uint32_t output_dim) {
    h3_gpu_tensor *fused = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * hidden_dim * 2);
    h3_gpu_tensor *activated = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * hidden_dim);
    if (!fused || !activated) {
        h3_gpu_tensor_free(fused);
        h3_gpu_tensor_free(activated);
        return 0;
    }
    int ok = h3_gpu_linear_bf16(gpu, fused, input, fc1_weight, nullptr,
                                rows, input_dim, hidden_dim * 2) &&
             h3_gpu_swiglu_bf16(gpu, activated, fused, rows, hidden_dim) &&
             h3_gpu_linear_bf16(gpu, output, activated, fc2_weight, nullptr,
                                rows, hidden_dim, output_dim);
    h3_gpu_tensor_free(fused);
    h3_gpu_tensor_free(activated);
    return ok;
}

extern "C" int h3_gpu_mlp_nax_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                     h3_gpu_tensor *activated,
                                     const h3_gpu_tensor *input,
                                     const h3_gpu_tensor *fc1_weight,
                                     const h3_gpu_tensor *fc2_weight,
                                     uint32_t rows, uint32_t input_dim,
                                     uint32_t hidden_dim, uint32_t output_dim) {
    h3_gpu_tensor *fused = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * hidden_dim * 2);
    if (!fused || !activated) {
        h3_gpu_tensor_free(fused);
        return 0;
    }
    int ok = h3_gpu_linear_bf16(gpu, fused, input, fc1_weight, nullptr,
                                rows, input_dim, hidden_dim * 2) &&
             h3_gpu_swiglu_bf16(gpu, activated, fused, rows, hidden_dim) &&
             h3_gpu_linear_bf16(gpu, output, activated, fc2_weight, nullptr,
                                rows, hidden_dim, output_dim);
    h3_gpu_tensor_free(fused);
    return ok;
}

__global__ static void norm_kernel(
        void *output, h3_gpu_dtype output_dtype,
        const void *input, h3_gpu_dtype input_dtype,
        const void *weight, h3_gpu_dtype weight_dtype,
        const void *bias, h3_gpu_dtype bias_dtype,
        uint32_t rows, uint32_t width, float epsilon, int layer_norm) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float shared[];
    float sum = 0.0f;
    float square = 0.0f;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        float value = h3cspeed_device_load(input, input_dtype,
                                            (size_t)row * width + column);
        sum += value;
        square = fmaf(value, value, square);
    }
    shared[threadIdx.x] = sum;
    shared[blockDim.x + threadIdx.x] = square;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
            shared[blockDim.x + threadIdx.x] +=
                shared[blockDim.x + threadIdx.x + stride];
        }
        __syncthreads();
    }
    float mean = layer_norm ? shared[0] / (float)width : 0.0f;
    float variance;
    if (layer_norm) {
        /* Match the Metal reference's two-pass LayerNorm reduction.  The
         * E[x^2] - E[x]^2 form loses precision when the row has a large
         * nonzero mean, which can destabilize the VAE output projection. */
        float centered_square = 0.0f;
        for (uint32_t column = threadIdx.x; column < width;
             column += blockDim.x) {
            float centered = h3cspeed_device_load(
                input, input_dtype, (size_t)row * width + column) - mean;
            centered_square = fmaf(centered, centered, centered_square);
        }
        shared[blockDim.x + threadIdx.x] = centered_square;
        __syncthreads();
        for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
            if (threadIdx.x < stride)
                shared[blockDim.x + threadIdx.x] +=
                    shared[blockDim.x + threadIdx.x + stride];
            __syncthreads();
        }
        variance = shared[blockDim.x] / (float)width;
    } else {
        variance = shared[blockDim.x] / (float)width;
    }
    if (layer_norm) variance = fmaxf(variance, 0.0f);
    float inverse = rsqrtf(variance + epsilon);
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        float value = h3cspeed_device_load(input, input_dtype,
                                            (size_t)row * width + column);
        value = (value - mean) * inverse;
        if (weight) value *= h3cspeed_device_load(weight, weight_dtype, column);
        if (bias) value += h3cspeed_device_load(bias, bias_dtype, column);
        h3cspeed_device_store(output, output_dtype,
                              (size_t)row * width + column, value);
    }
}

static int norm_dispatch(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight,
                         const h3_gpu_tensor *bias, uint32_t rows,
                         uint32_t width, float epsilon, int layer_norm) {
    size_t elements = 0;
    if (!gpu || !output || !input || !weight || !rows || !width ||
        !(epsilon > 0.0f) || !std::isfinite(epsilon) ||
        (layer_norm && !bias) ||
        !h3cspeed_size_mul(rows, width, &elements) ||
        output->elements < elements || input->elements < elements ||
        (weight && weight->elements < width) || (bias && bias->elements < width) ||
        !h3cspeed_tensor_wait(gpu, input) ||
        (weight && !h3cspeed_tensor_wait(gpu, weight)) ||
        (bias && !h3cspeed_tensor_wait(gpu, bias))) return 0;
    unsigned threads = 256;
    norm_kernel<<<rows, threads, threads * 2 * sizeof(float),
                  gpu->compute_stream>>>(
        output->data, output->dtype, input->data, input->dtype,
        weight ? weight->data : nullptr, weight ? weight->dtype : H3_GPU_F32,
        bias ? bias->data : nullptr, bias ? bias->dtype : H3_GPU_F32,
        rows, width, epsilon, layer_norm);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "normalization CUDA kernel");
}

extern "C" int h3_gpu_rms_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                     const h3_gpu_tensor *input,
                                     const h3_gpu_tensor *weight,
                                     uint32_t rows, uint32_t width,
                                     float epsilon) {
    return norm_dispatch(gpu, output, input, weight, nullptr,
                         rows, width, epsilon, 0);
}

extern "C" int h3_gpu_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                      const h3_gpu_tensor *input,
                                      const h3_gpu_tensor *weight,
                                      uint32_t rows, uint32_t width,
                                      float epsilon) {
    return norm_dispatch(gpu, output, input, weight, nullptr,
                         rows, width, epsilon, 0);
}

extern "C" int h3_gpu_layer_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                       const h3_gpu_tensor *input,
                                       const h3_gpu_tensor *weight,
                                       const h3_gpu_tensor *bias,
                                       uint32_t rows, uint32_t width,
                                       float epsilon) {
    return norm_dispatch(gpu, output, input, weight, bias,
                         rows, width, epsilon, 1);
}

extern "C" int h3_gpu_layer_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                        const h3_gpu_tensor *input,
                                        const h3_gpu_tensor *weight,
                                        const h3_gpu_tensor *bias,
                                        uint32_t rows, uint32_t width,
                                        float epsilon) {
    return norm_dispatch(gpu, output, input, weight, bias,
                         rows, width, epsilon, 1);
}

__global__ static void adaln_kernel(
        void *output, h3_gpu_dtype output_dtype,
        float *inverse_output,
        const void *input, h3_gpu_dtype input_dtype, size_t input_offset,
        const void *norm_weight, h3_gpu_dtype weight_dtype,
        const void *modulation, h3_gpu_dtype modulation_dtype,
        const uint32_t *row_map, uint32_t rows, uint32_t width,
        uint32_t slots, uint32_t modulation_rows, uint32_t shift_slot,
        uint32_t scale_slot, float epsilon) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float shared[];
    float square = 0.0f;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        float value = h3cspeed_device_load(input, input_dtype,
            input_offset + (size_t)row * width + column);
        square += value * value;
    }
    shared[threadIdx.x] = square;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    float inverse_value = rsqrtf(shared[0] / (float)width + epsilon);
    if (inverse_output && threadIdx.x == 0) {
        inverse_output[row] = inverse_value;
    }
    uint32_t mapped = row_map ? row_map[row] : row;
    if (mapped >= modulation_rows) {
        for (uint32_t column = threadIdx.x; column < width;
             column += blockDim.x) {
            h3cspeed_device_store(output, output_dtype,
                                  (size_t)row * width + column, 0.0f);
        }
        return;
    }
    size_t modulation_base = (size_t)mapped * slots * width;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        float value = h3cspeed_device_load(input, input_dtype,
            input_offset + (size_t)row * width + column) * inverse_value;
        value *= h3cspeed_device_load(norm_weight, weight_dtype, column);
        float shift = h3cspeed_device_load(modulation, modulation_dtype,
            modulation_base + (size_t)shift_slot * width + column);
        float scale = h3cspeed_device_load(modulation, modulation_dtype,
            modulation_base + (size_t)scale_slot * width + column);
        value = value * (1.0f + scale) + shift;
        h3cspeed_device_store(output, output_dtype,
                              (size_t)row * width + column, value);
    }
}

static int adaln_dispatch(h3_gpu *gpu, h3_gpu_tensor *output,
                          h3_gpu_tensor *inverse,
                          const h3_gpu_tensor *input, size_t input_offset,
                          const h3_gpu_tensor *norm_weight,
                          const h3_gpu_tensor *modulation,
                          const h3_gpu_tensor *row_map, uint32_t rows,
                          uint32_t width, uint32_t slots,
                          uint32_t shift_slot, uint32_t scale_slot,
                          float epsilon) {
    size_t elements = 0;
    size_t slot_width = 0;
    if (!gpu || !output || !input || !norm_weight || !modulation ||
        !rows || !width || !slots || shift_slot >= slots ||
        scale_slot >= slots || !(epsilon > 0.0f) ||
        !std::isfinite(epsilon) ||
        !h3cspeed_size_mul(rows, width, &elements) ||
        !h3cspeed_size_mul(slots, width, &slot_width) || !slot_width ||
        output->elements < elements ||
        (inverse && (inverse->dtype != H3_GPU_F32 || inverse->elements < rows)) ||
        input_offset > input->elements ||
        elements > input->elements - input_offset ||
        norm_weight->elements < width ||
        modulation->elements < slot_width ||
        (row_map && (row_map->dtype != H3_GPU_U32 || row_map->elements < rows)) ||
        (inverse && !h3cspeed_tensor_wait(gpu, inverse)) ||
        !h3cspeed_tensor_wait(gpu, input) || !h3cspeed_tensor_wait(gpu, norm_weight) ||
        !h3cspeed_tensor_wait(gpu, modulation) ||
        (row_map && !h3cspeed_tensor_wait(gpu, row_map))) return 0;
    size_t modulation_row_count = modulation->elements / slot_width;
    if (!modulation_row_count ||
        (!row_map && rows > modulation_row_count) ||
        (row_map && row_map->u32_max_valid &&
         (size_t)row_map->u32_max >= modulation_row_count)) return 0;
    uint32_t kernel_modulation_rows = modulation_row_count > UINT32_MAX ?
        UINT32_MAX : (uint32_t)modulation_row_count;
    adaln_kernel<<<rows, 256, 256 * sizeof(float), gpu->compute_stream>>>(
        output->data, output->dtype,
        inverse ? static_cast<float *>(inverse->data) : nullptr,
        input->data, input->dtype, input_offset,
        norm_weight->data, norm_weight->dtype,
        modulation->data, modulation->dtype,
        row_map ? static_cast<const uint32_t *>(row_map->data) : nullptr,
        rows, width, slots, kernel_modulation_rows, shift_slot, scale_slot,
        epsilon);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "AdaLN CUDA kernel");
}

extern "C" int h3_gpu_adaln_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *input,
                                  const h3_gpu_tensor *norm_weight,
                                  const h3_gpu_tensor *modulation,
                                  const h3_gpu_tensor *row_map,
                                  uint32_t rows, uint32_t width,
                                  uint32_t slots, uint32_t shift_slot,
                                  uint32_t scale_slot, float epsilon) {
    return adaln_dispatch(gpu, output, nullptr, input, 0, norm_weight, modulation,
                          row_map, rows, width, slots, shift_slot, scale_slot,
                          epsilon);
}

extern "C" int h3_gpu_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                   const h3_gpu_tensor *input,
                                   const h3_gpu_tensor *norm_weight,
                                   const h3_gpu_tensor *modulation,
                                   const h3_gpu_tensor *row_map,
                                   uint32_t rows, uint32_t width,
                                   uint32_t slots, uint32_t shift_slot,
                                   uint32_t scale_slot, float epsilon) {
    return adaln_dispatch(gpu, output, nullptr, input, 0, norm_weight, modulation,
                          row_map, rows, width, slots, shift_slot, scale_slot,
                          epsilon);
}

extern "C" int h3_gpu_adaln_bf16_offset(
        h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
        size_t input_offset, const h3_gpu_tensor *norm_weight,
        const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map,
        uint32_t rows, uint32_t width, uint32_t slots,
        uint32_t shift_slot, uint32_t scale_slot, float epsilon) {
    return adaln_dispatch(gpu, output, nullptr, input, input_offset, norm_weight,
                          modulation, row_map, rows, width, slots,
                          shift_slot, scale_slot, epsilon);
}

__global__ static void gate_kernel(
        void *output, h3_gpu_dtype output_dtype,
        const void *residual, h3_gpu_dtype residual_dtype,
        const void *branch, h3_gpu_dtype branch_dtype,
        const void *modulation, h3_gpu_dtype modulation_dtype,
        const uint32_t *row_map, uint32_t rows, uint32_t width,
        uint32_t slots, uint32_t modulation_rows, uint32_t gate_slot) {
    size_t total = (size_t)rows * width;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += (size_t)gridDim.x * blockDim.x) {
        uint32_t row = (uint32_t)(index / width);
        uint32_t column = (uint32_t)(index % width);
        uint32_t mapped = row_map ? row_map[row] : row;
        if (mapped >= modulation_rows) {
            h3cspeed_device_store(output, output_dtype, index, 0.0f);
            continue;
        }
        float gate = h3cspeed_device_load(modulation, modulation_dtype,
            ((size_t)mapped * slots + gate_slot) * width + column);
        float value = h3cspeed_device_load(residual, residual_dtype, index) +
                      gate * h3cspeed_device_load(branch, branch_dtype, index);
        h3cspeed_device_store(output, output_dtype, index, value);
    }
}

static int gate_dispatch(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *residual,
                         const h3_gpu_tensor *branch,
                         const h3_gpu_tensor *modulation,
                         const h3_gpu_tensor *row_map, uint32_t rows,
                         uint32_t width, uint32_t slots, uint32_t gate_slot) {
    size_t elements = 0;
    size_t slot_width = 0;
    if (!gpu || !output || !residual || !branch || !modulation ||
        !rows || !width || !slots || gate_slot >= slots ||
        !h3cspeed_size_mul(rows, width, &elements) ||
        !h3cspeed_size_mul(slots, width, &slot_width) || !slot_width ||
        output->elements < elements || residual->elements < elements ||
        branch->elements < elements || modulation->elements < slot_width ||
        (row_map && (row_map->dtype != H3_GPU_U32 || row_map->elements < rows)) ||
        !h3cspeed_tensor_wait(gpu, residual) || !h3cspeed_tensor_wait(gpu, branch) ||
        !h3cspeed_tensor_wait(gpu, modulation) ||
        (row_map && !h3cspeed_tensor_wait(gpu, row_map))) return 0;
    size_t modulation_row_count = modulation->elements / slot_width;
    if (!modulation_row_count ||
        (!row_map && rows > modulation_row_count) ||
        (row_map && row_map->u32_max_valid &&
         (size_t)row_map->u32_max >= modulation_row_count)) return 0;
    uint32_t kernel_modulation_rows = modulation_row_count > UINT32_MAX ?
        UINT32_MAX : (uint32_t)modulation_row_count;
    gate_kernel<<<h3cspeed_blocks(elements), 256, 0, gpu->compute_stream>>>(
        output->data, output->dtype, residual->data, residual->dtype,
        branch->data, branch->dtype, modulation->data, modulation->dtype,
        row_map ? static_cast<const uint32_t *>(row_map->data) : nullptr,
        rows, width, slots, kernel_modulation_rows, gate_slot);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "gate CUDA kernel");
}

extern "C" int h3_gpu_gate_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                 const h3_gpu_tensor *residual,
                                 const h3_gpu_tensor *branch,
                                 const h3_gpu_tensor *modulation,
                                 const h3_gpu_tensor *row_map,
                                 uint32_t rows, uint32_t width,
                                 uint32_t slots, uint32_t gate_slot) {
    return gate_dispatch(gpu, output, residual, branch, modulation,
                         row_map, rows, width, slots, gate_slot);
}

extern "C" int h3_gpu_gate_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *residual,
                                  const h3_gpu_tensor *branch,
                                  const h3_gpu_tensor *modulation,
                                  const h3_gpu_tensor *row_map,
                                  uint32_t rows, uint32_t width,
                                  uint32_t slots, uint32_t gate_slot) {
    return gate_dispatch(gpu, output, residual, branch, modulation,
                         row_map, rows, width, slots, gate_slot);
}

extern "C" int h3_gpu_gate_adaln_bf16(
        h3_gpu *gpu, h3_gpu_tensor *gated_residual, h3_gpu_tensor *output,
        const h3_gpu_tensor *residual, const h3_gpu_tensor *branch,
        const h3_gpu_tensor *norm_weight,
        const h3_gpu_tensor *gate_modulation,
        const h3_gpu_tensor *norm_modulation,
        const h3_gpu_tensor *row_map, uint32_t rows, uint32_t width,
        uint32_t slots, uint32_t gate_slot, uint32_t shift_slot,
        uint32_t scale_slot, float epsilon) {
    return gate_dispatch(gpu, gated_residual, residual, branch,
                         gate_modulation, row_map, rows, width, slots,
                         gate_slot) &&
           adaln_dispatch(gpu, output, nullptr, gated_residual, 0, norm_weight,
                          norm_modulation, row_map, rows, width, slots,
                          shift_slot, scale_slot, epsilon);
}

extern "C" int h3_gpu_adaln_linear_bf16(
        h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *inverse,
        const h3_gpu_tensor *input, size_t input_offset,
        const h3_gpu_tensor *norm_weight,
        const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map,
        const h3_gpu_tensor *weight, const h3_gpu_tensor *bias,
        uint32_t rows, uint32_t width, uint32_t output_dim,
        uint32_t slots, uint32_t shift_slot, uint32_t scale_slot,
        float epsilon) {
    size_t normalized_elements = 0;
    size_t output_elements = 0;
    size_t input_elements = 0;
    size_t weight_elements = 0;
    size_t modulation_elements = 0;
    if (!gpu || !output || !inverse || !input || !norm_weight ||
        !modulation || !row_map || !weight || !rows || !width ||
        !output_dim || !slots || shift_slot >= slots ||
        scale_slot >= slots || !h3cspeed_size_mul(rows, width,
                                                  &normalized_elements) ||
        !h3cspeed_size_mul(rows, output_dim, &output_elements) ||
        !h3cspeed_size_mul(output_dim, width, &weight_elements) ||
        !h3cspeed_size_mul(slots, width, &modulation_elements) ||
        input_offset > input->elements ||
        !h3cspeed_size_mul(rows, width, &input_elements) ||
        input_elements > input->elements - input_offset) {
        h3cspeed_set_error(gpu, "fused AdaLN/head",
                           "invalid rows, dimensions, offset, or tensor arguments");
        return 0;
    }
    if (!(epsilon > 0.0f) || !std::isfinite(epsilon)) {
        h3cspeed_set_error(gpu, "fused AdaLN/head",
                           "epsilon must be finite and greater than zero");
        return 0;
    }
    if (inverse->dtype != H3_GPU_F32 || inverse->elements < rows) {
        char detail[256];
        snprintf(detail, sizeof(detail),
                 "inverse RMS must be F32 with at least %u row values "
                 "(dtype=%s elements=%zu)", rows,
                 h3cspeed_dtype_name(inverse->dtype), inverse->elements);
        h3cspeed_set_error(gpu, "fused AdaLN/head", detail);
        return 0;
    }
    if (output->dtype != H3_GPU_BF16 || output->elements < output_elements ||
        input->dtype != H3_GPU_BF16 || norm_weight->dtype != H3_GPU_BF16 ||
        norm_weight->elements < width || modulation->dtype != H3_GPU_BF16 ||
        modulation->elements < modulation_elements || row_map->dtype != H3_GPU_U32 ||
        row_map->elements < rows || weight->dtype != H3_GPU_BF16 ||
        weight->elements < weight_elements ||
        (bias && (bias->dtype != H3_GPU_BF16 || bias->elements < output_dim))) {
        h3cspeed_set_error(gpu, "fused AdaLN/head",
                           "tensor dtype or element count does not match BF16 final head");
        return 0;
    }
    size_t modulation_rows = modulation->elements / modulation_elements;
    if (!modulation_rows ||
        (row_map->u32_max_valid &&
         (size_t)row_map->u32_max >= modulation_rows)) {
        h3cspeed_set_error(gpu, "fused AdaLN/head",
                           "row map references modulation rows outside the tensor");
        return 0;
    }
    h3_gpu_tensor *normalized = h3_gpu_tensor_new_bf16(gpu, normalized_elements);
    if (!normalized) return 0;
    int ok = adaln_dispatch(gpu, normalized, inverse, input, input_offset, norm_weight,
                            modulation, row_map, rows, width, slots,
                            shift_slot, scale_slot, epsilon) &&
             h3cspeed_linear(gpu, output, 0, normalized, 0, weight, bias,
                             rows, width, output_dim);
    h3_gpu_tensor_free(normalized);
    return ok;
}

__global__ static void scale_add_kernel(
        void *output, h3_gpu_dtype output_dtype,
        const void *residual, h3_gpu_dtype residual_dtype,
        const void *branch, h3_gpu_dtype branch_dtype,
        const void *scale, h3_gpu_dtype scale_dtype,
        uint32_t rows, uint32_t width) {
    size_t total = (size_t)rows * width;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += (size_t)gridDim.x * blockDim.x) {
        uint32_t column = (uint32_t)(index % width);
        float value = h3cspeed_device_load(residual, residual_dtype, index) +
            h3cspeed_device_load(scale, scale_dtype, column) *
            h3cspeed_device_load(branch, branch_dtype, index);
        h3cspeed_device_store(output, output_dtype, index, value);
    }
}

extern "C" int h3_gpu_scale_add_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                      const h3_gpu_tensor *residual,
                                      const h3_gpu_tensor *branch,
                                      const h3_gpu_tensor *scale,
                                      uint32_t rows, uint32_t width) {
    size_t elements = (size_t)rows * width;
    if (!gpu || !output || !residual || !branch || !scale || !rows || !width ||
        output->elements < elements || residual->elements < elements ||
        branch->elements < elements || scale->elements < width) {
        if (gpu) h3cspeed_set_error(
            gpu, "scale-add CUDA kernel",
            "residual/branch/output require rows*width elements and scale "
            "requires at least width elements");
        return 0;
    }
    if (!h3cspeed_tensor_wait(gpu, residual) ||
        !h3cspeed_tensor_wait(gpu, branch) ||
        !h3cspeed_tensor_wait(gpu, scale)) return 0;
    scale_add_kernel<<<h3cspeed_blocks(elements), 256, 0, gpu->compute_stream>>>(
        output->data, output->dtype, residual->data, residual->dtype,
        branch->data, branch->dtype, scale->data, scale->dtype,
        rows, width);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "scale-add CUDA kernel");
}

__global__ static void qkv_rope_kernel(
        void *query, h3_gpu_dtype query_dtype,
        void *key, h3_gpu_dtype key_dtype,
        void *value, h3_gpu_dtype value_dtype,
        const void *qkv, h3_gpu_dtype qkv_dtype,
        const void *q_norm, h3_gpu_dtype q_norm_dtype,
        const void *k_norm, h3_gpu_dtype k_norm_dtype,
        const void *rope_cos, h3_gpu_dtype cos_dtype,
        const void *rope_sin, h3_gpu_dtype sin_dtype,
        uint32_t sequence, uint32_t heads, uint32_t head_dim,
        uint32_t rope_half, float epsilon, int grouped, int apply_norm,
        int rms_only) {
    uint32_t row = blockIdx.x;
    uint32_t head = blockIdx.y;
    if (row >= sequence || head >= heads) return;
    extern __shared__ float shared[];
    float q_square = 0.0f;
    float k_square = 0.0f;
    size_t width = (size_t)heads * 3 * head_dim;
    for (uint32_t dimension = threadIdx.x; dimension < head_dim;
         dimension += blockDim.x) {
        size_t q_index = grouped ?
            (size_t)row * width + ((size_t)head * 3) * head_dim + dimension :
            (size_t)row * width + (size_t)head * head_dim + dimension;
        size_t k_index = grouped ?
            (size_t)row * width + ((size_t)head * 3 + 1) * head_dim + dimension :
            (size_t)row * width + ((size_t)heads + head) * head_dim + dimension;
        float q = h3cspeed_device_load(qkv, qkv_dtype, q_index);
        float k = h3cspeed_device_load(qkv, qkv_dtype, k_index);
        q_square = fmaf(q, q, q_square);
        k_square = fmaf(k, k, k_square);
    }
    shared[threadIdx.x] = q_square;
    shared[blockDim.x + threadIdx.x] = k_square;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
            shared[blockDim.x + threadIdx.x] +=
                shared[blockDim.x + threadIdx.x + stride];
        }
        __syncthreads();
    }
    int normalize = apply_norm || rms_only;
    float q_inverse = normalize ?
        rsqrtf(shared[0] / (float)head_dim + epsilon) : 1.0f;
    float k_inverse = normalize ?
        rsqrtf(shared[blockDim.x] / (float)head_dim + epsilon) : 1.0f;
    for (uint32_t dimension = threadIdx.x; dimension < head_dim;
         dimension += blockDim.x) {
        size_t q_index = grouped ?
            (size_t)row * width + ((size_t)head * 3) * head_dim + dimension :
            (size_t)row * width + (size_t)head * head_dim + dimension;
        size_t k_index = grouped ?
            (size_t)row * width + ((size_t)head * 3 + 1) * head_dim + dimension :
            (size_t)row * width + ((size_t)heads + head) * head_dim + dimension;
        size_t v_index = grouped ?
            (size_t)row * width + ((size_t)head * 3 + 2) * head_dim + dimension :
            (size_t)row * width + ((size_t)heads * 2 + head) * head_dim + dimension;
        float q = h3cspeed_device_load(qkv, qkv_dtype, q_index) * q_inverse;
        float k = h3cspeed_device_load(qkv, qkv_dtype, k_index) * k_inverse;
        if (apply_norm) {
            q *= h3cspeed_device_load(q_norm, q_norm_dtype, dimension);
            k *= h3cspeed_device_load(k_norm, k_norm_dtype, dimension);
        }
        if (dimension < rope_half) {
            uint32_t paired = dimension + rope_half;
            size_t q_pair_index = grouped ?
                (size_t)row * width + ((size_t)head * 3) * head_dim + paired :
                (size_t)row * width + (size_t)head * head_dim + paired;
            size_t k_pair_index = grouped ?
                (size_t)row * width + ((size_t)head * 3 + 1) * head_dim + paired :
                (size_t)row * width + ((size_t)heads + head) * head_dim + paired;
            float q_pair = h3cspeed_device_load(qkv, qkv_dtype, q_pair_index) * q_inverse;
            float k_pair = h3cspeed_device_load(qkv, qkv_dtype, k_pair_index) * k_inverse;
            if (apply_norm) {
                q_pair *= h3cspeed_device_load(q_norm, q_norm_dtype, paired);
                k_pair *= h3cspeed_device_load(k_norm, k_norm_dtype, paired);
            }
            float cosine = h3cspeed_device_load(rope_cos, cos_dtype,
                                                 (size_t)row * rope_half + dimension);
            float sine = h3cspeed_device_load(rope_sin, sin_dtype,
                                               (size_t)row * rope_half + dimension);
            float q_first = q * cosine - q_pair * sine;
            float q_second = q_pair * cosine + q * sine;
            float k_first = k * cosine - k_pair * sine;
            float k_second = k_pair * cosine + k * sine;
            /* The checkpoint/video input is row-major interleaved [Q,K,V].
             * CUDA SDPA consumes its query/key/value tensors in the existing
             * head-major [head,row,dimension] view, so only the input is
             * deinterleaved here; the backend's output contract is unchanged. */
            size_t destination = ((size_t)head * sequence + row) * head_dim;
            h3cspeed_device_store(query, query_dtype, destination + dimension, q_first);
            h3cspeed_device_store(query, query_dtype, destination + paired, q_second);
            h3cspeed_device_store(key, key_dtype, destination + dimension, k_first);
            h3cspeed_device_store(key, key_dtype, destination + paired, k_second);
        } else if (dimension >= rope_half * 2) {
            size_t destination = ((size_t)head * sequence + row) * head_dim + dimension;
            h3cspeed_device_store(query, query_dtype, destination, q);
            h3cspeed_device_store(key, key_dtype, destination, k);
        }
        size_t destination = ((size_t)head * sequence + row) * head_dim + dimension;
        float v = h3cspeed_device_load(qkv, qkv_dtype, v_index);
        h3cspeed_device_store(value, value_dtype, destination, v);
    }
}

static int qkv_rope_dispatch(h3_gpu *gpu, h3_gpu_tensor *query,
                             h3_gpu_tensor *key, h3_gpu_tensor *value,
                             const h3_gpu_tensor *qkv,
                             const h3_gpu_tensor *q_norm,
                             const h3_gpu_tensor *k_norm,
                             const h3_gpu_tensor *rope_cos,
                             const h3_gpu_tensor *rope_sin,
                             uint32_t sequence, uint32_t heads,
                             uint32_t head_dim, uint32_t rope_half,
                             float epsilon, int grouped, int apply_norm,
                             int rms_only) {
    size_t elements = (size_t)sequence * heads * head_dim;
    if (!gpu || !query || !key || !value || !qkv || !rope_cos || !rope_sin ||
        !sequence || !heads || !head_dim || rope_half > head_dim / 2 ||
        query->elements < elements || key->elements < elements ||
        value->elements < elements || qkv->elements < elements * 3 ||
        rope_cos->elements < (size_t)sequence * rope_half ||
        rope_sin->elements < (size_t)sequence * rope_half ||
        (apply_norm && (!q_norm || !k_norm || q_norm->elements < head_dim ||
                        k_norm->elements < head_dim)) ||
        !h3cspeed_tensor_wait(gpu, qkv) ||
        !h3cspeed_tensor_wait(gpu, rope_cos) ||
        !h3cspeed_tensor_wait(gpu, rope_sin) ||
        (apply_norm && (!h3cspeed_tensor_wait(gpu, q_norm) ||
                        !h3cspeed_tensor_wait(gpu, k_norm)))) return 0;
    dim3 grid(sequence, heads);
    qkv_rope_kernel<<<grid, 128, 256 * sizeof(float), gpu->compute_stream>>>(
        query->data, query->dtype, key->data, key->dtype,
        value->data, value->dtype, qkv->data, qkv->dtype,
        q_norm ? q_norm->data : nullptr, q_norm ? q_norm->dtype : H3_GPU_F32,
        k_norm ? k_norm->data : nullptr, k_norm ? k_norm->dtype : H3_GPU_F32,
        rope_cos->data, rope_cos->dtype, rope_sin->data, rope_sin->dtype,
        sequence, heads, head_dim, rope_half, epsilon, grouped, apply_norm,
        rms_only);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "QKV/RoPE CUDA kernel");
}

extern "C" int h3_gpu_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                                     h3_gpu_tensor *key, h3_gpu_tensor *value,
                                     const h3_gpu_tensor *qkv,
                                     const h3_gpu_tensor *q_norm,
                                     const h3_gpu_tensor *k_norm,
                                     const h3_gpu_tensor *rope_cos,
                                     const h3_gpu_tensor *rope_sin,
                                     uint32_t sequence, uint32_t heads,
                                     uint32_t head_dim, uint32_t rope_half,
                                     float epsilon) {
    return qkv_rope_dispatch(gpu, query, key, value, qkv, q_norm, k_norm,
                             rope_cos, rope_sin, sequence, heads, head_dim,
                             rope_half, epsilon, 0, 1, 0);
}

extern "C" int h3_gpu_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                                      h3_gpu_tensor *key, h3_gpu_tensor *value,
                                      const h3_gpu_tensor *qkv,
                                      const h3_gpu_tensor *q_norm,
                                      const h3_gpu_tensor *k_norm,
                                      const h3_gpu_tensor *rope_cos,
                                      const h3_gpu_tensor *rope_sin,
                                      uint32_t sequence, uint32_t heads,
                                      uint32_t head_dim, uint32_t rope_half,
                                      float epsilon) {
    return qkv_rope_dispatch(gpu, query, key, value, qkv, q_norm, k_norm,
                             rope_cos, rope_sin, sequence, heads, head_dim,
                             rope_half, epsilon, 0, 1, 0);
}

extern "C" int h3_gpu_grouped_qkv_rope_bf16(
        h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
        h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
        const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm,
        const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
        uint32_t sequence, uint32_t heads, uint32_t head_dim,
        uint32_t rope_half, float epsilon) {
    return qkv_rope_dispatch(gpu, query, key, value, qkv, q_norm, k_norm,
                             rope_cos, rope_sin, sequence, heads, head_dim,
                             rope_half, epsilon, 1, 1, 0);
}

extern "C" int h3_gpu_vision_qkv_rope_bf16(
        h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
        h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
        const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
        uint32_t sequence, uint32_t heads, uint32_t head_dim,
        uint32_t rope_half) {
    return qkv_rope_dispatch(gpu, query, key, value, qkv, nullptr, nullptr,
                             rope_cos, rope_sin, sequence, heads, head_dim,
                             rope_half, 0.0f, 0, 0, 0);
}

extern "C" int h3_gpu_video_qkv_rope_f32(
        h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
        h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
        const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
        uint32_t sequence, uint32_t heads, uint32_t head_dim,
        uint32_t rope_half, float epsilon) {
    return qkv_rope_dispatch(gpu, query, key, value, qkv, nullptr, nullptr,
                             rope_cos, rope_sin, sequence, heads, head_dim,
                             rope_half, epsilon, 1, 0, 1);
}

extern "C" int h3_gpu_grouped_qkv_linear_rope_bf16(
        h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
        h3_gpu_tensor *value, h3_gpu_tensor *qkv,
        const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
        const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm,
        const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
        uint32_t rows, uint32_t input_dim, uint32_t heads,
        uint32_t head_dim, uint32_t rope_half, float epsilon) {
    uint32_t qkv_dim = heads * head_dim * 3;
    return h3cspeed_linear(gpu, qkv, 0, input, 0, weight, nullptr,
                           rows, input_dim, qkv_dim) &&
           h3_gpu_grouped_qkv_rope_bf16(gpu, query, key, value, qkv,
                                        q_norm, k_norm, rope_cos, rope_sin,
                                        rows, heads, head_dim, rope_half,
                                        epsilon);
}

__global__ static void embedding_kernel(
        void *output, h3_gpu_dtype output_dtype,
        const void *weight, h3_gpu_dtype weight_dtype,
        const uint32_t *ids, uint32_t tokens, uint32_t vocab_size,
        uint32_t width) {
    size_t total = (size_t)tokens * width;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += (size_t)gridDim.x * blockDim.x) {
        uint32_t token = (uint32_t)(index / width);
        uint32_t column = (uint32_t)(index % width);
        uint32_t identifier = ids[token];
        float value = identifier < vocab_size ?
            h3cspeed_device_load(weight, weight_dtype,
                                 (size_t)identifier * width + column) : 0.0f;
        h3cspeed_device_store(output, output_dtype, index, value);
    }
}

extern "C" int h3_gpu_embedding_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                       const h3_gpu_tensor *weight,
                                       const h3_gpu_tensor *token_ids,
                                       uint32_t tokens, uint32_t vocab_size,
                                       uint32_t width) {
    size_t elements = (size_t)tokens * width;
    if (!gpu || !output || !weight || !token_ids ||
        token_ids->dtype != H3_GPU_U32 || output->elements < elements ||
        token_ids->elements < tokens ||
        weight->elements < (size_t)vocab_size * width ||
        !h3cspeed_tensor_wait(gpu, weight) ||
        !h3cspeed_tensor_wait(gpu, token_ids)) return 0;
    embedding_kernel<<<h3cspeed_blocks(elements), 256, 0, gpu->compute_stream>>>(
        output->data, output->dtype, weight->data, weight->dtype,
        static_cast<const uint32_t *>(token_ids->data), tokens, vocab_size,
        width);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "embedding CUDA kernel");
}

__global__ static void quantize_rows_kernel(
        int8_t *output, float *scales, const void *input,
        h3_gpu_dtype input_dtype, uint32_t rows, uint32_t width,
        int head_major, uint32_t heads, uint32_t head_dim) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float shared[];
    float maximum = 0.0f;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        size_t index;
        if (head_major) {
            uint32_t head = column / head_dim;
            uint32_t dimension = column % head_dim;
            index = ((size_t)head * rows + row) * head_dim + dimension;
        } else {
            index = (size_t)row * width + column;
        }
        maximum = fmaxf(maximum, fabsf(h3cspeed_device_load(input, input_dtype, index)));
    }
    shared[threadIdx.x] = maximum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        __syncthreads();
    }
    float scale = shared[0] > 0.0f ? shared[0] / 127.0f : 1.0f;
    if (threadIdx.x == 0) scales[row] = scale;
    float inverse = 1.0f / scale;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        size_t index;
        if (head_major) {
            uint32_t head = column / head_dim;
            uint32_t dimension = column % head_dim;
            index = ((size_t)head * rows + row) * head_dim + dimension;
        } else {
            index = (size_t)row * width + column;
        }
        float value = nearbyintf(h3cspeed_device_load(input, input_dtype, index) * inverse);
        value = fminf(127.0f, fmaxf(-127.0f, value));
        output[(size_t)row * width + column] = (int8_t)value;
    }
}

/* Apply the same regular Hadamard matrix used by comfy-kitchen ConvRot.
 * The 4x4 seed is Kronecker-expanded (H4 ^ n) and every stage is normalized
 * by 1/2, so a group of 4^n receives the orthogonal 1/sqrt(group) scaling.
 * Keeping this as butterflies avoids a 256x256 matrix allocation per row. */
__device__ static float *convrot_hadamard_group(float *values,
                                                uint32_t group_size,
                                                uint32_t lane) {
    float *source = values;
    float *destination = values + blockDim.x;
    for (uint32_t stride = 1; stride < group_size; stride *= 4) {
        if (lane < group_size) {
            uint32_t block = (lane / (4u * stride)) * (4u * stride);
            uint32_t offset = lane % stride;
            uint32_t base = block + offset;
            uint32_t which = (lane / stride) % 4u;
            float a = source[base];
            float b = source[base + stride];
            float c = source[base + 2u * stride];
            float d = source[base + 3u * stride];
            float result;
            switch (which) {
                case 0: result = a + b + c - d; break;
                case 1: result = a + b - c + d; break;
                case 2: result = a - b + c + d; break;
                default: result = -a + b + c + d; break;
            }
            destination[base + which * stride] = result * 0.5f;
        }
        __syncthreads();
        float *temporary = source;
        source = destination;
        destination = temporary;
    }
    return source;
}

__device__ static size_t quantize_input_index(uint32_t row,
                                               uint32_t column,
                                               uint32_t width,
                                               uint32_t rows,
                                               int head_major,
                                               uint32_t head_dim) {
    if (!head_major) return (size_t)row * width + column;
    uint32_t head = column / head_dim;
    uint32_t dimension = column % head_dim;
    return ((size_t)head * rows + (size_t)row) * head_dim + dimension;
}

/* ConvRot variant of row quantization.  Input is rotated in groups before the
 * existing per-row INT8 scale/rounding boundary.  The weight marker is the
 * only caller-visible selector; ordinary generated INT8 follows the legacy
 * kernel byte-for-byte. */
__global__ static void quantize_rows_convrot_kernel(
        int8_t *output, float *scales, const void *input,
        h3_gpu_dtype input_dtype, uint32_t rows, uint32_t width,
        int head_major, uint32_t heads, uint32_t head_dim,
        uint32_t group_size) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float shared[];
    uint32_t lane = threadIdx.x;
    float maximum = 0.0f;
    uint32_t groups = width / group_size;

    /* First pass computes the row scale from x @ H, without materializing a
     * rotated activation tensor. */
    for (uint32_t group = 0; group < groups; group++) {
        if (lane < group_size) {
            uint32_t column = group * group_size + lane;
            size_t index = quantize_input_index(row, column, width, rows,
                                                head_major, head_dim);
            shared[lane] = h3cspeed_device_load(input, input_dtype, index);
        }
        __syncthreads();
        float *rotated = convrot_hadamard_group(shared, group_size, lane);
        /* comfy-kitchen builds H in the activation dtype.  Its BF16 matmul
         * therefore rounds the rotated activation once before INT8 max/round;
         * mirror that boundary while retaining FP32 butterfly arithmetic. */
        if (lane < group_size && input_dtype == H3_GPU_BF16)
            rotated[lane] = __bfloat162float(__float2bfloat16_rn(rotated[lane]));
        __syncthreads();
        if (lane < group_size)
            maximum = fmaxf(maximum, fabsf(rotated[lane]));
        __syncthreads();
    }
    shared[lane] = maximum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
        if (lane < stride)
            shared[lane] = fmaxf(shared[lane], shared[lane + stride]);
        __syncthreads();
    }
    float scale = shared[0] > 0.0f ? shared[0] / 127.0f : 1.0f;
    if (lane == 0) scales[row] = scale;
    __syncthreads();

    /* Recompute the transform for the output pass.  This doubles reads but
     * bounds temporary storage to one group and keeps low-VRAM behavior. */
    float inverse = 1.0f / scale;
    for (uint32_t group = 0; group < groups; group++) {
        if (lane < group_size) {
            uint32_t column = group * group_size + lane;
            size_t index = quantize_input_index(row, column, width, rows,
                                                head_major, head_dim);
            shared[lane] = h3cspeed_device_load(input, input_dtype, index);
        }
        __syncthreads();
        float *rotated = convrot_hadamard_group(shared, group_size, lane);
        if (lane < group_size && input_dtype == H3_GPU_BF16)
            rotated[lane] = __bfloat162float(__float2bfloat16_rn(rotated[lane]));
        __syncthreads();
        if (lane < group_size) {
            float value = nearbyintf(rotated[lane] * inverse);
            value = fminf(127.0f, fmaxf(-127.0f, value));
            output[(size_t)row * width + group * group_size + lane] =
                (int8_t)value;
        }
        __syncthreads();
    }
}

static int convrot_group_valid(uint32_t group_size) {
    if (group_size < 4 || group_size > 256) return 0;
    /* ConvRot uses a Kronecker power of the 4x4 seed. */
    while (group_size > 1) {
        if (group_size % 4 != 0) return 0;
        group_size /= 4;
    }
    return 1;
}

static int h3cspeed_quantize_rows_convrot(
        h3_gpu *gpu, h3_gpu_tensor *quantized, h3_gpu_tensor *scales,
        const h3_gpu_tensor *input, uint32_t rows, uint32_t width,
        int head_major, uint32_t heads, uint32_t head_dim,
        uint32_t group_size) {
    if (!convrot_group_valid(group_size) || width % group_size != 0) {
        h3cspeed_set_error(gpu, "ConvRot INT8 quantization",
                           "group size must be a power of four <= 256 and divide input width");
        return 0;
    }
    if (!gpu || !quantized || !scales || !input ||
        quantized->dtype != H3_GPU_I8 || scales->dtype != H3_GPU_F32 ||
        quantized->elements < (size_t)rows * width || scales->elements < rows ||
        input->elements < (size_t)rows * width ||
        (head_major && (uint64_t)heads * head_dim != width) ||
        !h3cspeed_tensor_wait(gpu, input)) return 0;
    quantize_rows_convrot_kernel<<<rows, 256, 512 * sizeof(float),
                                   gpu->compute_stream>>>(
        static_cast<int8_t *>(quantized->data),
        static_cast<float *>(scales->data), input->data, input->dtype,
        rows, width, head_major, heads, head_dim, group_size);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "ConvRot row quantization CUDA kernel");
}

int h3cspeed_quantize_rows(h3_gpu *gpu, h3_gpu_tensor *quantized,
                           h3_gpu_tensor *scales, const h3_gpu_tensor *input,
                           uint32_t rows, uint32_t width, int head_major,
                           uint32_t heads, uint32_t head_dim) {
    if (!gpu || !quantized || !scales || !input ||
        quantized->dtype != H3_GPU_I8 || scales->dtype != H3_GPU_F32 ||
        quantized->elements < (size_t)rows * width || scales->elements < rows ||
        input->elements < (size_t)rows * width ||
        (head_major && (uint64_t)heads * head_dim != width) ||
        !h3cspeed_tensor_wait(gpu, input)) return 0;
    quantize_rows_kernel<<<rows, 256, 256 * sizeof(float), gpu->compute_stream>>>(
        static_cast<int8_t *>(quantized->data),
        static_cast<float *>(scales->data), input->data, input->dtype,
        rows, width, head_major, heads, head_dim);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "row quantization CUDA kernel");
}

static int make_generated_weight_offloadable(h3_gpu *gpu,
                                                h3_gpu_tensor *tensor,
                                                const char *label) {
    if (!gpu || !tensor) return 0;
    if (!gpu->offload.enabled || !tensor->bytes) return 1;

    /* Generated INT8 weights have no safetensors source to reread. Persist one
     * authoritative copy in system RAM before allowing their VRAM allocation
     * to enter the same LRU cache as file-backed BF16/F32 weights. */
    if (!h3cspeed_cuda_ok(gpu,
            profiled_stream_synchronize(
                gpu, gpu->compute_stream, H3CSPEED_PROFILE_STREAM_COMPUTE),
            "synchronize generated weight before RAM offload")) return 0;

    pthread_mutex_lock(&gpu->offload_lock);
    if (tensor->offloadable) {
        pthread_mutex_unlock(&gpu->offload_lock);
        return 1;
    }
    if (!tensor->data || !host_backing_allocate_locked(gpu, tensor)) {
        char detail[320];
        snprintf(detail, sizeof(detail),
                 "%s needs %.2f MiB of system RAM; increase "
                 "H3_CUDA_HOST_CACHE_MIB or assign more RAM to WSL2",
                 label ? label : "generated weight",
                 (double)tensor->bytes / (1024.0 * 1024.0));
        h3cspeed_set_error(gpu, "generated weight RAM offload", detail);
        pthread_mutex_unlock(&gpu->offload_lock);
        return 0;
    }
    if (!h3cspeed_cuda_ok(gpu,
            cudaMemcpy(tensor->host_data, tensor->data, tensor->bytes,
                       cudaMemcpyDeviceToHost),
            "copy generated weight to system RAM")) {
        host_backing_release_locked(gpu, tensor);
        pthread_mutex_unlock(&gpu->offload_lock);
        return 0;
    }
    tensor->host_valid = 1;
    tensor->source_bytes = tensor->bytes;
    tensor->source_offset = 0;
    tensor->source_streaming = 0;
    tensor->offloadable = 1;
    tensor->pin_epoch = 0;
    pthread_mutex_lock(&tensor->lock);
    tensor->ready_valid = 0;
    tensor->last_use_valid = 0;
    pthread_mutex_unlock(&tensor->lock);
    gpu->resident_weight_bytes += tensor->bytes;
    gpu->peak_resident_weight_bytes = std::max(
        gpu->peak_resident_weight_bytes, gpu->resident_weight_bytes);
    lru_append_locked(gpu, tensor);
    int ok = trim_offload_cache_locked(gpu);
    pthread_mutex_unlock(&gpu->offload_lock);
    return ok;
}

extern "C" int h3_gpu_quantize_weight_int8(h3_gpu *gpu,
                                             h3_gpu_tensor *output,
                                             h3_gpu_tensor *scales,
                                             const h3_gpu_tensor *input,
                                             uint32_t rows,
                                             uint32_t columns) {
    if (!h3cspeed_quantize_rows(gpu, output, scales, input,
                               rows, columns, 0, 0, 0)) return 0;
    return make_generated_weight_offloadable(gpu, output, "INT8 weight") &&
           make_generated_weight_offloadable(gpu, scales,
                                             "INT8 weight scales");
}

__global__ static void dequantize_gemm_kernel(
        void *output, h3_gpu_dtype output_dtype, const int32_t *accumulator,
        const float *input_scales, const float *weight_scales,
        uint32_t rows, uint32_t columns) {
    size_t total = (size_t)rows * columns;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += (size_t)gridDim.x * blockDim.x) {
        uint32_t row = (uint32_t)(index / columns);
        uint32_t column = (uint32_t)(index % columns);
        float value = (float)accumulator[index] * input_scales[row] *
                      weight_scales[column];
        h3cspeed_device_store(output, output_dtype, index, value);
    }
}

static int linear_int8_existing(h3_gpu *gpu, h3_gpu_tensor *output,
                                const h3_gpu_tensor *quantized_input,
                                const h3_gpu_tensor *input_scales,
                                const h3_gpu_tensor *weight,
                                const h3_gpu_tensor *weight_scales,
                                uint32_t rows, uint32_t input_dim,
                                uint32_t output_dim) {
    if (!gpu || !output || !quantized_input || !input_scales || !weight ||
        !weight_scales || quantized_input->dtype != H3_GPU_I8 ||
        weight->dtype != H3_GPU_I8 || input_scales->dtype != H3_GPU_F32 ||
        weight_scales->dtype != H3_GPU_F32 ||
        quantized_input->elements < (size_t)rows * input_dim ||
        input_scales->elements < rows ||
        weight->elements < (size_t)output_dim * input_dim ||
        weight_scales->elements < output_dim ||
        output->elements < (size_t)rows * output_dim ||
        !h3cspeed_tensor_wait(gpu, quantized_input) ||
        !h3cspeed_tensor_wait(gpu, input_scales) ||
        !h3cspeed_tensor_wait(gpu, weight) ||
        !h3cspeed_tensor_wait(gpu, weight_scales)) return 0;

    size_t accumulator_bytes = (size_t)rows * output_dim * sizeof(int32_t);
    int32_t *accumulator = static_cast<int32_t *>(
        h3cspeed_scratch_reserve(gpu, accumulator_bytes));
    if (!accumulator) return 0;
    int32_t alpha = 1;
    int32_t beta = 0;
    cublasStatus_t status = cublasGemmEx(
        gpu->blas, CUBLAS_OP_T, CUBLAS_OP_N,
        (int)output_dim, (int)rows, (int)input_dim,
        &alpha,
        weight->data, CUDA_R_8I, (int)input_dim,
        quantized_input->data, CUDA_R_8I, (int)input_dim,
        &beta,
        accumulator, CUDA_R_32I, (int)output_dim,
        CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (!h3cspeed_cublas_ok(gpu, status, "cuBLAS int8 GEMM")) return 0;
    dequantize_gemm_kernel<<<h3cspeed_blocks((size_t)rows * output_dim), 256, 0,
                              gpu->compute_stream>>>(
        output->data, output->dtype, accumulator,
        static_cast<const float *>(input_scales->data),
        static_cast<const float *>(weight_scales->data), rows, output_dim);
    h3cspeed_count_linear(gpu);
    return h3cspeed_launch_ok(gpu, "int8 GEMM dequantization");
}

extern "C" int h3_gpu_linear_int8_bf16(
        h3_gpu *gpu, h3_gpu_tensor *output,
        h3_gpu_tensor *quantized_input, h3_gpu_tensor *input_scales,
        const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
        const h3_gpu_tensor *weight_scales, uint32_t rows,
        uint32_t input_dim, uint32_t output_dim,
        int use_slower_uncached_int8_scales) {
    (void)use_slower_uncached_int8_scales;
    int quantized = weight && weight->convrot ?
        h3cspeed_quantize_rows_convrot(
            gpu, quantized_input, input_scales, input, rows, input_dim,
            0, 0, 0, weight->convrot_group_size) :
        h3cspeed_quantize_rows(gpu, quantized_input, input_scales, input,
                               rows, input_dim, 0, 0, 0);
    return quantized &&
           linear_int8_existing(gpu, output, quantized_input, input_scales,
                                 weight, weight_scales, rows, input_dim,
                                 output_dim);
}

extern "C" int h3_gpu_linear_int8_head_major_bf16(
        h3_gpu *gpu, h3_gpu_tensor *output,
        h3_gpu_tensor *quantized_input, h3_gpu_tensor *input_scales,
        const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
        const h3_gpu_tensor *weight_scales, uint32_t rows,
        uint32_t heads, uint32_t head_dim, uint32_t output_dim) {
    uint32_t width = heads * head_dim;
    int quantized = weight && weight->convrot ?
        h3cspeed_quantize_rows_convrot(
            gpu, quantized_input, input_scales, input, rows, width,
            1, heads, head_dim, weight->convrot_group_size) :
        h3cspeed_quantize_rows(gpu, quantized_input, input_scales, input,
                               rows, width, 1, heads, head_dim);
    return quantized &&
           linear_int8_existing(gpu, output, quantized_input, input_scales,
                                 weight, weight_scales, rows, width, output_dim);
}

extern "C" int h3_gpu_grouped_qkv_linear_rope_int8(
        h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
        h3_gpu_tensor *value, h3_gpu_tensor *quantized_input,
        h3_gpu_tensor *input_scales, const h3_gpu_tensor *input,
        const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales,
        const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm,
        const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
        uint32_t rows, uint32_t input_dim, uint32_t heads,
        uint32_t head_dim, uint32_t rope_half, float epsilon,
        int input_is_quantized, int use_slower_unfused_qkv_rope,
        int use_slower_scalar_qkv_rms,
        int use_slower_uncached_int8_scales) {
    (void)input_is_quantized;
    (void)use_slower_unfused_qkv_rope;
    (void)use_slower_scalar_qkv_rms;
    (void)use_slower_uncached_int8_scales;
    uint32_t qkv_dim = heads * head_dim * 3;
    h3_gpu_tensor *qkv = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * qkv_dim);
    if (!qkv) return 0;
    int ok = h3_gpu_linear_int8_bf16(
                 gpu, qkv, quantized_input, input_scales, input, weight,
                 weight_scales, rows, input_dim, qkv_dim, 0);
    if (ok) {
        /* Comfy checkpoints keep the projection rows as contiguous
         * [Q-all, K-all, V-all].  The pinned upstream checkpoint uses the
         * head-interleaved [head,Q,K,V] layout.  ConvRot is the fail-closed
         * checkpoint marker that selects the Comfy layout. */
        ok = weight->convrot ?
            h3_gpu_qkv_rope_bf16(
                gpu, query, key, value, qkv, q_norm, k_norm,
                rope_cos, rope_sin, rows, heads, head_dim, rope_half,
                epsilon) :
            h3_gpu_grouped_qkv_rope_bf16(
                gpu, query, key, value, qkv, q_norm, k_norm,
                rope_cos, rope_sin, rows, heads, head_dim, rope_half,
                epsilon);
    }
    h3_gpu_tensor_free(qkv);
    return ok;
}

extern "C" int h3_gpu_mlp_int8_bf16(
        h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *activated,
        h3_gpu_tensor *quantized_activation,
        h3_gpu_tensor *activation_scales, const h3_gpu_tensor *input,
        const h3_gpu_tensor *fc1_weight, const h3_gpu_tensor *fc1_scales,
        const h3_gpu_tensor *fc2_weight, const h3_gpu_tensor *fc2_scales,
        const h3_gpu_tensor *fc1_bf16, const h3_gpu_tensor *fc2_bf16,
        uint32_t rows, uint32_t input_dim, uint32_t hidden_dim,
        uint32_t output_dim, int use_slower_grouped_quantizer,
        int use_slower_dynamic_fc1_k, int use_int8_row_fc2,
        int input_is_quantized) {
    (void)use_slower_grouped_quantizer;
    (void)use_slower_dynamic_fc1_k;
    (void)use_int8_row_fc2;
    (void)input_is_quantized;
    if (!h3_gpu_has_int8_mlp(gpu) || !fc1_weight || !fc2_weight ||
        fc1_weight->dtype != H3_GPU_I8 || fc2_weight->dtype != H3_GPU_I8) {
        if (!fc1_bf16 || !fc2_bf16) return 0;
        return h3_gpu_mlp_bf16(gpu, output, input, fc1_bf16, fc2_bf16,
                               rows, input_dim, hidden_dim, output_dim);
    }
    h3_gpu_tensor *quantized_input = h3_gpu_tensor_new_i8(
        gpu, (size_t)rows * input_dim);
    h3_gpu_tensor *input_scales = h3_gpu_tensor_new_f32(gpu, rows);
    h3_gpu_tensor *fused = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * hidden_dim * 2);
    if (!quantized_input || !input_scales || !fused || !activated ||
        !quantized_activation || !activation_scales) {
        h3_gpu_tensor_free(quantized_input);
        h3_gpu_tensor_free(input_scales);
        h3_gpu_tensor_free(fused);
        return 0;
    }
    int ok = h3_gpu_linear_int8_bf16(
                 gpu, fused, quantized_input, input_scales, input,
                 fc1_weight, fc1_scales, rows, input_dim,
                 hidden_dim * 2, 0) &&
             h3_gpu_swiglu_bf16(gpu, activated, fused, rows, hidden_dim) &&
             h3_gpu_linear_int8_bf16(
                 gpu, output, quantized_activation, activation_scales,
                 activated, fc2_weight, fc2_scales, rows, hidden_dim,
                 output_dim, 0);
    h3_gpu_tensor_free(quantized_input);
    h3_gpu_tensor_free(input_scales);
    h3_gpu_tensor_free(fused);
    return ok;
}

extern "C" int h3_gpu_gate_adaln_quantize_int8(
        h3_gpu *gpu, h3_gpu_tensor *gated_residual,
        h3_gpu_tensor *quantized_output, h3_gpu_tensor *quantized_scales,
        const h3_gpu_tensor *residual, const h3_gpu_tensor *branch,
        const h3_gpu_tensor *norm_weight,
        const h3_gpu_tensor *gate_modulation,
        const h3_gpu_tensor *norm_modulation,
        const h3_gpu_tensor *row_map, uint32_t rows,
        uint32_t padded_rows, uint32_t width, uint32_t slots,
        uint32_t gate_slot, uint32_t shift_slot, uint32_t scale_slot,
        float epsilon) {
    if (!gpu || !gated_residual || !quantized_output || !quantized_scales ||
        !residual || !branch || !norm_weight || !gate_modulation ||
        !norm_modulation || !row_map || !rows || !width ||
        padded_rows < rows || quantized_output->dtype != H3_GPU_I8 ||
        quantized_scales->dtype != H3_GPU_F32 ||
        quantized_output->elements < (size_t)padded_rows * width ||
        quantized_scales->elements < padded_rows) return 0;
    h3_gpu_tensor *normalized = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * width);
    if (!normalized) return 0;
    int ok = h3_gpu_gate_adaln_bf16(
                 gpu, gated_residual, normalized, residual, branch,
                 norm_weight, gate_modulation, norm_modulation, row_map,
                 rows, width, slots, gate_slot, shift_slot, scale_slot,
                 epsilon) &&
             h3cspeed_quantize_rows(gpu, quantized_output, quantized_scales,
                                    normalized, rows, width, 0, 0, 0);
    if (ok && padded_rows > rows) {
        size_t offset = (size_t)rows * width;
        size_t bytes = (size_t)(padded_rows - rows) * width;
        ok = h3cspeed_cuda_ok(gpu,
            cudaMemsetAsync(static_cast<int8_t *>(quantized_output->data) + offset,
                            0, bytes, gpu->compute_stream),
            "zero padded int8 rows") &&
             h3cspeed_cuda_ok(gpu,
            cudaMemsetAsync(static_cast<float *>(quantized_scales->data) + rows,
                            0, (size_t)(padded_rows - rows) * sizeof(float),
                            gpu->compute_stream),
            "zero padded int8 row scales");
    }
    h3_gpu_tensor_free(normalized);
    return ok;
}

__global__ static void head_rms_kernel(
        void *tensor, h3_gpu_dtype tensor_dtype,
        const void *weight, h3_gpu_dtype weight_dtype,
        uint32_t rows, uint32_t head_dim, float epsilon) {
    uint32_t vector = blockIdx.x;
    if (vector >= rows) return;
    extern __shared__ float shared[];
    float square = 0.0f;
    for (uint32_t dimension = threadIdx.x; dimension < head_dim;
         dimension += blockDim.x) {
        float value = h3cspeed_device_load(tensor, tensor_dtype,
                                            (size_t)vector * head_dim + dimension);
        square += value * value;
    }
    shared[threadIdx.x] = square;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(shared[0] / (float)head_dim + epsilon);
    for (uint32_t dimension = threadIdx.x; dimension < head_dim;
         dimension += blockDim.x) {
        size_t index = (size_t)vector * head_dim + dimension;
        float value = h3cspeed_device_load(tensor, tensor_dtype, index) * inverse *
                      h3cspeed_device_load(weight, weight_dtype, dimension);
        h3cspeed_device_store(tensor, tensor_dtype, index, value);
    }
}

extern "C" int h3_gpu_head_rms_norm_bf16(
        h3_gpu *gpu, h3_gpu_tensor *tensor, const h3_gpu_tensor *weight,
        uint32_t sequence, uint32_t heads, uint32_t head_dim,
        float epsilon) {
    uint32_t vectors = sequence * heads;
    if (!gpu || !tensor || !weight ||
        tensor->elements < (size_t)vectors * head_dim ||
        weight->elements < head_dim || !h3cspeed_tensor_wait(gpu, tensor) ||
        !h3cspeed_tensor_wait(gpu, weight)) return 0;
    head_rms_kernel<<<vectors, 128, 128 * sizeof(float), gpu->compute_stream>>>(
        tensor->data, tensor->dtype, weight->data, weight->dtype,
        vectors, head_dim, epsilon);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "head RMS norm CUDA kernel");
}

__global__ static void text_qk_rope_kernel(
        void *output, h3_gpu_dtype output_dtype,
        const void *input, h3_gpu_dtype input_dtype,
        const void *norm, h3_gpu_dtype norm_dtype,
        const void *rope_cos, h3_gpu_dtype cos_dtype,
        const void *rope_sin, h3_gpu_dtype sin_dtype,
        uint32_t sequence, uint32_t heads, uint32_t head_dim,
        float epsilon) {
    uint32_t row = blockIdx.x;
    uint32_t head = blockIdx.y;
    if (row >= sequence || head >= heads) return;
    extern __shared__ float shared[];
    float square = 0.0f;
    size_t source_base = (size_t)row * heads * head_dim +
                         (size_t)head * head_dim;
    for (uint32_t dimension = threadIdx.x; dimension < head_dim;
         dimension += blockDim.x) {
        float value = h3cspeed_device_load(input, input_dtype,
                                            source_base + dimension);
        square += value * value;
    }
    shared[threadIdx.x] = square;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(shared[0] / (float)head_dim + epsilon);
    uint32_t half = head_dim / 2;
    for (uint32_t dimension = threadIdx.x; dimension < half;
         dimension += blockDim.x) {
        float first = h3cspeed_device_load(input, input_dtype,
                                            source_base + dimension) * inverse *
                      h3cspeed_device_load(norm, norm_dtype, dimension);
        float second = h3cspeed_device_load(input, input_dtype,
                                             source_base + half + dimension) * inverse *
                       h3cspeed_device_load(norm, norm_dtype, half + dimension);
        float cosine = h3cspeed_device_load(rope_cos, cos_dtype,
                                             (size_t)row * half + dimension);
        float sine = h3cspeed_device_load(rope_sin, sin_dtype,
                                           (size_t)row * half + dimension);
        size_t destination = ((size_t)head * sequence + row) * head_dim;
        h3cspeed_device_store(output, output_dtype, destination + dimension,
                              first * cosine - second * sine);
        h3cspeed_device_store(output, output_dtype, destination + half + dimension,
                              second * cosine + first * sine);
    }
}

extern "C" int h3_gpu_text_qk_rope_bf16(
        h3_gpu *gpu, h3_gpu_tensor *query_output,
        h3_gpu_tensor *key_output, const h3_gpu_tensor *query_input,
        const h3_gpu_tensor *key_input, const h3_gpu_tensor *q_norm,
        const h3_gpu_tensor *k_norm, const h3_gpu_tensor *rope_cos,
        const h3_gpu_tensor *rope_sin, uint32_t sequence,
        uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim,
        float epsilon) {
    size_t rope_elements = (size_t)sequence * (head_dim / 2);
    if (!gpu || !query_output || !key_output || !query_input || !key_input ||
        !q_norm || !k_norm || !rope_cos || !rope_sin ||
        !sequence || !query_heads || !kv_heads || !head_dim || (head_dim & 1u) ||
        query_output->elements < (size_t)sequence * query_heads * head_dim ||
        key_output->elements < (size_t)sequence * kv_heads * head_dim ||
        query_input->elements < (size_t)sequence * query_heads * head_dim ||
        key_input->elements < (size_t)sequence * kv_heads * head_dim ||
        q_norm->elements < head_dim || k_norm->elements < head_dim ||
        rope_cos->elements < rope_elements || rope_sin->elements < rope_elements ||
        !h3cspeed_tensor_wait(gpu, query_input) ||
        !h3cspeed_tensor_wait(gpu, key_input) ||
        !h3cspeed_tensor_wait(gpu, q_norm) || !h3cspeed_tensor_wait(gpu, k_norm) ||
        !h3cspeed_tensor_wait(gpu, rope_cos) || !h3cspeed_tensor_wait(gpu, rope_sin))
        return 0;
    dim3 query_grid(sequence, query_heads);
    dim3 key_grid(sequence, kv_heads);
    text_qk_rope_kernel<<<query_grid, 128, 128 * sizeof(float),
                          gpu->compute_stream>>>(
        query_output->data, query_output->dtype,
        query_input->data, query_input->dtype,
        q_norm->data, q_norm->dtype, rope_cos->data, rope_cos->dtype,
        rope_sin->data, rope_sin->dtype, sequence, query_heads,
        head_dim, epsilon);
    text_qk_rope_kernel<<<key_grid, 128, 128 * sizeof(float),
                          gpu->compute_stream>>>(
        key_output->data, key_output->dtype,
        key_input->data, key_input->dtype,
        k_norm->data, k_norm->dtype, rope_cos->data, rope_cos->dtype,
        rope_sin->data, rope_sin->dtype, sequence, kv_heads,
        head_dim, epsilon);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "text QK/RoPE CUDA kernel");
}

__global__ static void rope_inplace_kernel(
        void *tensor, h3_gpu_dtype tensor_dtype,
        const float *rope_cos, const float *rope_sin,
        uint32_t sequence, uint32_t heads, uint32_t head_dim) {
    uint32_t row = blockIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t half = head_dim / 2;
    for (uint32_t dimension = threadIdx.x; dimension < half;
         dimension += blockDim.x) {
        /* Text encoder tensors stay row-major, matching the Metal API. */
        size_t base = ((size_t)row * heads + head) * head_dim;
        float first = h3cspeed_device_load(tensor, tensor_dtype, base + dimension);
        float second = h3cspeed_device_load(tensor, tensor_dtype, base + half + dimension);
        float cosine = rope_cos[(size_t)row * half + dimension];
        float sine = rope_sin[(size_t)row * half + dimension];
        h3cspeed_device_store(tensor, tensor_dtype, base + dimension,
                              first * cosine - second * sine);
        h3cspeed_device_store(tensor, tensor_dtype, base + half + dimension,
                              second * cosine + first * sine);
    }
}

extern "C" int h3_gpu_rope_text_bf16(
        h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
        const h3_gpu_tensor *rope_cos_f32,
        const h3_gpu_tensor *rope_sin_f32, uint32_t sequence,
        uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim) {
    size_t query_elements = (size_t)sequence * query_heads * head_dim;
    size_t key_elements = (size_t)sequence * kv_heads * head_dim;
    size_t rope_elements = (size_t)sequence * (head_dim / 2);
    if (!gpu || !query || !key || !rope_cos_f32 || !rope_sin_f32 ||
        !sequence || !query_heads || !kv_heads || !head_dim || (head_dim & 1u) ||
        rope_cos_f32->dtype != H3_GPU_F32 || rope_sin_f32->dtype != H3_GPU_F32 ||
        query->elements < query_elements || key->elements < key_elements ||
        rope_cos_f32->elements < rope_elements ||
        rope_sin_f32->elements < rope_elements ||
        !h3cspeed_tensor_wait(gpu, query) || !h3cspeed_tensor_wait(gpu, key) ||
        !h3cspeed_tensor_wait(gpu, rope_cos_f32) ||
        !h3cspeed_tensor_wait(gpu, rope_sin_f32)) return 0;
    dim3 query_grid(sequence, query_heads);
    dim3 key_grid(sequence, kv_heads);
    rope_inplace_kernel<<<query_grid, 128, 0, gpu->compute_stream>>>(
        query->data, query->dtype,
        static_cast<const float *>(rope_cos_f32->data),
        static_cast<const float *>(rope_sin_f32->data),
        sequence, query_heads, head_dim);
    rope_inplace_kernel<<<key_grid, 128, 0, gpu->compute_stream>>>(
        key->data, key->dtype,
        static_cast<const float *>(rope_cos_f32->data),
        static_cast<const float *>(rope_sin_f32->data),
        sequence, kv_heads, head_dim);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "text in-place RoPE CUDA kernel");
}

__global__ static void euler_kernel(
        float *sample, size_t sample_offset,
        const void *last, h3_gpu_dtype last_dtype,
        const void *previous, h3_gpu_dtype previous_dtype,
        size_t elements, float delta, float ratio) {
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += (size_t)gridDim.x * blockDim.x) {
        float current = h3cspeed_device_load(last, last_dtype, index);
        float old = previous ? h3cspeed_device_load(previous, previous_dtype, index) : current;
        sample[sample_offset + index] += delta * (current + ratio * (current - old));
    }
}

extern "C" int h3_gpu_euler_bf16(
        h3_gpu *gpu, h3_gpu_tensor *sample, size_t sample_offset,
        const h3_gpu_tensor *last, const h3_gpu_tensor *previous,
        uint32_t elements, float delta, float ratio) {
    if (!gpu || !sample || !last || sample->dtype != H3_GPU_F32 ||
        sample_offset > sample->elements || elements > sample->elements - sample_offset ||
        last->elements < elements || !h3cspeed_tensor_wait(gpu, sample) ||
        !h3cspeed_tensor_wait(gpu, last) ||
        (previous && (!h3cspeed_tensor_wait(gpu, previous) ||
                      previous->elements < elements))) return 0;
    euler_kernel<<<h3cspeed_blocks(elements), 256, 0, gpu->compute_stream>>>(
        static_cast<float *>(sample->data), sample_offset,
        last->data, last->dtype, previous ? previous->data : nullptr,
        previous ? previous->dtype : last->dtype,
        elements, delta, ratio);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "Euler CUDA kernel");
}

__global__ static void token_pool_kernel(
        void *output, h3_gpu_dtype output_dtype,
        const void *input, h3_gpu_dtype input_dtype, size_t input_offset,
        void *original, h3_gpu_dtype original_dtype, size_t original_offset,
        void *baseline, h3_gpu_dtype baseline_dtype, size_t baseline_offset,
        const uint32_t *baseline_indices, const uint32_t *pairs,
        uint32_t input_rows, uint32_t rows, uint32_t baseline_rows,
        uint32_t width) {
    size_t total_original = (size_t)input_rows * width;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total_original; index += (size_t)gridDim.x * blockDim.x) {
        float value = h3cspeed_device_load(input, input_dtype, input_offset + index);
        h3cspeed_device_store(original, original_dtype, original_offset + index, value);
    }
    size_t total = (size_t)rows * width;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += (size_t)gridDim.x * blockDim.x) {
        uint32_t row = (uint32_t)(index / width);
        uint32_t column = (uint32_t)(index % width);
        uint32_t left = pairs[(size_t)row * 2];
        uint32_t right = pairs[(size_t)row * 2 + 1];
        if (left >= input_rows) left = 0;
        float value = h3cspeed_device_load(input, input_dtype,
            input_offset + (size_t)left * width + column);
        if (right < input_rows) {
            value = 0.5f * (value + h3cspeed_device_load(input, input_dtype,
                input_offset + (size_t)right * width + column));
        }
        h3cspeed_device_store(output, output_dtype, index, value);
        uint32_t baseline_index = baseline_indices ? baseline_indices[row] : row;
        if (baseline && baseline_index < baseline_rows) {
            h3cspeed_device_store(baseline, baseline_dtype,
                baseline_offset + (size_t)baseline_index * width + column, value);
        }
    }
}

extern "C" int h3_gpu_token_pool_bf16(
        h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
        size_t input_offset, h3_gpu_tensor *original, size_t original_offset,
        h3_gpu_tensor *baseline, size_t baseline_offset,
        const h3_gpu_tensor *baseline_indices, const h3_gpu_tensor *pairs,
        uint32_t input_rows, uint32_t rows, uint32_t baseline_rows,
        uint32_t width) {
    if (!gpu || !output || !input || !original || !pairs ||
        pairs->dtype != H3_GPU_U32 || pairs->elements < (size_t)rows * 2 ||
        output->elements < (size_t)rows * width ||
        input_offset > input->elements ||
        (size_t)input_rows * width > input->elements - input_offset ||
        original_offset > original->elements ||
        (size_t)input_rows * width > original->elements - original_offset ||
        (baseline && (baseline_offset > baseline->elements ||
          (size_t)baseline_rows * width > baseline->elements - baseline_offset)) ||
        (baseline_indices && (baseline_indices->dtype != H3_GPU_U32 ||
                              baseline_indices->elements < rows)) ||
        !h3cspeed_tensor_wait(gpu, input) || !h3cspeed_tensor_wait(gpu, pairs) ||
        (baseline_indices && !h3cspeed_tensor_wait(gpu, baseline_indices))) return 0;
    size_t work = std::max((size_t)input_rows * width, (size_t)rows * width);
    token_pool_kernel<<<h3cspeed_blocks(work), 256, 0, gpu->compute_stream>>>(
        output->data, output->dtype, input->data, input->dtype, input_offset,
        original->data, original->dtype, original_offset,
        baseline ? baseline->data : nullptr,
        baseline ? baseline->dtype : H3_GPU_BF16, baseline_offset,
        baseline_indices ? static_cast<const uint32_t *>(baseline_indices->data) : nullptr,
        static_cast<const uint32_t *>(pairs->data), input_rows, rows,
        baseline_rows, width);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "token pool CUDA kernel");
}

extern "C" int h3_gpu_token_pool_adaln_bf16(
        h3_gpu *gpu, h3_gpu_tensor *residual, h3_gpu_tensor *output,
        const h3_gpu_tensor *input, size_t input_offset,
        h3_gpu_tensor *original, size_t original_offset,
        h3_gpu_tensor *baseline, size_t baseline_offset,
        const h3_gpu_tensor *baseline_indices, const h3_gpu_tensor *pairs,
        const h3_gpu_tensor *norm_weight,
        const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map,
        uint32_t input_rows, uint32_t rows, uint32_t baseline_rows,
        uint32_t width, uint32_t slots, uint32_t shift_slot,
        uint32_t scale_slot, float epsilon) {
    return h3_gpu_token_pool_bf16(
               gpu, residual, input, input_offset, original, original_offset,
               baseline, baseline_offset, baseline_indices, pairs,
               input_rows, rows, baseline_rows, width) &&
           h3_gpu_adaln_bf16(gpu, output, residual, norm_weight,
                             modulation, row_map, rows, width, slots,
                             shift_slot, scale_slot, epsilon);
}

__global__ static void token_expand_kernel(
        void *output, h3_gpu_dtype output_dtype,
        const void *original, h3_gpu_dtype original_dtype,
        size_t original_offset, const void *reduced,
        h3_gpu_dtype reduced_dtype, const void *baseline,
        h3_gpu_dtype baseline_dtype, size_t baseline_offset,
        const uint32_t *baseline_indices, const uint32_t *parents,
        uint32_t rows, uint32_t reduced_rows, uint32_t baseline_rows,
        uint32_t width, uint32_t exact_prefix_rows, float update_scale) {
    size_t total = (size_t)rows * width;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += (size_t)gridDim.x * blockDim.x) {
        uint32_t row = (uint32_t)(index / width);
        uint32_t column = (uint32_t)(index % width);
        float value = h3cspeed_device_load(original, original_dtype,
                                            original_offset + index);
        if (row >= exact_prefix_rows) {
            uint32_t parent = parents[row];
            if (parent < reduced_rows) {
                float update = h3cspeed_device_load(reduced, reduced_dtype,
                    (size_t)parent * width + column);
                uint32_t baseline_index = baseline_indices ?
                    baseline_indices[parent] : parent;
                if (baseline && baseline_index < baseline_rows) {
                    update -= h3cspeed_device_load(baseline, baseline_dtype,
                        baseline_offset + (size_t)baseline_index * width + column);
                }
                value += update_scale * update;
            }
        }
        h3cspeed_device_store(output, output_dtype, index, value);
    }
}

extern "C" int h3_gpu_token_expand_delta_bf16(
        h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *original,
        size_t original_offset, const h3_gpu_tensor *reduced,
        const h3_gpu_tensor *baseline, size_t baseline_offset,
        const h3_gpu_tensor *baseline_indices,
        const h3_gpu_tensor *parents, uint32_t rows,
        uint32_t reduced_rows, uint32_t baseline_rows, uint32_t width,
        uint32_t exact_prefix_rows, float update_scale) {
    if (!gpu || !output || !original || !reduced || !parents ||
        parents->dtype != H3_GPU_U32 || parents->elements < rows ||
        output->elements < (size_t)rows * width ||
        original_offset > original->elements ||
        (size_t)rows * width > original->elements - original_offset ||
        reduced->elements < (size_t)reduced_rows * width ||
        (baseline && (baseline_offset > baseline->elements ||
          (size_t)baseline_rows * width > baseline->elements - baseline_offset)) ||
        (baseline_indices && (baseline_indices->dtype != H3_GPU_U32 ||
                              baseline_indices->elements < reduced_rows)) ||
        !h3cspeed_tensor_wait(gpu, original) ||
        !h3cspeed_tensor_wait(gpu, reduced) ||
        !h3cspeed_tensor_wait(gpu, parents) ||
        (baseline && !h3cspeed_tensor_wait(gpu, baseline)) ||
        (baseline_indices && !h3cspeed_tensor_wait(gpu, baseline_indices))) return 0;
    token_expand_kernel<<<h3cspeed_blocks((size_t)rows * width), 256, 0,
                           gpu->compute_stream>>>(
        output->data, output->dtype, original->data, original->dtype,
        original_offset, reduced->data, reduced->dtype,
        baseline ? baseline->data : nullptr,
        baseline ? baseline->dtype : H3_GPU_BF16, baseline_offset,
        baseline_indices ? static_cast<const uint32_t *>(baseline_indices->data) : nullptr,
        static_cast<const uint32_t *>(parents->data), rows, reduced_rows,
        baseline_rows, width, exact_prefix_rows, update_scale);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "token expand CUDA kernel");
}

extern "C" int h3_gpu_token_expand_adaln_bf16(
        h3_gpu *gpu, h3_gpu_tensor *residual, h3_gpu_tensor *output,
        const h3_gpu_tensor *original, size_t original_offset,
        const h3_gpu_tensor *reduced, const h3_gpu_tensor *baseline,
        size_t baseline_offset, const h3_gpu_tensor *baseline_indices,
        const h3_gpu_tensor *parents, const h3_gpu_tensor *norm_weight,
        const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map,
        uint32_t rows, uint32_t reduced_rows, uint32_t baseline_rows,
        uint32_t width, uint32_t exact_prefix_rows, float update_scale,
        uint32_t slots, uint32_t shift_slot, uint32_t scale_slot,
        float epsilon) {
    return h3_gpu_token_expand_delta_bf16(
               gpu, residual, original, original_offset, reduced, baseline,
               baseline_offset, baseline_indices, parents, rows,
               reduced_rows, baseline_rows, width, exact_prefix_rows,
               update_scale) &&
           h3_gpu_adaln_bf16(gpu, output, residual, norm_weight,
                             modulation, row_map, rows, width, slots,
                             shift_slot, scale_slot, epsilon);
}

/*
 * Quantized-weight readers
 * ------------------------
 *
 * These are deliberately kept as a vertical-slice implementation outside the
 * public h3_gpu.h ABI.  The source files are read in bounded host chunks and
 * every conversion is performed by a CUDA kernel; no path routes through a CPU
 * GEMM.  A converted host backing copy is then retained for safe low-VRAM
 * eviction and reload.
 */

static int quant_open_range(const char *path, uint64_t offset, size_t bytes,
                            int *descriptor, char *error, size_t error_size) {
    if (descriptor) *descriptor = -1;
    if (!path || !*path || !descriptor) {
        if (error && error_size) snprintf(error, error_size,
                                          "quantized source path is required");
        return 0;
    }
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        if (error && error_size) snprintf(error, error_size,
                                          "cannot open %s: %s", path,
                                          strerror(errno));
        return 0;
    }
    if (!validate_file_range(fd, path, offset, bytes, error, error_size)) {
        close(fd);
        return 0;
    }
    *descriptor = fd;
    return 1;
}

static int quant_prepare_output(h3_gpu_tensor *destination, size_t elements,
                                char *error, size_t error_size) {
    if (!destination || !destination->gpu ||
        (destination->dtype != H3_GPU_BF16 && destination->dtype != H3_GPU_F32) ||
        destination->elements < elements) {
        if (error && error_size) snprintf(error, error_size,
                                          "quantized destination must be BF16/F32 and large enough");
        return 0;
    }
    h3_gpu *gpu = destination->gpu;
    if (destination->data) {
        if (!h3cspeed_cuda_ok(gpu,
                              profiled_stream_synchronize(
                                  gpu, gpu->compute_stream,
                                  H3CSPEED_PROFILE_STREAM_COMPUTE),
                              "synchronize quantized destination")) return 0;
    } else if (destination->bytes) {
        pthread_mutex_lock(&gpu->offload_lock);
        const int restoring_weight = destination->offloadable ? 1 : 0;
        int ok = device_allocate_locked(gpu, &destination->data,
                                        destination->bytes, restoring_weight,
                                        destination);
        pthread_mutex_unlock(&gpu->offload_lock);
        if (!ok) {
            if (error && error_size) snprintf(error, error_size, "%s",
                                              h3_gpu_error(gpu));
            return 0;
        }
    }
    if (gpu->offload.enabled) {
        pthread_mutex_lock(&gpu->offload_lock);
        int ok = destination->host_data || host_backing_allocate_locked(
            gpu, destination);
        if (ok) {
            destination->host_valid = 0;
            pthread_mutex_lock(&destination->lock);
            destination->ready_valid = 0;
            destination->last_use_valid = 0;
            pthread_mutex_unlock(&destination->lock);
        }
        pthread_mutex_unlock(&gpu->offload_lock);
        if (!ok) {
            if (error && error_size) snprintf(error, error_size,
                                              "cannot allocate quantized host backing");
            return 0;
        }
    }
    return 1;
}

static int quant_commit_output(h3_gpu_tensor *destination,
                               char *error, size_t error_size) {
    if (!destination || !destination->gpu || !destination->data) return 0;
    h3_gpu *gpu = destination->gpu;
    if (!h3cspeed_cuda_ok(gpu,
                          profiled_stream_synchronize(
                              gpu, gpu->compute_stream,
                              H3CSPEED_PROFILE_STREAM_COMPUTE),
                          "synchronize quantized conversion")) return 0;
    if (gpu->offload.enabled && destination->host_data) {
        if (!h3cspeed_cuda_ok(gpu,
                              cudaMemcpy(destination->host_data,
                                         destination->data, destination->bytes,
                                         cudaMemcpyDeviceToHost),
                              "save quantized host backing")) return 0;
        pthread_mutex_lock(&gpu->offload_lock);
        destination->host_valid = 1;
        destination->source_bytes = destination->bytes;
        destination->source_offset = 0;
        destination->source_streaming = 0;
        if (!destination->offloadable) {
            destination->offloadable = 1;
            destination->pin_epoch = 0;
            gpu->resident_weight_bytes += destination->bytes;
            gpu->peak_resident_weight_bytes = std::max(
                gpu->peak_resident_weight_bytes,
                gpu->resident_weight_bytes);
        }
        host_lru_append_locked(gpu, destination);
        lru_append_locked(gpu, destination);
        int ok = trim_offload_cache_locked(gpu);
        pthread_mutex_unlock(&gpu->offload_lock);
        if (!ok) {
            if (error && error_size && !error[0])
                snprintf(error, error_size, "%s", h3_gpu_error(gpu));
            return 0;
        }
    }
    if (error && error_size) error[0] = '\0';
    return 1;
}

static h3_gpu_tensor *quant_temp_bytes(h3_gpu *gpu, size_t bytes) {
    return tensor_new_internal(gpu, H3_GPU_I8, bytes, 1);
}

static int quant_copy_temp(h3_gpu *gpu, h3_gpu_tensor *temporary,
                           const void *host, size_t bytes,
                           char *error, size_t error_size) {
    if (!temporary || temporary->bytes < bytes || !host) return 0;
    if (!h3cspeed_cuda_ok(gpu,
                          profiled_h2d_async(gpu, temporary->data, host, bytes,
                                             gpu->upload_stream),
                          "upload quantized source chunk") ||
        !h3cspeed_cuda_ok(gpu,
                          profiled_stream_synchronize(
                              gpu, gpu->upload_stream,
                              H3CSPEED_PROFILE_STREAM_UPLOAD),
                          "synchronize quantized source chunk")) {
        if (error && error_size && !error[0]) snprintf(error, error_size,
                                                       "%s", h3_gpu_error(gpu));
        return 0;
    }
    return 1;
}

__global__ static void quant_f16_to_output_kernel(
        const uint16_t *source, void *destination, h3_gpu_dtype output_dtype,
        size_t elements) {
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += (size_t)gridDim.x * blockDim.x) {
        float value = __half2float(__ushort_as_half(source[index]));
        h3cspeed_device_store(destination, output_dtype, index, value);
    }
}

__global__ static void quant_i8_to_bf16_kernel(
        const int8_t *source, const float *scales, __nv_bfloat16 *destination,
        uint32_t rows, uint32_t columns, uint32_t row_offset) {
    size_t elements = (size_t)rows * columns;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += (size_t)gridDim.x * blockDim.x) {
        uint32_t row = (uint32_t)(index / columns);
        destination[(size_t)(row_offset + row) * columns + index % columns] =
            __float2bfloat16_rn((float)source[index] * scales[row]);
    }
}

__global__ static void quant_mul_bf16_kernel(
        __nv_bfloat16 *output, const __nv_bfloat16 *input,
        const __nv_bfloat16 *column_scale, uint32_t rows,
        uint32_t columns) {
    size_t elements = (size_t)rows * columns;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += (size_t)gridDim.x * blockDim.x) {
        uint32_t column = (uint32_t)(index % columns);
        float value = __bfloat162float(input[index]) *
                      __bfloat162float(column_scale[column]);
        output[index] = __float2bfloat16_rn(value);
    }
}

extern "C" int h3cspeed_gpu_mul_bf16(
        h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
        const h3_gpu_tensor *column_scale, uint32_t rows, uint32_t columns) {
    size_t elements = (size_t)rows * columns;
    if (!gpu || !output || !input || !column_scale ||
        output->dtype != H3_GPU_BF16 || input->dtype != H3_GPU_BF16 ||
        column_scale->dtype != H3_GPU_BF16 || output->elements < elements ||
        input->elements < elements || column_scale->elements < columns ||
        !h3cspeed_tensor_wait(gpu, input) ||
        !h3cspeed_tensor_wait(gpu, column_scale)) return 0;
    quant_mul_bf16_kernel<<<h3cspeed_blocks(elements), 256, 0,
                            gpu->compute_stream>>>(
        static_cast<__nv_bfloat16 *>(output->data),
        static_cast<const __nv_bfloat16 *>(input->data),
        static_cast<const __nv_bfloat16 *>(column_scale->data), rows, columns);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "BF16 column scale CUDA kernel");
}

__device__ static float nvfp4_e2m1_value(uint32_t value) {
    constexpr float table[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
       -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f};
    return table[value & 15u];
}

__device__ static float nvfp4_e4m3_value(uint8_t bits) {
    uint32_t sign = bits >> 7;
    uint32_t exponent = (bits >> 3) & 15u;
    uint32_t mantissa = bits & 7u;
    float value;
    if (exponent == 0) {
        value = (float)mantissa * (1.0f / 8.0f) * (1.0f / 64.0f);
    } else if (exponent == 15) {
        /* E4M3FN reserves the all-ones mantissa as NaN.  Quantizers do not
         * emit it for scales; treating it as zero is fail-closed. */
        value = mantissa == 7u ? 0.0f :
            (1.0f + (float)mantissa * (1.0f / 8.0f)) * 256.0f;
    } else {
        value = (1.0f + (float)mantissa * (1.0f / 8.0f)) *
                exp2f((float)exponent - 7.0f);
    }
    return sign ? -value : value;
}

__device__ static size_t nvfp4_blocked_scale_index(
        uint32_t row, uint32_t block, uint32_t blocks_per_row,
        int blocked_layout) {
    if (!blocked_layout) return (size_t)row * blocks_per_row + block;
    uint32_t column_group = block / 4u;
    uint32_t column_in_group = block & 3u;
    uint32_t row_group = row / 128u;
    uint32_t row_in_group = row & 127u;
    uint32_t row32 = row_in_group & 31u;
    uint32_t column4 = row_in_group / 32u;
    uint32_t n_column_groups = (blocks_per_row + 3u) / 4u;
    size_t tile = (size_t)row_group * n_column_groups + column_group;
    return tile * 512u + (size_t)row32 * 16u +
           (size_t)column4 * 4u + column_in_group;
}

__global__ static void quant_nvfp4_to_bf16_kernel(
        const uint8_t *packed, const uint8_t *scales,
        const __nv_bfloat16 *pre_scale, __nv_bfloat16 *destination,
        uint32_t rows, uint32_t columns, uint32_t row_offset,
        uint32_t blocks_per_row, uint32_t scale_rows, float tensor_scale,
        int blocked_layout) {
    size_t packed_columns = columns / 2u;
    size_t elements = (size_t)rows * columns;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += (size_t)gridDim.x * blockDim.x) {
        uint32_t row = (uint32_t)(index / columns);
        uint32_t column = (uint32_t)(index % columns);
        uint8_t packed_value = packed[(size_t)row * packed_columns + column / 2u];
        uint32_t nibble = (column & 1u) ?
            (uint32_t)(packed_value & 0x0fu) :
            (uint32_t)(packed_value >> 4);
        uint32_t block = column / 16u;
        size_t scale_index = nvfp4_blocked_scale_index(
            row + row_offset, block, blocks_per_row, blocked_layout);
        /* Match Comfy-Kitchen's eager BF16 dequantization boundaries.  The
         * global and block scales are first converted to BF16, their product
         * is rounded to BF16, and the E2M1 value is applied with one final
         * BF16 rounding.  Collapsing these operations into one F32 expression
         * measurably changes real Qwen weights across fifty layers. */
        float global_bf16 = __bfloat162float(__float2bfloat16_rn(tensor_scale));
        float block_bf16 = __bfloat162float(
            __float2bfloat16_rn(nvfp4_e4m3_value(scales[scale_index])));
        float total_scale = __bfloat162float(
            __float2bfloat16_rn(global_bf16 * block_bf16));
        float value = nvfp4_e2m1_value(nibble) * total_scale;
        if (pre_scale) value *= __bfloat162float(pre_scale[column]);
        destination[(size_t)(row_offset + row) * columns + column] =
            __float2bfloat16_rn(value);
    }
    (void)scale_rows;
}

static int quant_read_f16(h3_gpu_tensor *destination, const char *path,
                          uint64_t file_offset, size_t elements,
                          h3_gpu_dtype output_dtype, char *error,
                          size_t error_size) {
    if (!destination || !destination->gpu ||
        (output_dtype != H3_GPU_BF16 && output_dtype != H3_GPU_F32) ||
        destination->elements < elements ||
        !quant_prepare_output(destination, elements, error, error_size)) return 0;
    h3_gpu *gpu = destination->gpu;
    size_t max_bytes = gpu->offload.staging_bytes ?
        (size_t)gpu->offload.staging_bytes : 64u * 1024u * 1024u;
    size_t max_elements = std::max((size_t)1, max_bytes / sizeof(uint16_t));
    int descriptor = -1;
    size_t source_bytes = 0;
    if (!h3cspeed_size_mul(elements, sizeof(uint16_t), &source_bytes) ||
        !quant_open_range(path, file_offset, source_bytes, &descriptor,
                          error, error_size)) return 0;
    h3_gpu_tensor *temporary = nullptr;
    uint16_t *host = nullptr;
    int ok = 1;
    for (size_t offset = 0; ok && offset < elements;) {
        size_t count = std::min(elements - offset, max_elements);
        size_t bytes = count * sizeof(uint16_t);
        host = static_cast<uint16_t *>(malloc(bytes));
        temporary = quant_temp_bytes(gpu, bytes);
        if (!host || !temporary ||
            !read_exact(gpu, descriptor, host, bytes, file_offset + offset * 2,
                        error, error_size) ||
            !quant_copy_temp(gpu, temporary, host, bytes, error, error_size)) {
            ok = 0;
            break;
        }
        quant_f16_to_output_kernel<<<h3cspeed_blocks(count), 256, 0,
                                     gpu->compute_stream>>>(
            static_cast<const uint16_t *>(temporary->data),
            static_cast<unsigned char *>(destination->data) +
                offset * h3cspeed_dtype_size(output_dtype), output_dtype, count);
        ok = h3cspeed_launch_ok(gpu, "F16 conversion kernel");
        h3_gpu_tensor_free(temporary); temporary = nullptr;
        free(host); host = nullptr;
        offset += count;
    }
    if (temporary) h3_gpu_tensor_free(temporary);
    free(host);
    close(descriptor);
    if (!ok || !quant_commit_output(destination, error, error_size)) return 0;
    return 1;
}

extern "C" h3_gpu_tensor *h3cspeed_gpu_tensor_load_f16_as_bf16(
        h3_gpu *gpu, const char *path, uint64_t file_offset, size_t elements) {
    h3_gpu_tensor *tensor = tensor_new_internal(gpu, H3_GPU_BF16, elements,
                                                 gpu && !gpu->offload.enabled);
    char error[512] = {0};
    if (tensor && !h3cspeed_gpu_tensor_read_f16_as_bf16(
            tensor, path, file_offset, elements, error, sizeof(error))) {
        h3cspeed_set_error(gpu, "load F16 as BF16", error);
        h3_gpu_tensor_free(tensor);
        tensor = nullptr;
    }
    return tensor;
}

extern "C" h3_gpu_tensor *h3cspeed_gpu_tensor_load_f16_as_f32(
        h3_gpu *gpu, const char *path, uint64_t file_offset, size_t elements) {
    h3_gpu_tensor *tensor = tensor_new_internal(gpu, H3_GPU_F32, elements,
                                                 gpu && !gpu->offload.enabled);
    char error[512] = {0};
    if (tensor && !quant_read_f16(tensor, path, file_offset, elements,
                                  H3_GPU_F32, error, sizeof(error))) {
        h3cspeed_set_error(gpu, "load F16 as F32", error);
        h3_gpu_tensor_free(tensor);
        tensor = nullptr;
    }
    return tensor;
}

extern "C" int h3cspeed_gpu_tensor_read_f16_as_bf16(
        h3_gpu_tensor *destination, const char *path, uint64_t file_offset,
        size_t elements, char *error, size_t error_size) {
    if (!destination || destination->dtype != H3_GPU_BF16) return 0;
    return quant_read_f16(destination, path, file_offset, elements,
                          H3_GPU_BF16, error, error_size);
}

extern "C" int h3cspeed_gpu_tensor_read_i8_as_bf16(
        h3_gpu_tensor *destination, const char *weight_path,
        uint64_t weight_offset, const char *scale_path, uint64_t scale_offset,
        uint32_t rows, uint32_t columns, char *error, size_t error_size) {
    size_t elements = 0, weight_bytes = 0, scale_bytes = 0;
    if (!destination || destination->dtype != H3_GPU_BF16 ||
        !h3cspeed_size_mul(rows, columns, &elements) ||
        !h3cspeed_size_mul(elements, sizeof(int8_t), &weight_bytes) ||
        !h3cspeed_size_mul(rows, sizeof(float), &scale_bytes) ||
        destination->elements < elements ||
        !quant_prepare_output(destination, elements, error, error_size)) return 0;
    h3_gpu *gpu = destination->gpu;
    int weight_fd = -1, scale_fd = -1;
    if (!quant_open_range(weight_path, weight_offset, weight_bytes, &weight_fd,
                          error, error_size) ||
        !quant_open_range(scale_path, scale_offset, scale_bytes, &scale_fd,
                          error, error_size)) {
        if (weight_fd >= 0) close(weight_fd);
        return 0;
    }
    size_t max_bytes = gpu->offload.staging_bytes ?
        (size_t)gpu->offload.staging_bytes : 64u * 1024u * 1024u;
    size_t rows_per_chunk = std::max((size_t)1, max_bytes / std::max((size_t)1,
                                                                        (size_t)columns));
    rows_per_chunk = std::min(rows_per_chunk, (size_t)rows);
    size_t chunk_weight_bytes = rows_per_chunk * columns;
    int8_t *host_weight = static_cast<int8_t *>(malloc(chunk_weight_bytes));
    float *host_scale = static_cast<float *>(malloc(rows_per_chunk * sizeof(float)));
    h3_gpu_tensor *weight_temp = quant_temp_bytes(gpu, chunk_weight_bytes);
    h3_gpu_tensor *scale_temp = tensor_new_internal(
        gpu, H3_GPU_F32, rows_per_chunk, 1);
    int ok = host_weight && host_scale && weight_temp && scale_temp;
    for (uint32_t row = 0; ok && row < rows; row += (uint32_t)rows_per_chunk) {
        uint32_t count = (uint32_t)std::min((size_t)(rows - row), rows_per_chunk);
        size_t bytes = (size_t)count * columns;
        ok = read_exact(gpu, weight_fd, host_weight, bytes,
                        weight_offset + (uint64_t)row * columns,
                        error, error_size) &&
             read_exact(gpu, scale_fd, host_scale, (size_t)count * sizeof(float),
                        scale_offset + (uint64_t)row * sizeof(float),
                        error, error_size) &&
             quant_copy_temp(gpu, weight_temp, host_weight, bytes,
                             error, error_size) &&
             quant_copy_temp(gpu, scale_temp, host_scale,
                             (size_t)count * sizeof(float), error, error_size);
        if (ok) {
            quant_i8_to_bf16_kernel<<<h3cspeed_blocks((size_t)count * columns),
                                      256, 0, gpu->compute_stream>>>(
                static_cast<const int8_t *>(weight_temp->data),
                static_cast<const float *>(scale_temp->data),
                static_cast<__nv_bfloat16 *>(destination->data), count,
                columns, row);
            ok = h3cspeed_launch_ok(gpu, "I8 dequantization kernel");
        }
    }
    if (weight_temp) h3_gpu_tensor_free(weight_temp);
    if (scale_temp) h3_gpu_tensor_free(scale_temp);
    free(host_weight); free(host_scale);
    close(weight_fd); close(scale_fd);
    return ok && quant_commit_output(destination, error, error_size);
}

extern "C" int h3cspeed_gpu_tensor_read_nvfp4_as_bf16(
        h3_gpu_tensor *destination, const char *packed_path,
        uint64_t packed_offset, const char *scale_path, uint64_t scale_offset,
        float tensor_scale, const char *pre_scale_path,
        uint64_t pre_scale_offset, uint32_t rows, uint32_t columns,
        char *error, size_t error_size) {
    size_t elements = 0, packed_bytes = 0;
    size_t blocks = ((size_t)columns + 15u) / 16u;
    size_t scale_bytes = 0, pre_bytes = 0;
    if (!destination || destination->dtype != H3_GPU_BF16 || !columns ||
        (columns & 1u) || !h3cspeed_size_mul(rows, columns, &elements) ||
        !h3cspeed_size_mul(rows, columns / 2u, &packed_bytes) ||
        !h3cspeed_size_mul(rows, blocks, &scale_bytes) ||
        !h3cspeed_size_mul(columns, sizeof(__nv_bfloat16), &pre_bytes) ||
        destination->elements < elements ||
        !quant_prepare_output(destination, elements, error, error_size)) return 0;
    h3_gpu *gpu = destination->gpu;
    int packed_fd = -1, scale_fd = -1, pre_fd = -1;
    if (!quant_open_range(packed_path, packed_offset, packed_bytes, &packed_fd,
                          error, error_size) ||
        !quant_open_range(scale_path, scale_offset, scale_bytes, &scale_fd,
                          error, error_size)) {
        if (packed_fd >= 0) close(packed_fd);
        return 0;
    }
    if (pre_scale_path && *pre_scale_path &&
        !quant_open_range(pre_scale_path, pre_scale_offset, pre_bytes, &pre_fd,
                          error, error_size)) {
        close(packed_fd); close(scale_fd); return 0;
    }
    uint8_t *host_scale = static_cast<uint8_t *>(malloc(scale_bytes));
    __nv_bfloat16 *host_pre = pre_fd >= 0 ?
        static_cast<__nv_bfloat16 *>(malloc(pre_bytes)) : nullptr;
    size_t max_bytes = gpu->offload.staging_bytes ?
        (size_t)gpu->offload.staging_bytes : 64u * 1024u * 1024u;
    size_t packed_row_bytes = columns / 2u;
    size_t rows_per_chunk = std::max((size_t)1, max_bytes /
                                               std::max((size_t)1, packed_row_bytes));
    rows_per_chunk = std::min(rows_per_chunk, (size_t)rows);
    size_t chunk_bytes = rows_per_chunk * packed_row_bytes;
    uint8_t *host_packed = static_cast<uint8_t *>(malloc(chunk_bytes));
    h3_gpu_tensor *packed_temp = quant_temp_bytes(gpu, chunk_bytes);
    h3_gpu_tensor *scale_temp = quant_temp_bytes(gpu, scale_bytes);
    h3_gpu_tensor *pre_temp = pre_fd >= 0 ?
        tensor_new_internal(gpu, H3_GPU_BF16, columns, 1) : nullptr;
    int ok = host_scale && host_packed && packed_temp && scale_temp &&
             (pre_fd < 0 || (host_pre && pre_temp));
    if (!ok && error && error_size && !error[0])
        snprintf(error, error_size,
                 "cannot allocate NVFP4 conversion staging buffers");
    if (ok) ok = read_exact(gpu, scale_fd, host_scale, scale_bytes, scale_offset,
                             error, error_size) &&
                quant_copy_temp(gpu, scale_temp, host_scale, scale_bytes,
                                error, error_size);
    if (ok && pre_fd >= 0)
        ok = read_exact(gpu, pre_fd, host_pre, pre_bytes, pre_scale_offset,
                        error, error_size) &&
             quant_copy_temp(gpu, pre_temp, host_pre, pre_bytes,
                             error, error_size);
    int blocked_layout = (rows % 128u == 0 && blocks % 4u == 0) ? 1 : 0;
    for (uint32_t row = 0; ok && row < rows; row += (uint32_t)rows_per_chunk) {
        uint32_t count = (uint32_t)std::min((size_t)(rows - row), rows_per_chunk);
        size_t bytes = (size_t)count * packed_row_bytes;
        ok = read_exact(gpu, packed_fd, host_packed, bytes,
                        packed_offset + (uint64_t)row * packed_row_bytes,
                        error, error_size) &&
             quant_copy_temp(gpu, packed_temp, host_packed, bytes,
                             error, error_size);
        if (ok) {
            quant_nvfp4_to_bf16_kernel<<<h3cspeed_blocks((size_t)count * columns),
                                         256, 0, gpu->compute_stream>>>(
                static_cast<const uint8_t *>(packed_temp->data),
                static_cast<const uint8_t *>(scale_temp->data),
                /* pre_quant_scale is applied to the activation by the Qwen
                 * text encoder immediately before this weight is consumed.
                 * Keep validating/loading the optional companion above, but
                 * never fold it into the dequantized weight or it is applied
                 * twice at runtime. */
                nullptr,
                static_cast<__nv_bfloat16 *>(destination->data), count, columns,
                row, (uint32_t)blocks, rows, tensor_scale, blocked_layout);
            ok = h3cspeed_launch_ok(gpu, "NVFP4 dequantization kernel");
        }
    }
    if (packed_temp) h3_gpu_tensor_free(packed_temp);
    if (scale_temp) h3_gpu_tensor_free(scale_temp);
    if (pre_temp) h3_gpu_tensor_free(pre_temp);
    free(host_packed); free(host_scale); free(host_pre);
    close(packed_fd); close(scale_fd); if (pre_fd >= 0) close(pre_fd);
    return ok && quant_commit_output(destination, error, error_size);
}
