#include "h3_cuda_common.cuh"

#include <algorithm>
#include <climits>
#include <cmath>
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

static int read_exact(int descriptor, void *buffer, size_t bytes,
                      uint64_t offset, char *error, size_t error_size);

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
    return h3cspeed_cuda_ok(gpu, cudaEventSynchronize(event), operation);
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
                                        int count_eviction) {
    if (!gpu || !tensor || !tensor->data) return 1;
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
    lru_remove_locked(gpu, tensor);
    if (tensor->offloadable) {
        gpu->resident_weight_bytes = gpu->resident_weight_bytes >= tensor->bytes ?
            gpu->resident_weight_bytes - tensor->bytes : 0;
        if (count_eviction) {
            gpu->offload_evictions++;
            gpu->offload_evicted_bytes += tensor->bytes;
        }
    }
    track_device_release(gpu, tensor->bytes);
    return 1;
}

static h3_gpu_tensor *eviction_candidate_locked(h3_gpu *gpu,
                                                 h3_gpu_tensor *protected_tensor) {
    for (h3_gpu_tensor *candidate = gpu->lru_head; candidate;
         candidate = candidate->lru_next) {
        if (candidate == protected_tensor || !candidate->data ||
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
        if (!release_tensor_device_locked(gpu, candidate, 1, 1)) return 0;
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
        if (!release_tensor_device_locked(gpu, candidate, 1, 1)) return 0;
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
    cudaError_t status = cudaMalloc(pointer, bytes);
    if (status != cudaSuccess && gpu->offload.enabled) {
        (void)cudaGetLastError();
        h3_gpu_tensor *candidate = nullptr;
        while ((candidate = eviction_candidate_locked(gpu, protected_tensor))) {
            if (!release_tensor_device_locked(gpu, candidate, 1, 1)) return 0;
        }
        status = cudaMalloc(pointer, bytes);
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
    return 1;
}

static int upload_weight_locked(h3_gpu *gpu, h3_gpu_tensor *tensor) {
    if (!gpu || !tensor || !tensor->data || !tensor->source_bytes) return 0;
    if (tensor->host_data && tensor->host_valid)
        host_lru_append_locked(gpu, tensor);
    if (tensor->host_data && tensor->host_valid && tensor->host_pinned) {
        if (!h3cspeed_cuda_ok(gpu,
            cudaMemcpyAsync(tensor->data, tensor->host_data,
                            tensor->source_bytes, cudaMemcpyHostToDevice,
                            gpu->upload_stream),
            "upload pinned host weight")) return 0;
    } else {
        pthread_mutex_lock(&gpu->staging_lock);
        if (!staging_allocate_locked(gpu)) {
            pthread_mutex_unlock(&gpu->staging_lock);
            return 0;
        }
        int descriptor = -1;
        if (!tensor->host_data || !tensor->host_valid) {
            descriptor = open(tensor->source_path, O_RDONLY | O_CLOEXEC);
            if (descriptor < 0) {
                char detail[384];
                snprintf(detail, sizeof(detail), "cannot open %s: %s",
                         tensor->source_path ? tensor->source_path : "(null)",
                         strerror(errno));
                h3cspeed_set_error(gpu, "file-backed weight upload", detail);
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
            size_t chunk = std::min(tensor->source_bytes - done,
                                    gpu->staging_bytes);
            if (descriptor >= 0) {
                char read_error[256] = {0};
                if (!read_exact(descriptor, gpu->staging, chunk,
                                tensor->source_offset + done,
                                read_error, sizeof(read_error))) {
                    h3cspeed_set_error(gpu, "file-backed weight read", read_error);
                    ok = 0;
                    break;
                }
            } else {
                memcpy(gpu->staging,
                       static_cast<const unsigned char *>(tensor->host_data) + done,
                       chunk);
            }
            if (!h3cspeed_cuda_ok(gpu,
                cudaMemcpyAsync(static_cast<unsigned char *>(tensor->data) + done,
                                gpu->staging, chunk, cudaMemcpyHostToDevice,
                                gpu->upload_stream),
                "staged host weight upload") ||
                !h3cspeed_cuda_ok(gpu,
                    cudaStreamSynchronize(gpu->upload_stream),
                    "staged host weight synchronization")) {
                ok = 0;
                break;
            }
            done += chunk;
        }
        if (descriptor >= 0) close(descriptor);
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
            "zero unused weight slot tail")) return 0;
    if (!h3cspeed_tensor_record_upload(tensor)) return 0;
    gpu->offload_uploads++;
    gpu->offload_upload_bytes += tensor->source_bytes;
    return 1;
}

int h3cspeed_tensor_prepare(h3_gpu *gpu, h3_gpu_tensor *tensor) {
    if (!gpu || !tensor || tensor->gpu != gpu) {
        if (gpu) h3cspeed_set_error(gpu, "tensor prepare", "invalid CUDA tensor");
        return 0;
    }
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
            if (tensor->data) (void)release_tensor_device_locked(gpu, tensor, 0, 0);
            pthread_mutex_unlock(&gpu->offload_lock);
            return 0;
        }
        lru_append_locked(gpu, tensor);
    } else if (tensor->offloadable) {
        lru_append_locked(gpu, tensor);
    }
    if (tensor->offloadable) tensor->pin_epoch = gpu->operation_epoch;
    pthread_mutex_unlock(&gpu->offload_lock);

    pthread_mutex_lock(&tensor->lock);
    int ready_valid = tensor->ready_valid;
    cudaEvent_t ready = tensor->ready;
    pthread_mutex_unlock(&tensor->lock);
    if (!ready_valid) return 1;
    return h3cspeed_cuda_ok(gpu,
        cudaStreamWaitEvent(gpu->compute_stream, ready, 0),
        "cudaStreamWaitEvent(weight ready)");
}

void h3cspeed_operation_complete(h3_gpu *gpu) {
    if (!gpu || !gpu->offload.enabled) return;
    pthread_mutex_lock(&gpu->offload_lock);
    uint64_t completed_epoch = gpu->operation_epoch;
    int all_recorded = 1;
    for (h3_gpu_tensor *tensor = gpu->lru_head; tensor;
         tensor = tensor->lru_next) {
        if (!tensor->data || tensor->pin_epoch != completed_epoch) continue;
        pthread_mutex_lock(&tensor->lock);
        cudaError_t status = cudaEventRecord(tensor->last_use,
                                              gpu->compute_stream);
        if (status == cudaSuccess) tensor->last_use_valid = 1;
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
    gpu->profile_enabled = environment_flag_enabled("H3_PROFILE");
    (void)cudaEventCreate(&gpu->profile_start);
    (void)cudaEventCreate(&gpu->profile_mark);
    if (error && error_size) error[0] = '\0';
    return gpu;
}

extern "C" void h3_gpu_free(h3_gpu *gpu) {
    if (!gpu) return;
    (void)cudaSetDevice(gpu->device);
    if (gpu->compute_stream) (void)cudaStreamSynchronize(gpu->compute_stream);
    if (gpu->upload_stream) (void)cudaStreamSynchronize(gpu->upload_stream);
    if (gpu->profile_enabled || gpu->offload.enabled) {
        fprintf(stderr,
            "h3cspeed CUDA%s%s%s: device-live=%.2f MiB peak=%.2f MiB "
            "resident-weights=%.2f MiB peak-resident=%.2f MiB "
            "host-cache=%.2f MiB peak-host=%.2f MiB "
            "uploads=%" PRIu64 "/%.2f GiB evictions=%" PRIu64
            "/%.2f GiB host-evictions=%" PRIu64 "/%.2f GiB "
            "file-fallback=%" PRIu64 "/%.2f GiB linear=%" PRIu64
            " conv=%" PRIu64 " sdpa=%" PRIu64 "\n",
            gpu->profile_label[0] ? " [" : "",
            gpu->profile_label[0] ? gpu->profile_label : "",
            gpu->profile_label[0] ? "]" : "",
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
    if (gpu->scratch) cudaFree(gpu->scratch);
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
            cudaMemcpyAsync(tensor->data, values, tensor->bytes,
                            cudaMemcpyHostToDevice, gpu->upload_stream),
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

static int read_exact(int descriptor, void *buffer, size_t bytes,
                      uint64_t offset, char *error, size_t error_size) {
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
            return 0;
        }
        if (got == 0) {
            if (error && error_size) snprintf(error, error_size,
                "unexpected end of weight file");
            return 0;
        }
        done += (size_t)got;
    }
    return 1;
}

static int tensor_synchronize_before_host_overwrite_locked(
        h3_gpu_tensor *tensor, const char *operation) {
    if (!tensor || !tensor->gpu) return 0;
    h3_gpu *gpu = tensor->gpu;
    if (!gpu->offload.enabled) {
        return h3cspeed_cuda_ok(gpu,
                   cudaStreamSynchronize(gpu->compute_stream), operation) &&
               h3cspeed_cuda_ok(gpu,
                   cudaStreamSynchronize(gpu->upload_stream), operation);
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
    if (tensor->data && !release_tensor_device_locked(gpu, tensor, 0, 1)) return 0;
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
                ok = read_exact(descriptor, tensor->host_data, bytes,
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
        if (!read_exact(descriptor, gpu->staging, chunk, file_offset + done,
                        error, error_size) ||
            !h3cspeed_cuda_ok(gpu,
                cudaMemcpyAsync(static_cast<unsigned char *>(tensor->data) + done,
                                gpu->staging, chunk, cudaMemcpyHostToDevice,
                                gpu->upload_stream),
                "stream weight upload") ||
            !h3cspeed_cuda_ok(gpu,
                cudaStreamSynchronize(gpu->upload_stream),
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
        (void)release_tensor_device_locked(gpu, tensor,
                                           tensor->offloadable ? 1 : 0, 0);
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
        !h3cspeed_cuda_ok(gpu, cudaStreamSynchronize(gpu->compute_stream),
                           "read tensor compute synchronization") ||
        !h3cspeed_cuda_ok(gpu, cudaStreamSynchronize(gpu->upload_stream),
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
        !h3cspeed_cuda_ok(gpu, cudaStreamSynchronize(gpu->compute_stream),
                           "synchronize tensor before host write") ||
        !h3cspeed_cuda_ok(gpu, cudaStreamSynchronize(gpu->upload_stream),
                           "synchronize upload before host write") ||
        !h3cspeed_cuda_ok(gpu,
            cudaMemcpyAsync(static_cast<unsigned char *>(tensor->data) +
                                destination_offset * h3cspeed_dtype_size(expected),
                            values, elements * h3cspeed_dtype_size(expected),
                            cudaMemcpyHostToDevice, gpu->upload_stream),
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
        (void)cudaEventRecord(gpu->profile_start, gpu->compute_stream);
    }
    return 1;
}

extern "C" int h3_gpu_continue(h3_gpu *gpu) {
    if (!gpu) return 0;
    pthread_mutex_lock(&gpu->lock);
    gpu->stats.submissions++;
    pthread_mutex_unlock(&gpu->lock);
    return 1;
}

extern "C" int h3_gpu_submit(h3_gpu *gpu) {
    if (!gpu) return 0;
    h3cspeed_operation_complete(gpu);
    struct timespec start, stop;
    clock_gettime(CLOCK_MONOTONIC, &start);
    int ok = h3cspeed_cuda_ok(gpu, cudaStreamSynchronize(gpu->compute_stream),
                               "cudaStreamSynchronize(compute)") &&
             h3cspeed_cuda_ok(gpu, cudaStreamSynchronize(gpu->upload_stream),
                               "cudaStreamSynchronize(upload)");
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
    snprintf(gpu->profile_label, sizeof(gpu->profile_label), "%s", label ? label : "");
}

extern "C" void h3_gpu_profile_mark(h3_gpu *gpu, const char *phase) {
    if (!gpu || !gpu->profile_enabled) return;
    (void)cudaEventRecord(gpu->profile_mark, gpu->compute_stream);
    (void)cudaEventSynchronize(gpu->profile_mark);
    float milliseconds = 0.0f;
    (void)cudaEventElapsedTime(&milliseconds, gpu->profile_start, gpu->profile_mark);
    fprintf(stderr, "h3cspeed CUDA%s%s%s: %s %.3f s\n",
            gpu->profile_label[0] ? " [" : "",
            gpu->profile_label[0] ? gpu->profile_label : "",
            gpu->profile_label[0] ? "]" : "",
            phase ? phase : "mark", milliseconds / 1000.0f);
}

void *h3cspeed_scratch_reserve(h3_gpu *gpu, size_t bytes) {
    if (!gpu || !bytes) return nullptr;
    pthread_mutex_lock(&gpu->scratch_lock);
    if (bytes > gpu->scratch_bytes) {
        if (!h3cspeed_cuda_ok(gpu, cudaStreamSynchronize(gpu->compute_stream),
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
            cudaStreamSynchronize(gpu->compute_stream),
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
    return h3cspeed_quantize_rows(gpu, quantized_input, input_scales, input,
                                 rows, input_dim, 0, 0, 0) &&
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
    return h3cspeed_quantize_rows(gpu, quantized_input, input_scales, input,
                                 rows, width, 1, heads, head_dim) &&
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
                 weight_scales, rows, input_dim, qkv_dim, 0) &&
             h3_gpu_grouped_qkv_rope_bf16(
                 gpu, query, key, value, qkv, q_norm, k_norm,
                 rope_cos, rope_sin, rows, heads, head_dim, rope_half,
                 epsilon);
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
