#include "h3_metal.h"

#include <cuda_runtime.h>

#if defined(_WIN32)
#include "h3_msvc_compat.h"
#endif

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

extern "C" int h3_metal_probe(h3_device_info *info, char *error,
                               size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!info) return 0;
    memset(info, 0, sizeof(*info));
    int device = 0;
#if defined(_WIN32) && defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
    const char *device_env = getenv("H3_CUDA_DEVICE");
#if defined(_WIN32) && defined(__clang__)
#pragma clang diagnostic pop
#endif
    cudaError_t status;
    if (device_env && *device_env) {
        char *end = NULL;
        errno = 0;
        long parsed = strtol(device_env, &end, 10);
        if (errno || end == device_env || *end || parsed < 0 || parsed > INT_MAX) {
            if (error && error_size)
                snprintf(error, error_size,
                         "H3_CUDA_DEVICE must be a non-negative device index");
            return 0;
        }
        device = (int)parsed;
        status = cudaSetDevice(device);
    } else {
        status = cudaGetDevice(&device);
    }
    if (status != cudaSuccess) {
        if (error && error_size) snprintf(error, error_size, "CUDA: %s",
                                           cudaGetErrorString(status));
        return 0;
    }
    cudaDeviceProp properties;
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
        if (error && error_size) snprintf(error, error_size, "CUDA properties: %s",
                                           cudaGetErrorString(status));
        return 0;
    }
    if (properties.major < 8) {
        if (error && error_size)
            snprintf(error, error_size,
                     "CUDA device %d is sm_%d%d; h3cspeed requires sm_80 or newer",
                     device, properties.major, properties.minor);
        return 0;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    (void)cudaMemGetInfo(&free_bytes, &total_bytes);
#if defined(_WIN32)
    MEMORYSTATUSEX memory;
    memset(&memory, 0, sizeof(memory));
    memory.dwLength = sizeof(memory);
    uint64_t physical = GlobalMemoryStatusEx(&memory) ?
        (uint64_t)memory.ullTotalPhys : (uint64_t)total_bytes;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    uint64_t physical = pages > 0 && page_size > 0 ?
        (uint64_t)pages * (uint64_t)page_size : (uint64_t)total_bytes;
#endif

    snprintf(info->name, sizeof(info->name), "%s", properties.name);
    snprintf(info->architecture, sizeof(info->architecture), "NVIDIA CUDA sm_%d%d",
             properties.major, properties.minor);
    info->physical_memory = physical;
    info->recommended_working_set = (uint64_t)total_bytes;
    info->max_buffer_length = (uint64_t)free_bytes;
    info->apple_gpu_family = 0;
    info->metal4 = 0;
    /* CUDA managed-memory support is not the same as physically unified
     * CPU/GPU memory. Keep the upstream field false on discrete NVIDIA GPUs. */
    info->unified_memory = 0;
    return 1;
}
