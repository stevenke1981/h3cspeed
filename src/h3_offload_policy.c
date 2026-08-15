#include "h3_offload_policy.h"

#include <ctype.h>
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

#define H3_MIB UINT64_C(1048576)
#define H3_GIB UINT64_C(1073741824)

static uint64_t min_u64(uint64_t left, uint64_t right) {
    return left < right ? left : right;
}

static int text_equal(const char *left, const char *right) {
    if (!left || !right) return 0;
    while (*left && *right) {
        if (tolower((unsigned char)*left) !=
            tolower((unsigned char)*right)) return 0;
        left++;
        right++;
    }
    return *left == '\0' && *right == '\0';
}

static int text_true(const char *value) {
    return text_equal(value, "1") || text_equal(value, "on") ||
           text_equal(value, "true") || text_equal(value, "yes");
}

static int text_false(const char *value) {
    return text_equal(value, "0") || text_equal(value, "off") ||
           text_equal(value, "false") || text_equal(value, "no");
}

static int parse_mib(const char *name, const char *text, uint64_t *bytes,
                     char *error, size_t error_size) {
    if (!text || !*text) return 1;
    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno || end == text || *end || value > UINT64_MAX / H3_MIB) {
        if (error && error_size)
            snprintf(error, error_size, "%s must be an integer number of MiB", name);
        return 0;
    }
    *bytes = (uint64_t)value * H3_MIB;
    return 1;
}

static int parse_bool(const char *name, const char *text, int fallback,
                      int *value, char *error, size_t error_size) {
    if (!text || !*text) {
        *value = fallback;
        return 1;
    }
    if (text_true(text)) {
        *value = 1;
        return 1;
    }
    if (text_false(text)) {
        *value = 0;
        return 1;
    }
    if (error && error_size)
        snprintf(error, error_size, "%s must be 0/1, off/on, false/true, or no/yes", name);
    return 0;
}

uint64_t h3cspeed_system_memory_available_bytes(void) {
#if defined(_WIN32)
    MEMORYSTATUSEX status = {0};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) return (uint64_t)status.ullAvailPhys;
    return 0;
#else
    FILE *source = fopen("/proc/meminfo", "r");
    if (source) {
        char line[256];
        while (fgets(line, sizeof(line), source)) {
            unsigned long long kib = 0;
            if (sscanf(line, "MemAvailable: %llu kB", &kib) == 1) {
                fclose(source);
                if (kib <= UINT64_MAX / UINT64_C(1024))
                    return (uint64_t)kib * UINT64_C(1024);
                return UINT64_MAX;
            }
        }
        fclose(source);
    }
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages <= 0) pages = sysconf(_SC_PHYS_PAGES);
    if (pages <= 0 || page_size <= 0) return 0;
    if ((uint64_t)pages > UINT64_MAX / (uint64_t)page_size) return UINT64_MAX;
    return (uint64_t)pages * (uint64_t)page_size;
#endif
}

const char *h3cspeed_offload_mode_name(h3cspeed_offload_mode mode) {
    switch (mode) {
        case H3CSPEED_OFFLOAD_DISABLED: return "disabled";
        case H3CSPEED_OFFLOAD_RAM_FILE: return "system-RAM + file fallback";
    }
    return "unknown";
}

int h3cspeed_offload_policy_resolve(
    uint64_t total_vram_bytes,
    uint64_t free_vram_bytes,
    uint64_t system_memory_bytes,
    const h3cspeed_offload_overrides *overrides,
    h3cspeed_offload_policy *policy,
    char *error,
    size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!policy || !total_vram_bytes) {
        if (error && error_size)
            snprintf(error, error_size, "offload policy needs non-zero VRAM and output storage");
        return 0;
    }
    memset(policy, 0, sizeof(*policy));
    policy->total_vram_bytes = total_vram_bytes;
    policy->free_vram_bytes = free_vram_bytes ? free_vram_bytes : total_vram_bytes;
    policy->system_memory_bytes = system_memory_bytes;
    policy->low_vram = total_vram_bytes <= UINT64_C(10) * H3_GIB;

    const char *mode = overrides ? overrides->mode : NULL;
    policy->automatic = !mode || !*mode || text_equal(mode, "auto");
    if (policy->automatic) {
        policy->enabled = policy->low_vram;
    } else if (text_equal(mode, "1") || text_equal(mode, "on") ||
               text_equal(mode, "ram") || text_equal(mode, "system") ||
               text_equal(mode, "ram+file")) {
        policy->enabled = 1;
    } else if (text_equal(mode, "0") || text_equal(mode, "off") ||
               text_equal(mode, "none") || text_equal(mode, "disabled")) {
        policy->enabled = 0;
    } else {
        if (error && error_size)
            snprintf(error, error_size,
                     "H3_CUDA_OFFLOAD must be auto, ram, ram+file, on, or off");
        return 0;
    }
    policy->mode = policy->enabled ? H3CSPEED_OFFLOAD_RAM_FILE :
                                     H3CSPEED_OFFLOAD_DISABLED;

    policy->safety_margin_bytes = policy->low_vram ? H3_GIB :
                                                     UINT64_C(768) * H3_MIB;
    uint64_t total_limit = total_vram_bytes > policy->safety_margin_bytes ?
        total_vram_bytes - policy->safety_margin_bytes : total_vram_bytes * 3 / 4;
    uint64_t free_limit = policy->free_vram_bytes > policy->safety_margin_bytes ?
        policy->free_vram_bytes - policy->safety_margin_bytes :
        policy->free_vram_bytes * 3 / 4;
    uint64_t percentage_limit = total_vram_bytes *
        (policy->low_vram ? UINT64_C(78) : UINT64_C(90)) / 100;
    policy->vram_budget_bytes = min_u64(min_u64(total_limit, free_limit),
                                        percentage_limit);
    if (policy->vram_budget_bytes < UINT64_C(512) * H3_MIB)
        policy->vram_budget_bytes = min_u64(total_limit, UINT64_C(512) * H3_MIB);

    uint64_t default_weight_cache = policy->low_vram ?
        UINT64_C(1536) * H3_MIB : UINT64_C(4096) * H3_MIB;
    policy->weight_cache_bytes = min_u64(default_weight_cache,
                                         policy->vram_budget_bytes / 3);

    /* The one-ahead DiT path rereads the same file-backed weights when its
     * conservative cache fills.  It is opt-in and already bounded by the
     * hard headroom clamp below, so let that path use more of currently
     * available RAM by default.  An explicit H3_CUDA_HOST_CACHE_MIB override
     * still wins and is clamped to the same safety limit. */
    uint64_t host_cache_percent =
        getenv("H3_CUDA_DIT_PREFETCH") &&
        !strcmp(getenv("H3_CUDA_DIT_PREFETCH"), "1") ? 85u : 60u;
    uint64_t default_host_cache =
        system_memory_bytes * host_cache_percent / 100;
    policy->host_cache_bytes = min_u64(default_host_cache,
                                       UINT64_C(64) * H3_GIB);
    policy->pinned_host_bytes = min_u64(policy->host_cache_bytes,
        (policy->low_vram ? UINT64_C(128) : UINT64_C(512)) * H3_MIB);
    policy->staging_bytes = UINT64_C(64) * H3_MIB;
    policy->release_scratch_on_submit = policy->low_vram;

    if (overrides) {
        if (!parse_mib("H3_CUDA_VRAM_BUDGET_MIB", overrides->vram_budget_mib,
                       &policy->vram_budget_bytes, error, error_size) ||
            !parse_mib("H3_CUDA_WEIGHT_CACHE_MIB", overrides->weight_cache_mib,
                       &policy->weight_cache_bytes, error, error_size) ||
            !parse_mib("H3_CUDA_HOST_CACHE_MIB", overrides->host_cache_mib,
                       &policy->host_cache_bytes, error, error_size) ||
            !parse_mib("H3_CUDA_PINNED_HOST_MIB", overrides->pinned_host_mib,
                       &policy->pinned_host_bytes, error, error_size) ||
            !parse_mib("H3_CUDA_STAGING_MIB", overrides->staging_mib,
                       &policy->staging_bytes, error, error_size) ||
            !parse_bool("H3_CUDA_RELEASE_SCRATCH", overrides->release_scratch,
                        policy->release_scratch_on_submit,
                        &policy->release_scratch_on_submit, error, error_size))
            return 0;
    }

    /* Keep enough pageable host memory for H3 metadata, FFmpeg, the OS and
     * WSL2 itself. An explicit cache request is therefore a ceiling, not a
     * promise to consume memory that was not available at context creation. */
    uint64_t host_hard_limit = system_memory_bytes > UINT64_C(2) * H3_GIB ?
        system_memory_bytes - UINT64_C(2) * H3_GIB :
        system_memory_bytes * UINT64_C(75) / UINT64_C(100);
    if (policy->host_cache_bytes > host_hard_limit)
        policy->host_cache_bytes = host_hard_limit;
    /* An explicit budget is still clamped to memory that was actually free at
     * context creation. This prevents a Windows/WSL display workload from
     * turning a nominal 8 GiB setting into immediate allocation failures. */
    uint64_t hard_limit = min_u64(total_limit, free_limit);
    if (policy->vram_budget_bytes > hard_limit)
        policy->vram_budget_bytes = hard_limit;
    if (policy->weight_cache_bytes > policy->vram_budget_bytes)
        policy->weight_cache_bytes = policy->vram_budget_bytes;
    if (policy->pinned_host_bytes > policy->host_cache_bytes)
        policy->pinned_host_bytes = policy->host_cache_bytes;
    if (policy->staging_bytes < UINT64_C(4) * H3_MIB)
        policy->staging_bytes = UINT64_C(4) * H3_MIB;
    if (policy->staging_bytes > UINT64_C(512) * H3_MIB)
        policy->staging_bytes = UINT64_C(512) * H3_MIB;

    if (policy->enabled &&
        policy->vram_budget_bytes < UINT64_C(512) * H3_MIB) {
        if (error && error_size)
            snprintf(error, error_size,
                     "H3_CUDA_VRAM_BUDGET_MIB must leave at least 512 MiB "
                     "for tracked CUDA allocations");
        return 0;
    }
    if (policy->enabled &&
        policy->weight_cache_bytes < UINT64_C(128) * H3_MIB) {
        if (error && error_size)
            snprintf(error, error_size,
                     "H3_CUDA_WEIGHT_CACHE_MIB must be at least 128 MiB "
                     "when offload is enabled");
        return 0;
    }

    if (!policy->enabled) {
        policy->weight_cache_bytes = 0;
        policy->host_cache_bytes = 0;
        policy->pinned_host_bytes = 0;
        policy->release_scratch_on_submit = 0;
    }
    return 1;
}

int h3cspeed_offload_policy_from_env(
    uint64_t total_vram_bytes,
    uint64_t free_vram_bytes,
    uint64_t system_memory_bytes,
    h3cspeed_offload_policy *policy,
    char *error,
    size_t error_size) {
    const char *mode = getenv("H3_CUDA_OFFLOAD");
    const char *low_vram = getenv("H3_CUDA_LOW_VRAM");
    if ((!mode || !*mode) && low_vram && *low_vram) {
        int force_low_vram = 0;
        if (!parse_bool("H3_CUDA_LOW_VRAM", low_vram, 0,
                        &force_low_vram, error, error_size)) return 0;
        if (force_low_vram) mode = "ram";
    }
    h3cspeed_offload_overrides overrides = {
        mode,
        getenv("H3_CUDA_VRAM_BUDGET_MIB"),
        getenv("H3_CUDA_WEIGHT_CACHE_MIB"),
        getenv("H3_CUDA_HOST_CACHE_MIB"),
        getenv("H3_CUDA_PINNED_HOST_MIB"),
        getenv("H3_CUDA_STAGING_MIB"),
        getenv("H3_CUDA_RELEASE_SCRATCH"),
    };
    return h3cspeed_offload_policy_resolve(
        total_vram_bytes, free_vram_bytes, system_memory_bytes,
        &overrides, policy, error, error_size);
}
