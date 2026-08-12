#ifndef H3CSPEED_OFFLOAD_POLICY_H
#define H3CSPEED_OFFLOAD_POLICY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    H3CSPEED_OFFLOAD_DISABLED = 0,
    H3CSPEED_OFFLOAD_RAM_FILE = 1
} h3cspeed_offload_mode;

typedef struct {
    const char *mode;
    const char *vram_budget_mib;
    const char *weight_cache_mib;
    const char *host_cache_mib;
    const char *pinned_host_mib;
    const char *staging_mib;
    const char *release_scratch;
} h3cspeed_offload_overrides;

typedef struct {
    h3cspeed_offload_mode mode;
    int enabled;
    int automatic;
    int low_vram;
    int release_scratch_on_submit;
    uint64_t total_vram_bytes;
    uint64_t free_vram_bytes;
    uint64_t system_memory_bytes;
    uint64_t safety_margin_bytes;
    uint64_t vram_budget_bytes;
    uint64_t weight_cache_bytes;
    uint64_t host_cache_bytes;
    uint64_t pinned_host_bytes;
    uint64_t staging_bytes;
} h3cspeed_offload_policy;

int h3cspeed_offload_policy_resolve(
    uint64_t total_vram_bytes,
    uint64_t free_vram_bytes,
    uint64_t system_memory_bytes,
    const h3cspeed_offload_overrides *overrides,
    h3cspeed_offload_policy *policy,
    char *error,
    size_t error_size);

int h3cspeed_offload_policy_from_env(
    uint64_t total_vram_bytes,
    uint64_t free_vram_bytes,
    uint64_t system_memory_bytes,
    h3cspeed_offload_policy *policy,
    char *error,
    size_t error_size);

const char *h3cspeed_offload_mode_name(h3cspeed_offload_mode mode);

/* Linux/WSL MemAvailable when present, with a POSIX page-count fallback. */
uint64_t h3cspeed_system_memory_available_bytes(void);

#ifdef __cplusplus
}
#endif

#endif
