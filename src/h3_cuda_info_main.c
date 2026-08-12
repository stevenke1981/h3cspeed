#include "h3.h"
#include "h3_metal.h"
#include "h3_offload_policy.h"

#include <inttypes.h>
#include <stdio.h>

static double gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static double mib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0);
}

int main(void) {
    h3_device_info info;
    char error[512];
    if (!h3_metal_probe(&info, error, sizeof(error))) {
        fprintf(stderr, "CUDA probe failed: %s\n", error);
        return 1;
    }
    printf("device: %s\n", info.name);
    printf("architecture: %s\n", info.architecture);
    printf("system memory: %.2f GiB\n", gib(info.physical_memory));
    printf("VRAM total: %.2f GiB\n", gib(info.recommended_working_set));
    printf("VRAM currently free: %.2f GiB\n", gib(info.max_buffer_length));
    printf("physically unified CPU/GPU memory: %s\n",
           info.unified_memory ? "yes" : "no");

    uint64_t available_host = h3cspeed_system_memory_available_bytes();
    printf("host memory currently available: %.2f GiB\n", gib(available_host));

    h3cspeed_offload_policy policy;
    if (!h3cspeed_offload_policy_from_env(
            info.recommended_working_set, info.max_buffer_length,
            available_host, &policy, error, sizeof(error))) {
        fprintf(stderr, "offload policy error: %s\n", error);
        return 1;
    }
    printf("offload mode: %s%s\n",
           h3cspeed_offload_mode_name(policy.mode),
           policy.automatic ? " (automatic)" : "");
    printf("low-VRAM profile: %s\n", policy.low_vram ? "yes" : "no");
    printf("CUDA allocation budget: %.2f GiB\n", gib(policy.vram_budget_bytes));
    if (policy.enabled) {
        printf("resident GPU weight cache: %.2f GiB\n",
               gib(policy.weight_cache_bytes));
        printf("system-RAM weight cache: %.2f GiB\n",
               gib(policy.host_cache_bytes));
        printf("pinned host-memory cap: %.0f MiB\n",
               mib(policy.pinned_host_bytes));
        printf("RAM/file transfer staging: %.0f MiB\n",
               mib(policy.staging_bytes));
        printf("release scratch at submit: %s\n",
               policy.release_scratch_on_submit ? "yes" : "no");
    }
    return 0;
}
