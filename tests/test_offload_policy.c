#include "h3_offload_policy.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIB UINT64_C(1048576)
#define GIB UINT64_C(1073741824)

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", \
                __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

static void test_3070ti_auto_profile(void) {
    h3cspeed_offload_policy policy;
    char error[256];
    h3cspeed_offload_overrides values = {0};
    CHECK(h3cspeed_offload_policy_resolve(
        8 * GIB, 7500 * MIB, 96 * GIB, &values,
        &policy, error, sizeof(error)));
    CHECK(policy.enabled);
    CHECK(policy.low_vram);
    CHECK(policy.mode == H3CSPEED_OFFLOAD_RAM_FILE);
    CHECK(policy.vram_budget_bytes >= 5 * GIB);
    CHECK(policy.vram_budget_bytes <= 6500 * MIB);
    CHECK(policy.weight_cache_bytes == 1536 * MIB);
    CHECK(policy.host_cache_bytes > 50 * GIB);
    CHECK(policy.pinned_host_bytes == 128 * MIB);
    CHECK(policy.release_scratch_on_submit);
}

static void test_large_card_stays_resident_by_default(void) {
    h3cspeed_offload_policy policy;
    char error[256];
    h3cspeed_offload_overrides values = {0};
    CHECK(h3cspeed_offload_policy_resolve(
        24 * GIB, 23 * GIB, 64 * GIB, &values,
        &policy, error, sizeof(error)));
    CHECK(!policy.enabled);
    CHECK(policy.mode == H3CSPEED_OFFLOAD_DISABLED);
    CHECK(policy.weight_cache_bytes == 0);
}

static void test_explicit_overrides_and_clamping(void) {
    h3cspeed_offload_policy policy;
    char error[256];
    h3cspeed_offload_overrides values = {
        "ram", "7000", "2048", "32768", "128", "32", "0"
    };
    CHECK(h3cspeed_offload_policy_resolve(
        8 * GIB, 6 * GIB, 64 * GIB, &values,
        &policy, error, sizeof(error)));
    CHECK(policy.enabled);
    CHECK(policy.vram_budget_bytes == 5 * GIB);
    CHECK(policy.weight_cache_bytes == 2 * GIB);
    CHECK(policy.host_cache_bytes == 32 * GIB);
    CHECK(policy.pinned_host_bytes == 128 * MIB);
    CHECK(policy.staging_bytes == 32 * MIB);
    CHECK(!policy.release_scratch_on_submit);
}

static void test_host_cache_leaves_operating_headroom(void) {
    h3cspeed_offload_policy policy;
    char error[256];
    h3cspeed_offload_overrides values = {
        "ram", NULL, NULL, "64000", NULL, NULL, NULL
    };
    CHECK(h3cspeed_offload_policy_resolve(
        8 * GIB, 7 * GIB, 8 * GIB, &values,
        &policy, error, sizeof(error)));
    CHECK(policy.host_cache_bytes == 6 * GIB);
    CHECK(policy.pinned_host_bytes == 128 * MIB);
}

static void test_unsafe_minimum_budgets_are_rejected(void) {
    h3cspeed_offload_policy policy;
    char error[256];
    h3cspeed_offload_overrides small_vram = {
        "ram", "256", "128", NULL, NULL, NULL, NULL
    };
    CHECK(!h3cspeed_offload_policy_resolve(
        8 * GIB, 7 * GIB, 32 * GIB, &small_vram,
        &policy, error, sizeof(error)));
    CHECK(strstr(error, "VRAM_BUDGET") != NULL);

    h3cspeed_offload_overrides small_weights = {
        "ram", "2048", "64", NULL, NULL, NULL, NULL
    };
    CHECK(!h3cspeed_offload_policy_resolve(
        8 * GIB, 7 * GIB, 32 * GIB, &small_weights,
        &policy, error, sizeof(error)));
    CHECK(strstr(error, "WEIGHT_CACHE") != NULL);
}


static void test_mode_and_boolean_are_case_insensitive(void) {
    h3cspeed_offload_policy policy;
    char error[256];
    h3cspeed_offload_overrides values = {
        "RAM+FILE", NULL, NULL, NULL, NULL, NULL, "OFF"
    };
    CHECK(h3cspeed_offload_policy_resolve(
        8 * GIB, 7 * GIB, 32 * GIB, &values,
        &policy, error, sizeof(error)));
    CHECK(policy.enabled);
    CHECK(!policy.release_scratch_on_submit);
}

static void test_bad_mode_is_rejected(void) {
    h3cspeed_offload_policy policy;
    char error[256];
    h3cspeed_offload_overrides values = {0};
    values.mode = "managed-memory";
    CHECK(!h3cspeed_offload_policy_resolve(
        8 * GIB, 7 * GIB, 32 * GIB, &values,
        &policy, error, sizeof(error)));
    CHECK(strstr(error, "H3_CUDA_OFFLOAD") != NULL);
}

int main(void) {
    test_3070ti_auto_profile();
    test_large_card_stays_resident_by_default();
    test_explicit_overrides_and_clamping();
    test_host_cache_leaves_operating_headroom();
    test_unsafe_minimum_budgets_are_rejected();
    test_mode_and_boolean_are_case_insensitive();
    test_bad_mode_is_rejected();
    puts("offload policy tests passed");
    return 0;
}
