#include "h3_profile.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: test_profile_report OUTPUT_DIRECTORY\n");
        return 2;
    }
    h3cspeed_profile_report report;
    memset(&report, 0, sizeof(report));
    report.context_id = 4242;
    report.device = 0;
    report.sm_major = 8;
    report.sm_minor = 6;
    report.label = "unit \"profile\"\nno-path";
    report.complete = 1;
    report.wall_seconds = 4.0;
    h3cspeed_profile_metrics_init(&report.metrics);
    report.metrics.begin_count = 2;
    report.metrics.continue_count = 3;
    report.metrics.submit_sync_count = 1;
    report.metrics.last_use_fence_count = 7;
    report.metrics.file_read_calls = 2;
    report.metrics.file_read_bytes = 4096;
    report.metrics.file_read_seconds = 1.5;
    report.metrics.pageable_copy_calls = 1;
    report.metrics.pageable_copy_bytes = 2048;
    report.metrics.pageable_copy_seconds = 0.25;
    report.metrics.h2d_enqueue_calls = 1;
    report.metrics.h2d_enqueue_bytes = 2048;
    report.metrics.h2d_enqueue_seconds = 0.05;
    report.metrics.compute_stream_syncs = 1;
    report.metrics.compute_stream_wait_seconds = 1.0;
    report.metrics.upload_stream_syncs = 1;
    report.metrics.upload_stream_wait_seconds = 0.5;
    report.metrics.event_syncs = 1;
    report.metrics.event_wait_seconds = 0.25;
    report.metrics.allocation_calls = 1;
    report.metrics.allocation_seconds = 0.1;
    report.metrics.compute_device_seconds = 0.75;
    if (!h3cspeed_profile_record_eviction(
            &report.metrics, H3CSPEED_PROFILE_EVICTION_CAPACITY_LRU,
            100, 0.01) ||
        !h3cspeed_profile_record_eviction(
            &report.metrics, H3CSPEED_PROFILE_EVICTION_PHASE_RETIRE,
            200, 0.02) ||
        !h3cspeed_profile_record_eviction(
            &report.metrics, H3CSPEED_PROFILE_EVICTION_ERROR_CLEANUP,
            300, 0.03) ||
        h3cspeed_profile_record_eviction(
            &report.metrics, (h3cspeed_profile_eviction_reason)99,
            400, 0.04)) {
        fprintf(stderr, "eviction reducer rejected a valid reason\n");
        return 3;
    }
    report.device_peak_bytes = 1024;
    report.resident_weight_peak_bytes = 512;
    report.host_cache_peak_bytes = 2048;
    report.offload_uploads = 4;
    report.offload_upload_bytes = 4096;
    report.offload_evictions = 2;
    report.offload_evicted_bytes = 1024;
    report.file_fallback_reads = 1;
    report.file_fallback_bytes = 512;
    report.direct_dispatches = 3;
    report.linear_dispatches = 4;
    report.convolution_dispatches = 5;
    report.attention_dispatches = 6;

    if (strcmp(h3cspeed_profile_safe_label(report.label), "redacted") != 0 ||
        strcmp(h3cspeed_profile_safe_label("H3 DiT"), "H3 DiT") != 0 ||
        strcmp(h3cspeed_profile_safe_phase("private model path"), "redacted") != 0 ||
        strcmp(h3cspeed_profile_safe_phase("load"), "load") != 0) {
        fprintf(stderr, "profile label redaction failed\n");
        return 4;
    }

    char path[1024] = {0};
    char error[256] = {0};
    if (!h3cspeed_profile_write_json_directory(
            argv[1], &report, path, sizeof(path), error, sizeof(error))) {
        fprintf(stderr, "profile write failed: %s\n", error);
        return 5;
    }
    printf("%s\n", path);
    return 0;
}
