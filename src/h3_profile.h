#ifndef H3CSPEED_PROFILE_H
#define H3CSPEED_PROFILE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define H3CSPEED_PROFILE_SCHEMA_VERSION 1u

typedef enum h3cspeed_profile_eviction_reason {
    H3CSPEED_PROFILE_EVICTION_CAPACITY_LRU = 0,
    H3CSPEED_PROFILE_EVICTION_PHASE_RETIRE = 1,
    H3CSPEED_PROFILE_EVICTION_ERROR_CLEANUP = 2
} h3cspeed_profile_eviction_reason;

typedef struct h3cspeed_profile_metrics {
    uint64_t begin_count;
    uint64_t continue_count;
    uint64_t submit_sync_count;
    uint64_t last_use_fence_count;
    uint64_t file_read_calls;
    uint64_t file_read_bytes;
    uint64_t pageable_copy_calls;
    uint64_t pageable_copy_bytes;
    uint64_t h2d_enqueue_calls;
    uint64_t h2d_enqueue_bytes;
    uint64_t compute_stream_syncs;
    uint64_t upload_stream_syncs;
    uint64_t event_syncs;
    uint64_t allocation_calls;
    uint64_t capacity_evictions;
    uint64_t capacity_evicted_bytes;
    uint64_t phase_retire_frees;
    uint64_t phase_retired_bytes;
    uint64_t error_cleanup_frees;
    uint64_t error_cleanup_bytes;
    double file_read_seconds;
    double pageable_copy_seconds;
    double h2d_enqueue_seconds;
    double compute_stream_wait_seconds;
    double upload_stream_wait_seconds;
    double event_wait_seconds;
    double allocation_seconds;
    double eviction_seconds;
    double compute_device_seconds;
} h3cspeed_profile_metrics;

typedef struct h3cspeed_profile_report {
    uint64_t context_id;
    int device;
    int sm_major;
    int sm_minor;
    const char *label;
    int complete;
    double wall_seconds;
    h3cspeed_profile_metrics metrics;
    uint64_t device_peak_bytes;
    uint64_t resident_weight_peak_bytes;
    uint64_t host_cache_peak_bytes;
    uint64_t offload_uploads;
    uint64_t offload_upload_bytes;
    uint64_t offload_evictions;
    uint64_t offload_evicted_bytes;
    uint64_t file_fallback_reads;
    uint64_t file_fallback_bytes;
    uint64_t direct_dispatches;
    uint64_t linear_dispatches;
    uint64_t convolution_dispatches;
    uint64_t attention_dispatches;
} h3cspeed_profile_report;

void h3cspeed_profile_metrics_init(h3cspeed_profile_metrics *metrics);
double h3cspeed_profile_now_seconds(void);
int h3cspeed_profile_record_eviction(h3cspeed_profile_metrics *metrics,
                                    h3cspeed_profile_eviction_reason reason,
                                    uint64_t bytes, double seconds);
int h3cspeed_profile_report_valid(const h3cspeed_profile_report *report);
const char *h3cspeed_profile_safe_label(const char *label);
const char *h3cspeed_profile_safe_phase(const char *phase);

/* Writes one unique JSON file to a caller-owned existing directory. The
 * temporary file is flushed and atomically published without replacement. */
int h3cspeed_profile_write_json_directory(
    const char *directory, const h3cspeed_profile_report *report,
    char *written_path, size_t written_path_size,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
