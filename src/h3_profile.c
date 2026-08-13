#include "h3_profile.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#if defined(_WIN32)
#include <io.h>
#include <process.h>
#include <windows.h>
#define H3CSPEED_GETPID _getpid
#define H3CSPEED_FILENO _fileno
#define H3CSPEED_FSYNC _commit
#else
#include <unistd.h>
#define H3CSPEED_GETPID getpid
#define H3CSPEED_FILENO fileno
#define H3CSPEED_FSYNC fsync
#endif

void h3cspeed_profile_metrics_init(h3cspeed_profile_metrics *metrics) {
    if (metrics) memset(metrics, 0, sizeof(*metrics));
}

double h3cspeed_profile_now_seconds(void) {
#if defined(_WIN32)
    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    if (!QueryPerformanceFrequency(&frequency) || !frequency.QuadPart ||
        !QueryPerformanceCounter(&counter)) return 0.0;
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec / 1e9;
#endif
}

int h3cspeed_profile_record_eviction(h3cspeed_profile_metrics *metrics,
                                    h3cspeed_profile_eviction_reason reason,
                                    uint64_t bytes, double seconds) {
    if (!metrics || !isfinite(seconds) || seconds < 0.0) return 0;
    switch (reason) {
        case H3CSPEED_PROFILE_EVICTION_CAPACITY_LRU:
            metrics->capacity_evictions++;
            metrics->capacity_evicted_bytes += bytes;
            break;
        case H3CSPEED_PROFILE_EVICTION_PHASE_RETIRE:
            metrics->phase_retire_frees++;
            metrics->phase_retired_bytes += bytes;
            break;
        case H3CSPEED_PROFILE_EVICTION_ERROR_CLEANUP:
            metrics->error_cleanup_frees++;
            metrics->error_cleanup_bytes += bytes;
            break;
        default:
            return 0;
    }
    metrics->eviction_seconds += seconds;
    return 1;
}

static int finite_nonnegative(double value) {
    return isfinite(value) && value >= 0.0;
}

int h3cspeed_profile_report_valid(const h3cspeed_profile_report *report) {
    if (!report || !report->context_id || report->device < 0 ||
        report->sm_major < 0 || report->sm_minor < 0 ||
        !report->label || !finite_nonnegative(report->wall_seconds)) return 0;
    const h3cspeed_profile_metrics *metrics = &report->metrics;
    return finite_nonnegative(metrics->file_read_seconds) &&
           finite_nonnegative(metrics->pageable_copy_seconds) &&
           finite_nonnegative(metrics->h2d_enqueue_seconds) &&
           finite_nonnegative(metrics->compute_stream_wait_seconds) &&
           finite_nonnegative(metrics->upload_stream_wait_seconds) &&
           finite_nonnegative(metrics->event_wait_seconds) &&
           finite_nonnegative(metrics->allocation_seconds) &&
           finite_nonnegative(metrics->eviction_seconds) &&
           finite_nonnegative(metrics->compute_device_seconds);
}

const char *h3cspeed_profile_safe_label(const char *label) {
    if (!label) return "redacted";
    if (strcmp(label, "H3 DiT") == 0) return "H3 DiT";
    if (strcmp(label, "Qwen text encoder") == 0) return "Qwen text encoder";
    if (strcmp(label, "Qwen vision encoder") == 0) return "Qwen vision encoder";
    if (strcmp(label, "video VAE encoder") == 0) return "video VAE encoder";
    if (strcmp(label, "video VAE decoder") == 0) return "video VAE decoder";
    if (strcmp(label, "resident video VAE decoder") == 0)
        return "resident video VAE decoder";
    if (strcmp(label, "audio VAE encoder") == 0) return "audio VAE encoder";
    if (strcmp(label, "audio VAE decoder") == 0) return "audio VAE decoder";
    return "redacted";
}

const char *h3cspeed_profile_safe_phase(const char *phase) {
    if (!phase) return "redacted";
    if (strcmp(phase, "load") == 0) return "load";
    if (strcmp(phase, "GPU Euler denoise") == 0) return "GPU Euler denoise";
    if (strcmp(phase, "RES denoise") == 0) return "RES denoise";
    if (strcmp(phase, "Euler denoise") == 0) return "Euler denoise";
    return "redacted";
}

static void set_error(char *error, size_t error_size, const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
}

static void sanitize_label(const char *input, char *output, size_t output_size) {
    size_t used = 0;
    if (!output || !output_size) return;
    for (const unsigned char *cursor = (const unsigned char *)(input ? input : "gpu");
         *cursor && used + 1 < output_size; cursor++) {
        unsigned char value = *cursor;
        output[used++] = (char)(isalnum(value) || value == '-' || value == '_' ?
                               value : '_');
    }
    if (!used && output_size > 1) {
        output[0] = 'g';
        output[1] = '\0';
        return;
    }
    output[used] = '\0';
}

static int json_string(FILE *stream, const char *value) {
    if (fputc('"', stream) == EOF) return 0;
    for (const unsigned char *cursor =
             (const unsigned char *)(value ? value : ""); *cursor; cursor++) {
        unsigned char character = *cursor;
        switch (character) {
            case '"': if (fputs("\\\"", stream) == EOF) return 0; break;
            case '\\': if (fputs("\\\\", stream) == EOF) return 0; break;
            case '\b': if (fputs("\\b", stream) == EOF) return 0; break;
            case '\f': if (fputs("\\f", stream) == EOF) return 0; break;
            case '\n': if (fputs("\\n", stream) == EOF) return 0; break;
            case '\r': if (fputs("\\r", stream) == EOF) return 0; break;
            case '\t': if (fputs("\\t", stream) == EOF) return 0; break;
            default:
                if (character < 0x20) {
                    if (fprintf(stream, "\\u%04x", (unsigned)character) < 0) return 0;
                } else if (fputc(character, stream) == EOF) return 0;
                break;
        }
    }
    return fputc('"', stream) != EOF;
}

static int write_json(FILE *stream, const h3cspeed_profile_report *report) {
    const h3cspeed_profile_metrics *m = &report->metrics;
    double accounted = m->file_read_seconds + m->pageable_copy_seconds +
        m->h2d_enqueue_seconds + m->compute_stream_wait_seconds +
        m->upload_stream_wait_seconds + m->event_wait_seconds +
        m->allocation_seconds;
    double coverage = report->wall_seconds > 0.0 ?
        accounted / report->wall_seconds : 0.0;
    if (coverage > 1.0) coverage = 1.0;
    if (fprintf(stream,
        "{\n  \"schema_version\": %u,\n  \"kind\": \"h3cspeed.cuda.profile\",\n"
        "  \"context\": {\"id\": %" PRIu64 ", \"pid\": %ld, \"label\": ",
        H3CSPEED_PROFILE_SCHEMA_VERSION, report->context_id,
        (long)H3CSPEED_GETPID()) < 0 || !json_string(stream, "redacted") ||
        fprintf(stream,
        ", \"device\": %d, \"sm\": \"%d%d\", \"complete\": %s},\n"
        "  \"wall\": {\"seconds\": %.9f, \"accounted_host_seconds\": %.9f, "
        "\"accounted_ratio\": %.9f, \"coverage_gate_valid\": false, "
        "\"coverage_gate_met\": false},\n"
        "  \"timing\": {\"file_read_seconds\": %.9f, "
        "\"pageable_copy_seconds\": %.9f, \"h2d_enqueue_seconds\": %.9f, "
        "\"compute_stream_wait_seconds\": %.9f, \"upload_stream_wait_seconds\": %.9f, "
        "\"event_wait_seconds\": %.9f, \"allocation_seconds\": %.9f, "
        "\"eviction_seconds\": %.9f, \"compute_device_seconds\": %.9f},\n"
        "  \"counts\": {\"begin\": %" PRIu64 ", \"continue\": %" PRIu64
        ", \"submit_sync\": %" PRIu64 ", \"last_use_fence\": %" PRIu64
        ", \"file_read\": %" PRIu64 ", \"pageable_copy\": %" PRIu64
        ", \"h2d_enqueue\": %" PRIu64 ", \"compute_stream_sync\": %" PRIu64
        ", \"upload_stream_sync\": %" PRIu64 ", \"event_sync\": %" PRIu64
        ", \"allocation\": %" PRIu64 ", \"capacity_lru\": %" PRIu64
        ", \"phase_retire\": %" PRIu64 ", \"error_cleanup\": %" PRIu64 "},\n"
        "  \"bytes\": {\"file_read\": %" PRIu64 ", \"pageable_copy\": %" PRIu64
        ", \"h2d\": %" PRIu64 ", \"capacity_lru\": %" PRIu64
        ", \"phase_retire\": %" PRIu64 ", \"error_cleanup\": %" PRIu64 "},\n"
        "  \"memory\": {\"device_peak_bytes\": %" PRIu64
        ", \"resident_weight_peak_bytes\": %" PRIu64
        ", \"host_cache_peak_bytes\": %" PRIu64 "},\n"
        "  \"offload\": {\"uploads\": %" PRIu64 ", \"upload_bytes\": %" PRIu64
        ", \"legacy_evictions\": %" PRIu64 ", \"legacy_evicted_bytes\": %" PRIu64
        ", \"file_fallback_reads\": %" PRIu64 ", \"file_fallback_bytes\": %" PRIu64 "},\n"
        "  \"dispatches\": {\"direct\": %" PRIu64 ", \"linear\": %" PRIu64
        ", \"convolution\": %" PRIu64 ", \"attention\": %" PRIu64 "},\n"
        "  \"validity\": {\"h2d_device_seconds\": false, "
        "\"compute_device_seconds\": %s, \"critical_path_scope\": "
        "\"gpu_context_host_operations\"},\n"
        "  \"non_additive\": [\"compute_device_seconds\", "
        "\"eviction_seconds_includes_event_wait\", "
        "\"device_and_host_timings_may_overlap\"]\n}\n",
        report->device, report->sm_major, report->sm_minor,
        report->complete ? "true" : "false", report->wall_seconds, accounted,
        coverage,
        m->file_read_seconds, m->pageable_copy_seconds, m->h2d_enqueue_seconds,
        m->compute_stream_wait_seconds, m->upload_stream_wait_seconds,
        m->event_wait_seconds, m->allocation_seconds, m->eviction_seconds,
        m->compute_device_seconds, m->begin_count, m->continue_count,
        m->submit_sync_count, m->last_use_fence_count, m->file_read_calls,
        m->pageable_copy_calls, m->h2d_enqueue_calls,
        m->compute_stream_syncs, m->upload_stream_syncs, m->event_syncs,
        m->allocation_calls, m->capacity_evictions, m->phase_retire_frees,
        m->error_cleanup_frees, m->file_read_bytes, m->pageable_copy_bytes,
        m->h2d_enqueue_bytes, m->capacity_evicted_bytes,
        m->phase_retired_bytes, m->error_cleanup_bytes,
        report->device_peak_bytes, report->resident_weight_peak_bytes,
        report->host_cache_peak_bytes, report->offload_uploads,
        report->offload_upload_bytes, report->offload_evictions,
        report->offload_evicted_bytes, report->file_fallback_reads,
        report->file_fallback_bytes, report->direct_dispatches,
        report->linear_dispatches, report->convolution_dispatches,
        report->attention_dispatches,
        m->compute_device_seconds > 0.0 ? "true" : "false") < 0) return 0;
    return !ferror(stream);
}

int h3cspeed_profile_write_json_directory(
    const char *directory, const h3cspeed_profile_report *report,
    char *written_path, size_t written_path_size,
    char *error, size_t error_size) {
    if (written_path && written_path_size) written_path[0] = '\0';
    if (error && error_size) error[0] = '\0';
    if (!directory || !*directory || !h3cspeed_profile_report_valid(report)) {
        set_error(error, error_size, "invalid profile report request");
        return 0;
    }
    struct stat information;
    if (stat(directory, &information) != 0 ||
#if defined(_WIN32)
        (information.st_mode & _S_IFDIR) == 0) {
#else
        !S_ISDIR(information.st_mode)) {
#endif
        set_error(error, error_size, "profile output directory does not exist");
        return 0;
    }
    char label[80];
    /* Labels are useful for known internal phase names, but callers may pass
     * sensitive text. Keep arbitrary values out of both filenames and JSON. */
    const char *safe_label = h3cspeed_profile_safe_label(report->label);
    sanitize_label(safe_label, label, sizeof(label));
    char final_path[1024];
    char temporary_path[1056];
    uint64_t nonce = (uint64_t)(h3cspeed_profile_now_seconds() * 1000000000.0);
    int final_length = snprintf(final_path, sizeof(final_path),
        "%s%sh3-profile-%ld-%" PRIu64 "-%016" PRIx64 "-%s.json", directory,
        directory[strlen(directory) - 1] == '/' ||
        directory[strlen(directory) - 1] == '\\' ? "" : "/",
        (long)H3CSPEED_GETPID(), report->context_id, nonce, label);
    int temporary_length = final_length >= 0 ?
        snprintf(temporary_path, sizeof(temporary_path), "%s.tmp", final_path) : -1;
    if (final_length < 0 || (size_t)final_length >= sizeof(final_path) ||
        temporary_length < 0 ||
        (size_t)temporary_length >= sizeof(temporary_path) ||
        (written_path && written_path_size &&
         (size_t)final_length >= written_path_size)) {
        set_error(error, error_size, "profile output path is too long");
        return 0;
    }
    FILE *stream = fopen(temporary_path, "wbx");
    if (!stream) {
        set_error(error, error_size, "cannot create profile temporary file");
        return 0;
    }
    int ok = write_json(stream, report) && fflush(stream) == 0 &&
             H3CSPEED_FSYNC(H3CSPEED_FILENO(stream)) == 0;
    if (fclose(stream) != 0) ok = 0;
#if defined(_WIN32)
    if (ok && rename(temporary_path, final_path) != 0) ok = 0;
#else
    /* Same-directory link publishes atomically and fails if final exists. */
    if (ok && link(temporary_path, final_path) != 0) ok = 0;
    if (ok && unlink(temporary_path) != 0) {
        int saved_errno = errno;
        (void)unlink(final_path);
        errno = saved_errno;
        ok = 0;
    }
#endif
    if (!ok) {
        int saved_errno = errno;
        (void)remove(temporary_path);
        if (error && error_size)
            snprintf(error, error_size, "cannot finalize profile report: %s",
                     strerror(saved_errno));
        return 0;
    }
    if (written_path && written_path_size)
        snprintf(written_path, written_path_size, "%s", final_path);
    return 1;
}
