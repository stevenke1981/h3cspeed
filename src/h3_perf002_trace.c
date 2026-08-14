#include "h3_perf002_trace.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#if defined(_WIN32)
#include "h3_msvc_compat.h"
#include <process.h>
#ifndef O_BINARY
#define O_BINARY 0
#endif
#define h3cspeed_close _close
#define h3cspeed_unlink _unlink
#define h3cspeed_sync _commit
#define h3cspeed_open(path, flags, mode) _open((path), (flags) | _O_BINARY, (mode))
#define fdopen _fdopen
#define getpid _getpid
#else
#include <unistd.h>
#include <pthread.h>
#define h3cspeed_close close
#define h3cspeed_unlink unlink
#define h3cspeed_sync fsync
#define h3cspeed_open(path, flags, mode) open((path), (flags), (mode))
#ifndef O_BINARY
#define O_BINARY 0
#endif
#endif

#define H3CSPEED_PERF002_SCHEDULER_ENV "H3CSPEED_PERF002_SCHEDULER_TRACE"
#define H3CSPEED_PERF002_ATTENTION_ENV "H3CSPEED_PERF002_ATTENTION_TRACE"
#define H3CSPEED_PERF002_BACKEND_ENV "H3_CUDA_ATTENTION"
#define H3CSPEED_PERF002_MAX_PATH 4096
#define H3CSPEED_PERF002_MAX_STEPS 1000

typedef struct {
    int enabled;
    char *scheduler_path;
    char *attention_path;
    int width;
    int height;
    int frames;
    int layers;
    int steps;
    uint64_t seed;
    float sigma_video[H3CSPEED_PERF002_MAX_STEPS + 1];
    float sigma_audio[H3CSPEED_PERF002_MAX_STEPS + 1];
    int sage_requested;
    unsigned long long sage_hits;
    unsigned long long expected_native_calls;
    unsigned long long unexpected_fallbacks;
    unsigned long long audio_euler_steps;
} h3cspeed_perf002_trace_state;

static h3cspeed_perf002_trace_state g_trace;
static unsigned long g_temp_counter;

#if defined(_WIN32)
static SRWLOCK g_trace_lock = SRWLOCK_INIT;
static volatile LONG g_trace_enabled;
static int trace_enabled_load(void) {
    return InterlockedCompareExchange(&g_trace_enabled, 0, 0) != 0;
}
static void trace_enabled_store(int enabled) {
    (void)InterlockedExchange(&g_trace_enabled, enabled ? 1 : 0);
}
static void h3cspeed_lock(void) {
    AcquireSRWLockExclusive(&g_trace_lock);
}
static void h3cspeed_unlock(void) {
    ReleaseSRWLockExclusive(&g_trace_lock);
}
#else
#include <stdatomic.h>
static pthread_mutex_t g_trace_lock = PTHREAD_MUTEX_INITIALIZER;
static _Atomic int g_trace_enabled;
static int trace_enabled_load(void) {
    return atomic_load_explicit(&g_trace_enabled, memory_order_acquire);
}
static void trace_enabled_store(int enabled) {
    atomic_store_explicit(&g_trace_enabled, enabled, memory_order_release);
}
static void h3cspeed_lock(void) {
    (void)pthread_mutex_lock(&g_trace_lock);
}
static void h3cspeed_unlock(void) {
    (void)pthread_mutex_unlock(&g_trace_lock);
}
#endif

static int path_is_absolute(const char *path) {
    if (!path || !*path) return 0;
#if defined(_WIN32)
    return ((path[0] >= 'A' && path[0] <= 'Z') ||
            (path[0] >= 'a' && path[0] <= 'z')) && path[1] == ':' &&
           (path[2] == '\\' || path[2] == '/') ? 1 :
           (path[0] == '\\' && path[1] == '\\');
#else
    return path[0] == '/';
#endif
}

static int path_is_clean(const char *path) {
    if (!path || strlen(path) >= H3CSPEED_PERF002_MAX_PATH ||
        !path_is_absolute(path)) return 0;
    for (const unsigned char *cursor = (const unsigned char *)path;
         *cursor; cursor++) {
        if (*cursor < 0x20 || *cursor == '\n' || *cursor == '\r') return 0;
    }
    return 1;
}

static int parent_directory_exists(const char *path) {
    char parent[H3CSPEED_PERF002_MAX_PATH];
    size_t length = strlen(path);
    if (length >= sizeof(parent)) return 0;
    memcpy(parent, path, length + 1);
    char *separator = strrchr(parent, '/');
#if defined(_WIN32)
    char *backslash = strrchr(parent, '\\');
    if (!separator || (backslash && backslash > separator)) separator = backslash;
#endif
    if (!separator) return 0;
    if (separator == parent) separator[1] = '\0';
    else *separator = '\0';
    struct stat status;
    if (stat(parent, &status) != 0 || !S_ISDIR(status.st_mode)) return 0;
    return 1;
}

static int trace_path_available(const char *path) {
    struct stat status;
    if (stat(path, &status) == 0) return 0;
    return errno == ENOENT && parent_directory_exists(path);
}

static int sigma_schedule_valid(int steps, const float *video,
                                const float *audio) {
    if (steps < 2 || steps > H3CSPEED_PERF002_MAX_STEPS || !video || !audio)
        return 0;
    for (int index = 0; index <= steps; index++) {
        float video_value = video[index];
        float audio_value = audio[index];
        if (!isfinite(video_value) || !isfinite(audio_value) ||
            video_value < 0.0f || video_value > 1.0f ||
            audio_value < 0.0f || audio_value > 1.0f) return 0;
        if (index && (video_value > video[index - 1] ||
                      audio_value > audio[index - 1])) return 0;
    }
    if (video[0] != 1.0f || audio[0] != 1.0f ||
        video[steps] != 0.0f || audio[steps] != 0.0f) return 0;
    return 1;
}

static void reset_state_locked(void) {
    free(g_trace.scheduler_path);
    free(g_trace.attention_path);
    g_trace.scheduler_path = NULL;
    g_trace.attention_path = NULL;
    g_trace.enabled = 0;
    trace_enabled_store(0);
    g_trace.sage_requested = 0;
    g_trace.sage_hits = 0;
    g_trace.expected_native_calls = 0;
    g_trace.unexpected_fallbacks = 0;
    g_trace.audio_euler_steps = 0;
    memset(g_trace.sigma_video, 0, sizeof(g_trace.sigma_video));
    memset(g_trace.sigma_audio, 0, sizeof(g_trace.sigma_audio));
}

int h3cspeed_perf002_trace_begin(int width, int height, int frames,
                                 int layers, int steps, uint64_t seed,
                                 const float *sigma_video,
                                 const float *sigma_audio) {
    const char *scheduler = getenv(H3CSPEED_PERF002_SCHEDULER_ENV);
    const char *attention = getenv(H3CSPEED_PERF002_ATTENTION_ENV);
    if (!scheduler && !attention) return 0;
    if (!scheduler || !attention || !path_is_clean(scheduler) ||
        !path_is_clean(attention) || !parent_directory_exists(scheduler) ||
        !parent_directory_exists(attention) || !trace_path_available(scheduler) ||
        !trace_path_available(attention) ||
        !sigma_schedule_valid(steps, sigma_video, sigma_audio) ||
        width < 1 || height < 1 || frames < 1 || layers < 1 || steps < 2)
        return -1;

    char *scheduler_copy = strdup(scheduler);
    char *attention_copy = strdup(attention);
    if (!scheduler_copy || !attention_copy) {
        free(scheduler_copy);
        free(attention_copy);
        return -1;
    }
    const char *backend = getenv(H3CSPEED_PERF002_BACKEND_ENV);
    if (backend && *backend && strcmp(backend, "sage") != 0 &&
        strcmp(backend, "native") != 0) {
        free(scheduler_copy);
        free(attention_copy);
        return -1;
    }
    h3cspeed_lock();
    if (g_trace.enabled) {
        h3cspeed_unlock();
        free(scheduler_copy);
        free(attention_copy);
        return -1;
    }
    g_trace.scheduler_path = scheduler_copy;
    g_trace.attention_path = attention_copy;
    g_trace.width = width;
    g_trace.height = height;
    g_trace.frames = frames;
    g_trace.layers = layers;
    g_trace.steps = steps;
    g_trace.seed = seed;
    memcpy(g_trace.sigma_video, sigma_video,
           ((size_t)steps + 1) * sizeof(*sigma_video));
    memcpy(g_trace.sigma_audio, sigma_audio,
           ((size_t)steps + 1) * sizeof(*sigma_audio));
    g_trace.sage_requested = backend && strcmp(backend, "sage") == 0;
    g_trace.enabled = 1;
    trace_enabled_store(1);
    h3cspeed_unlock();
    return 1;
}

void h3cspeed_perf002_trace_note_bf16_attention(int sage_hit,
                                                 int expected_native,
                                                 int unexpected_fallback) {
    if (!trace_enabled_load()) return;
    h3cspeed_lock();
    if (g_trace.enabled) {
        if (sage_hit) g_trace.sage_hits++;
        if (expected_native) g_trace.expected_native_calls++;
        if (unexpected_fallback) g_trace.unexpected_fallbacks++;
    }
    h3cspeed_unlock();
}

void h3cspeed_perf002_trace_note_audio_euler_step(void) {
    if (!trace_enabled_load()) return;
    h3cspeed_lock();
    if (g_trace.enabled) g_trace.audio_euler_steps++;
    h3cspeed_unlock();
}

static int write_json_file(const char *path, const char *json) {
    char temporary[H3CSPEED_PERF002_MAX_PATH + 64];
    int written = snprintf(temporary, sizeof(temporary), "%s.tmp.%lu.%lu",
                           path, (unsigned long)getpid(), ++g_temp_counter);
    if (written < 0 || (size_t)written >= sizeof(temporary)) return 0;
    int descriptor = h3cspeed_open(temporary,
                                   O_WRONLY | O_CREAT | O_EXCL | O_BINARY,
                                   0600);
    if (descriptor < 0) return 0;
    FILE *stream = fdopen(descriptor, "wb");
    if (!stream) {
        h3cspeed_close(descriptor);
        h3cspeed_unlink(temporary);
        return 0;
    }
    size_t length = strlen(json);
    int ok = fwrite(json, 1, length, stream) == length &&
             fflush(stream) == 0 && h3cspeed_sync(fileno(stream)) == 0;
    if (fclose(stream) != 0) ok = 0;
    if (!ok) {
        h3cspeed_unlink(temporary);
        return 0;
    }
    int published = 0;
#if defined(_WIN32)
    published = CreateHardLinkA(path, temporary, NULL) != 0;
    if (published) (void)h3cspeed_unlink(temporary);
#else
    published = link(temporary, path) == 0;
    if (published) (void)h3cspeed_unlink(temporary);
#endif
    if (!published) {
        h3cspeed_unlink(temporary);
        return 0;
    }
    return 1;
}

static int format_scheduler_json(char *output, size_t capacity) {
    int used = snprintf(output, capacity,
        "{\"schema_version\":1,\"engine\":\"h3cspeed\","
        "\"sampler\":\"dual_clock_euler\",\"schedule\":\"native_flow\","
        "\"video_shift\":12.0,\"audio_shift\":3.0,\"width\":%d,"
        "\"height\":%d,\"frames\":%d,\"steps\":%d,\"layers\":%d,"
        "\"seed\":%llu,\"sigma_video\":[",
        g_trace.width, g_trace.height, g_trace.frames, g_trace.steps,
        g_trace.layers, (unsigned long long)g_trace.seed);
    if (used < 0 || (size_t)used >= capacity) return 0;
    for (int index = 0; index <= g_trace.steps; index++) {
        int count = snprintf(output + used, capacity - (size_t)used,
                             "%s%.9g", index ? "," : "",
                             (double)g_trace.sigma_video[index]);
        if (count < 0 || (size_t)count >= capacity - (size_t)used) return 0;
        used += count;
    }
    int count = snprintf(output + used, capacity - (size_t)used,
                         "],\"sigma_audio\":[");
    if (count < 0 || (size_t)count >= capacity - (size_t)used) return 0;
    used += count;
    for (int index = 0; index <= g_trace.steps; index++) {
        count = snprintf(output + used, capacity - (size_t)used,
                         "%s%.9g", index ? "," : "",
                         (double)g_trace.sigma_audio[index]);
        if (count < 0 || (size_t)count >= capacity - (size_t)used) return 0;
        used += count;
    }
    count = snprintf(output + used, capacity - (size_t)used,
                     "],\"raw_audio_protocol_verified\":true}\n");
    return count >= 0 && (size_t)count < capacity - (size_t)used;
}

static int format_attention_json(char *output, size_t capacity) {
    int used = snprintf(output, capacity,
        "{\"schema_version\":1,\"engine\":\"h3cspeed\","
        "\"requested\":\"%s\",\"selected\":\"%s\","
        "\"scope\":\"dit_bf16\",\"backend_hits\":%llu,"
        "\"expected_native_calls\":%llu,\"unexpected_fallbacks\":%llu}\n",
        g_trace.sage_requested ? "sage" : "native",
        g_trace.sage_requested ? "sage" : "native",
        g_trace.sage_hits, g_trace.expected_native_calls,
        g_trace.unexpected_fallbacks);
    return used >= 0 && (size_t)used < capacity;
}

int h3cspeed_perf002_trace_finish(int raw_audio_protocol_verified) {
    char scheduler_json[16384];
    char attention_json[2048];
    h3cspeed_lock();
    if (!g_trace.enabled) {
        h3cspeed_unlock();
        return 1;
    }
    if (!raw_audio_protocol_verified ||
        g_trace.audio_euler_steps != (unsigned long long)g_trace.steps ||
        !format_scheduler_json(
            scheduler_json, sizeof(scheduler_json)) ||
        !format_attention_json(attention_json, sizeof(attention_json))) {
        reset_state_locked();
        h3cspeed_unlock();
        return 0;
    }
    char *scheduler_path = strdup(g_trace.scheduler_path);
    char *attention_path = strdup(g_trace.attention_path);
    int scheduler_ok = scheduler_path && attention_path &&
        write_json_file(scheduler_path, scheduler_json);
    int attention_ok = scheduler_ok && write_json_file(attention_path,
                                                        attention_json);
    if (!attention_ok && scheduler_ok) h3cspeed_unlink(scheduler_path);
    free(scheduler_path);
    free(attention_path);
    reset_state_locked();
    h3cspeed_unlock();
    return scheduler_ok && attention_ok;
}

void h3cspeed_perf002_trace_abort(void) {
    h3cspeed_lock();
    reset_state_locked();
    h3cspeed_unlock();
}
