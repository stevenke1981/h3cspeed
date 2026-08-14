#include "h3_perf002_trace.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <direct.h>
#define remove_file _unlink
static int set_environment(const char *name, const char *value) {
    return _putenv_s(name, value ? value : "");
}
#else
#include <unistd.h>
static int set_environment(const char *name, const char *value) {
    return value ? setenv(name, value, 1) : unsetenv(name);
}
#define remove_file unlink
#endif

static int fail(const char *message) {
    fprintf(stderr, "PERF-002 trace test failed: %s\n", message);
    return 1;
}

static int read_contains(const char *path, const char *needle) {
    FILE *stream = fopen(path, "rb");
    if (!stream) return 0;
    char buffer[32768];
    size_t count = fread(buffer, 1, sizeof(buffer) - 1, stream);
    fclose(stream);
    buffer[count] = '\0';
    return strstr(buffer, needle) != NULL;
}

static int join_path(char *output, size_t capacity, const char *directory,
                     const char *name) {
    int count = snprintf(output, capacity, "%s%c%s", directory,
#if defined(_WIN32)
                         '\\',
#else
                         '/',
#endif
                         name);
    return count >= 0 && (size_t)count < capacity;
}

int main(int argc, char **argv) {
    if (argc != 2) return fail("expected an absolute output directory");
    char scheduler[4096], attention[4096];
    if (!join_path(scheduler, sizeof(scheduler), argv[1],
                   "perf002-trace-scheduler.json") ||
        !join_path(attention, sizeof(attention), argv[1],
                   "perf002-trace-attention.json"))
        return fail("output path overflow");
    (void)remove_file(scheduler);
    (void)remove_file(attention);

    float sigma_video[] = {1.0f, 12.0f / 13.0f, 0.0f};
    float sigma_audio[] = {1.0f, 3.0f / 4.0f, 0.0f};

    if (set_environment("H3CSPEED_PERF002_SCHEDULER_TRACE", scheduler) != 0 ||
        set_environment("H3CSPEED_PERF002_ATTENTION_TRACE", NULL) != 0 ||
        h3cspeed_perf002_trace_begin(864, 480, 22, 50, 2, 42,
                                      sigma_video, sigma_audio) != -1)
        return fail("one-sided trace environment was accepted");
    if (set_environment("H3CSPEED_PERF002_ATTENTION_TRACE", "relative.json") != 0 ||
        h3cspeed_perf002_trace_begin(864, 480, 22, 50, 2, 42,
                                      sigma_video, sigma_audio) != -1)
        return fail("relative trace path was accepted");
    if (set_environment("H3CSPEED_PERF002_ATTENTION_TRACE", attention) != 0 ||
        set_environment("H3_CUDA_ATTENTION", "invalid") != 0 ||
        h3cspeed_perf002_trace_begin(864, 480, 22, 50, 2, 42,
                                      sigma_video, sigma_audio) != -1 ||
        set_environment("H3_CUDA_ATTENTION", "sage") != 0 ||
        h3cspeed_perf002_trace_begin(864, 480, 22, 50, 2, 42,
                                      sigma_video, sigma_audio) != 1)
        return fail("valid trace environment was rejected");
    if (h3cspeed_perf002_trace_begin(864, 480, 22, 50, 2, 42,
                                     sigma_video, sigma_audio) != -1)
        return fail("active trace was hijacked");
    h3cspeed_perf002_trace_note_bf16_attention(1, 0, 0);
    h3cspeed_perf002_trace_note_bf16_attention(0, 1, 0);
    h3cspeed_perf002_trace_note_audio_euler_step();
    h3cspeed_perf002_trace_note_audio_euler_step();
    if (!h3cspeed_perf002_trace_finish(1))
        return fail("trace publication failed");
    if (!read_contains(scheduler, "\"sigma_video\":[1,") ||
        !read_contains(scheduler, ",0],\"sigma_audio\":[1,") ||
        !read_contains(scheduler, "\"sigma_audio\":[1,0.75,0]") ||
        !read_contains(scheduler, "\"raw_audio_protocol_verified\":true") ||
        !read_contains(attention, "\"backend_hits\":1") ||
        !read_contains(attention, "\"expected_native_calls\":1") ||
        !read_contains(attention, "\"unexpected_fallbacks\":0"))
        return fail("trace schema or counters were incorrect");

    if (h3cspeed_perf002_trace_begin(864, 480, 22, 50, 2, 42,
                                      sigma_video, sigma_audio) != -1)
        return fail("existing trace target was clobbered");
    remove_file(scheduler);
    remove_file(attention);
    set_environment("H3CSPEED_PERF002_SCHEDULER_TRACE", NULL);
    set_environment("H3CSPEED_PERF002_ATTENTION_TRACE", NULL);
    set_environment("H3_CUDA_ATTENTION", NULL);
    puts("PERF-002 trace producer PASS");
    return 0;
}
