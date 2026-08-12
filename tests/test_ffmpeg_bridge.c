#include "h3_ffmpeg.h"
#include <pthread.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(const char *stage, const char *detail, const char *path) {
    fprintf(stderr, "ffmpeg bridge failed at %s: %s\n", stage,
            detail && *detail ? detail : "unknown error");
    if (path) remove(path);
    return 1;
}

typedef struct {
    const char *path;
    const unsigned char *video;
    const float *audio;
    int ok;
    char error[512];
} encode_job;

static void *encode_concurrently(void *opaque) {
    encode_job *job = (encode_job *)opaque;
    job->ok = h3_ffmpeg_write_av_rgb24_f32(
        job->path, job->video, 48, 16, 16, 24,
        job->audio, 64000, 2, 32000, job->error, sizeof(job->error));
    return NULL;
}

int main(void) {
    enum { WIDTH = 16, HEIGHT = 16, FRAMES = 48, FPS = 24,
           SAMPLE_RATE = 32000, SAMPLES = 64000, CHANNELS = 2 };
    const char *path = "h3cspeed-ffmpeg-bridge.mp4";
    unsigned char video[WIDTH * HEIGHT * 3 * FRAMES];
    float audio[CHANNELS * SAMPLES];
    char error[512] = {0};

    for (int frame = 0; frame < FRAMES; frame++) {
        for (int pixel = 0; pixel < WIDTH * HEIGHT; pixel++) {
            size_t offset = ((size_t)frame * WIDTH * HEIGHT + (size_t)pixel) * 3;
            video[offset] = (unsigned char)(frame * 40);
            video[offset + 1] = (unsigned char)(pixel % WIDTH * 12);
            video[offset + 2] = (unsigned char)(pixel / WIDTH * 12);
        }
    }
    memset(audio, 0, sizeof(audio));
    remove(path);

    int fillers[48];
    for (int index = 0; index < 48; index++) {
        fillers[index] = _open("NUL", _O_RDONLY | _O_NOINHERIT);
        if (fillers[index] < 0)
            return fail("high-fd setup", "cannot reserve CRT descriptors", path);
    }

    if (!h3_ffmpeg_write_av_rgb24_f32(path, video, FRAMES, WIDTH, HEIGHT,
                                      FPS, audio, SAMPLES, CHANNELS,
                                      SAMPLE_RATE, error, sizeof(error)))
        return fail("encode", error, path);

    int width = 0, height = 0;
    if (!h3_ffprobe_visual_size(path, &width, &height, error, sizeof(error)))
        return fail("probe", error, path);
    if (width != WIDTH || height != HEIGHT)
        return fail("dimensions", "unexpected encoded dimensions", path);

    float *image = NULL;
    if (!h3_ffmpeg_read_image_f32(path, WIDTH, HEIGHT,
                                  H3_IMAGE_FIT_STRETCH, &image,
                                  error, sizeof(error)))
        return fail("image decode", error, path);
    free(image);

    float *pcm = NULL;
    int samples = 0;
    if (!h3_ffmpeg_read_audio_f32(path, 64000, 1, &pcm, &samples,
                                  error, sizeof(error)))
        return fail("audio decode", error, path);
    free(pcm);
    if (samples < 1) return fail("audio samples", "empty decoded audio", path);

    encode_job jobs[2] = {
        {"h3cspeed-ffmpeg-concurrent-1.mp4", video, audio, 0, {0}},
        {"h3cspeed-ffmpeg-concurrent-2.mp4", video, audio, 0, {0}},
    };
    pthread_t threads[2];
    remove(jobs[0].path);
    remove(jobs[1].path);
    if (pthread_create(&threads[0], NULL, encode_concurrently, &jobs[0]) != 0 ||
        pthread_create(&threads[1], NULL, encode_concurrently, &jobs[1]) != 0)
        return fail("concurrent start", "cannot create encoder threads", path);
    if (pthread_join(threads[0], NULL) != 0 ||
        pthread_join(threads[1], NULL) != 0)
        return fail("concurrent join", "cannot join encoder threads", path);
    for (int index = 0; index < 2; index++) {
        if (!jobs[index].ok)
            return fail("concurrent encode", jobs[index].error, jobs[index].path);
        if (remove(jobs[index].path) != 0)
            return fail("concurrent cleanup", "cannot remove output", NULL);
    }
    for (int index = 0; index < 48; index++) _close(fillers[index]);

    if (remove(path) != 0) return fail("cleanup", "cannot remove output", NULL);
    puts("ffmpeg bridge: passed");
    return 0;
}
