/* Same-latent parity gate for the opt-in layer-major video VAE traversal
 * (plan P1A / PERF-004). Decodes one deterministic latent twice in-process --
 * tile-major first, then H3_VAE_LAYER_MAJOR=1 -- and requires bit-for-bit
 * identical RGB frames. Final-video inspection alone is not a numerical test,
 * so this harness compares raw decoded pixels instead. */

#include "h3_video_vae.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", \
                __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static int set_layer_major(const char *value) {
#if defined(_WIN32)
    return _putenv_s("H3_VAE_LAYER_MAJOR", value ? value : "") == 0;
#else
    return value ? setenv("H3_VAE_LAYER_MAJOR", value, 1) == 0 :
                   unsetenv("H3_VAE_LAYER_MAJOR") == 0;
#endif
}

static int apply_default_offload_environment(void) {
    static const char *defaults[][2] = {
        { "H3_CUDA_LOW_VRAM", "1" },
        { "H3_CUDA_OFFLOAD", "ram+file" },
        { "H3_CUDA_VRAM_BUDGET_MIB", "5888" },
        { "H3_CUDA_WEIGHT_CACHE_MIB", "1536" },
        { "H3_CUDA_PINNED_HOST_MIB", "128" },
        { "H3_CUDA_STAGING_MIB", "64" },
    };
    for (size_t index = 0; index < sizeof(defaults) / sizeof(defaults[0]);
         index++)
        if (!getenv(defaults[index][0])) {
#if defined(_WIN32)
            if (_putenv_s(defaults[index][0], defaults[index][1]) != 0)
                return 0;
#else
            if (setenv(defaults[index][0], defaults[index][1], 1) != 0)
                return 0;
#endif
        }
    return 1;
}

static int is_no_cuda_device_error(const char *error) {
    char normalized[512];
    size_t length = 0;
    if (!error) return 0;
    while (error[length] && length + 1 < sizeof(normalized)) {
        normalized[length] = (char)tolower((unsigned char)error[length]);
        length++;
    }
    normalized[length] = '\0';
    return strstr(normalized, "no cuda-capable device") != NULL ||
           strstr(normalized, "cuda_error_no_device") != NULL ||
           strstr(normalized, "cudaerrornodevice") != NULL ||
           strstr(normalized, "cuda driver version is insufficient") != NULL ||
           strstr(normalized, "cuda_error_insufficient_driver") != NULL ||
           strstr(normalized, "cudaerrorinsufficientdriver") != NULL;
}

/* Deterministic xorshift64 latent so both traversals always see identical
 * input without shipping a large fixture. */
static void fill_latent(float *latent, size_t count) {
    uint64_t state = 42;
    for (size_t index = 0; index < count; index++) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        latent[index] =
            (float)((double)(state >> 11) / (double)(UINT64_C(1) << 53) *
                    4.0 - 2.0);
    }
}

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == 1 || completed == total || completed % 12 == 0)
        fprintf(stderr, "video VAE weights: %d/%d blocks\n",
                completed, total);
}

static void report_stats(const char *label, const h3_video_frames *frames) {
    printf("%s: %.3f GiB allocated, %.3f GPU seconds, %llu linear, "
           "%llu sdpa, %llu submissions\n", label,
           (double)frames->gpu_stats.allocated_bytes /
               (1024.0 * 1024.0 * 1024.0),
           frames->gpu_stats.gpu_seconds,
           (unsigned long long)frames->gpu_stats.mps_linear_dispatches,
           (unsigned long long)frames->gpu_stats.mps_sdpa_dispatches,
           (unsigned long long)frames->gpu_stats.submissions);
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : getenv("H3CSPEED_MODEL_ROOT");
    int latent_time = argc > 2 ? atoi(argv[2]) : 7;
    int latent_height = argc > 3 ? atoi(argv[3]) : 16;
    int latent_width = argc > 4 ? atoi(argv[4]) : 32;
    if (!model_root || !*model_root) {
        fprintf(stderr,
                "layer-major VAE parity skipped: pass the quantized model "
                "root (containing FL2VA/video_vae/source) as argv[1] or set "
                "H3CSPEED_MODEL_ROOT\n");
        return 77;
    }
    char weights[1024];
    int written = snprintf(weights, sizeof(weights),
                           "%s/FL2VA/video_vae/source", model_root);
    if (written <= 0 || written >= (int)sizeof(weights)) {
        fprintf(stderr, "layer-major VAE parity skipped: model root path is "
                "too long\n");
        return 77;
    }
    char probe_path[1200];
    snprintf(probe_path, sizeof(probe_path), "%s/config.json", weights);
    FILE *probe = fopen(probe_path, "rb");
    if (!probe) {
        fprintf(stderr,
                "layer-major VAE parity skipped: %s is not a readable "
                "FL2VA video VAE source directory\n", weights);
        return 77;
    }
    fclose(probe);
    CHECK(apply_default_offload_environment());
    int chunks = (latent_time - 2) / 5;
    int expected_frames = chunks * 17 + 5;
    int pixel_h = latent_height * 16;
    int pixel_w = latent_width * 16;
    size_t latent_count = (size_t)24 * (size_t)latent_time *
                          (size_t)latent_height * (size_t)latent_width;
    float *latent = malloc(latent_count * sizeof(*latent));
    CHECK(latent);
    fill_latent(latent, latent_count);
    char error[512];

    CHECK(set_layer_major(NULL));
    h3_video_frames tile_major;
    memset(&tile_major, 0, sizeof(tile_major));
    if (!h3_video_vae_decode(weights, "h3_shaders.metal", latent,
                             latent_time, latent_height, latent_width,
                             progress, NULL, &tile_major,
                             error, sizeof(error))) {
        if (is_no_cuda_device_error(error)) {
            fprintf(stderr, "layer-major VAE parity skipped: %s\n", error);
            free(latent);
            return 77;
        }
        fprintf(stderr, "tile-major video VAE decode failed: %s\n", error);
        free(latent);
        return 1;
    }
    CHECK(tile_major.frames == expected_frames &&
          tile_major.height == pixel_h && tile_major.width == pixel_w);
    report_stats("tile-major", &tile_major);

    CHECK(set_layer_major("1"));
    h3_video_frames layer_major;
    memset(&layer_major, 0, sizeof(layer_major));
    if (!h3_video_vae_decode(weights, "h3_shaders.metal", latent,
                             latent_time, latent_height, latent_width,
                             progress, NULL, &layer_major,
                             error, sizeof(error))) {
        if (is_no_cuda_device_error(error)) {
            fprintf(stderr, "layer-major VAE parity skipped: %s\n", error);
            h3_video_frames_free(&tile_major);
            free(latent);
            return 77;
        }
        fprintf(stderr, "layer-major video VAE decode failed: %s\n", error);
        h3_video_frames_free(&tile_major);
        free(latent);
        return 1;
    }
    CHECK(set_layer_major(NULL));
    CHECK(layer_major.frames == expected_frames &&
          layer_major.height == pixel_h && layer_major.width == pixel_w);
    report_stats("layer-major", &layer_major);

    size_t total = (size_t)tile_major.frames * (size_t)tile_major.height *
                   (size_t)tile_major.width * 3;
    CHECK(layer_major.frames == tile_major.frames);
    int finite = 1;
    for (size_t index = 0; index < total; index++)
        if (!isfinite(tile_major.rgb[index]) ||
            !isfinite(layer_major.rgb[index])) {
            finite = 0;
            break;
        }
    if (!finite) {
        fprintf(stderr, "video VAE decode produced non-finite pixels\n");
        h3_video_frames_free(&tile_major);
        h3_video_frames_free(&layer_major);
        free(latent);
        return 1;
    }
    double maximum = 0.0, square_error = 0.0, square_value = 0.0;
    size_t mismatches = 0;
    double frame_maximum[256];
    memset(frame_maximum, 0, sizeof(frame_maximum));
    size_t frame_elements = (size_t)tile_major.height *
                            (size_t)tile_major.width * 3;
    for (size_t index = 0; index < total; index++) {
        double delta = fabs((double)tile_major.rgb[index] -
                            (double)layer_major.rgb[index]);
        if (tile_major.rgb[index] != layer_major.rgb[index]) mismatches++;
        if (delta > maximum) maximum = delta;
        size_t frame = index / frame_elements;
        if (frame < 256 && delta > frame_maximum[frame])
            frame_maximum[frame] = delta;
        square_error += delta * delta;
        square_value += (double)tile_major.rgb[index] *
                        (double)tile_major.rgb[index];
    }
    double relative_l2 = sqrt(square_error /
                              (square_value > 1e-24 ? square_value : 1e-24));
    for (int frame = 0; frame < tile_major.frames && frame < 256; frame++)
        printf("frame %02d: max-abs-diff %.9g\n", frame,
               frame_maximum[frame]);
    printf("layer-major parity %dx%dx%d latent -> %d frames %dx%d: "
           "bit-exact %s, mismatches %zu/%zu, max-abs %.9g, rel-L2 %.9g\n",
           latent_time, latent_height, latent_width, tile_major.frames,
           tile_major.height, tile_major.width,
           mismatches == 0 ? "yes" : "no", mismatches, total,
           maximum, relative_l2);
    int ok = mismatches == 0;
    if (!ok)
        fprintf(stderr,
                "layer-major video VAE decode is not bit-for-bit identical "
                "to tile-major decoding\n");
    else
        puts("ok: layer-major video VAE matches tile-major bit-for-bit");
    h3_video_frames_free(&tile_major);
    h3_video_frames_free(&layer_major);
    free(latent);
    return ok ? 0 : 1;
}
