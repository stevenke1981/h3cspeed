#include "h3_resize_portable.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    int first;
    int count;
    float weights[16];
} h3_filter;

static double sinc(double value) {
    if (fabs(value) < 1e-12) return 1.0;
    double x = M_PI * value;
    return sin(x) / x;
}

static double lanczos(double value, double radius) {
    value = fabs(value);
    if (value >= radius) return 0.0;
    return sinc(value) * sinc(value / radius);
}

static int clamp_index(int value, int limit) {
    if (value < 0) return 0;
    if (value >= limit) return limit - 1;
    return value;
}

static h3_filter *build_filters(int source, int destination) {
    if (source < 1 || destination < 1) return NULL;
    h3_filter *filters = calloc((size_t)destination, sizeof(*filters));
    if (!filters) return NULL;

    const double scale = (double)destination / (double)source;
    const double support_scale = scale < 1.0 ? scale : 1.0;
    const double radius = 3.0 / support_scale;
    const int taps = (int)ceil(radius * 2.0) + 2;
    if (taps > 16) {
        free(filters);
        return NULL;
    }

    for (int output = 0; output < destination; output++) {
        double center = ((double)output + 0.5) / scale - 0.5;
        int first = (int)floor(center - radius + 1.0);
        int last = (int)floor(center + radius);
        double sum = 0.0;
        int count = 0;
        for (int source_index = first; source_index <= last; source_index++) {
            double distance = (center - (double)source_index) * support_scale;
            double weight = lanczos(distance, 3.0);
            if (weight == 0.0) continue;
            filters[output].weights[count++] = (float)weight;
            sum += weight;
        }
        if (count == 0 || fabs(sum) < 1e-15) {
            filters[output].first = (int)llround(center);
            filters[output].count = 1;
            filters[output].weights[0] = 1.0f;
            continue;
        }
        filters[output].first = first;
        filters[output].count = count;
        for (int tap = 0; tap < count; tap++) {
            filters[output].weights[tap] = (float)(filters[output].weights[tap] / sum);
        }
    }
    return filters;
}

static uint8_t quantize(float value) {
    if (value <= 0.0f) return 0;
    if (value >= 255.0f) return 255;
    return (uint8_t)lrintf(value);
}

int h3cspeed_resize_rgb24_lanczos(const uint8_t *input, int frames,
                                 int input_width, int input_height,
                                 int output_width, int output_height,
                                 uint8_t **output) {
    if (output) *output = NULL;
    if (!input || !output || frames < 1 || input_width < 1 || input_height < 1 ||
        output_width < 1 || output_height < 1) return 0;

    size_t input_area = (size_t)input_width * (size_t)input_height;
    size_t output_area = (size_t)output_width * (size_t)output_height;
    if (input_area > SIZE_MAX / 3 || output_area > SIZE_MAX / 3 ||
        (size_t)frames > SIZE_MAX / (output_area * 3)) return 0;

    size_t output_bytes = (size_t)frames * output_area * 3;
    uint8_t *pixels = malloc(output_bytes);
    if (!pixels) return 0;
    if (input_width == output_width && input_height == output_height) {
        if ((size_t)frames > SIZE_MAX / (input_area * 3)) {
            free(pixels);
            return 0;
        }
        memcpy(pixels, input, (size_t)frames * input_area * 3);
        *output = pixels;
        return 1;
    }

    h3_filter *horizontal = build_filters(input_width, output_width);
    h3_filter *vertical = build_filters(input_height, output_height);
    if (!horizontal || !vertical) {
        free(horizontal);
        free(vertical);
        free(pixels);
        return 0;
    }

    size_t intermediate_values = (size_t)input_height * (size_t)output_width * 3;
    if (intermediate_values > SIZE_MAX / sizeof(float)) {
        free(horizontal); free(vertical); free(pixels);
        return 0;
    }
    float *intermediate = malloc(intermediate_values * sizeof(*intermediate));
    if (!intermediate) {
        free(horizontal); free(vertical); free(pixels);
        return 0;
    }

    size_t input_frame_bytes = input_area * 3;
    size_t output_frame_bytes = output_area * 3;
    for (int frame = 0; frame < frames; frame++) {
        const uint8_t *source = input + (size_t)frame * input_frame_bytes;
        uint8_t *destination = pixels + (size_t)frame * output_frame_bytes;

        for (int y = 0; y < input_height; y++) {
            for (int x = 0; x < output_width; x++) {
                const h3_filter *filter = &horizontal[x];
                float sum[3] = {0.0f, 0.0f, 0.0f};
                for (int tap = 0; tap < filter->count; tap++) {
                    int source_x = clamp_index(filter->first + tap, input_width);
                    const uint8_t *pixel = source +
                        ((size_t)y * (size_t)input_width + (size_t)source_x) * 3;
                    float weight = filter->weights[tap];
                    sum[0] += weight * (float)pixel[0];
                    sum[1] += weight * (float)pixel[1];
                    sum[2] += weight * (float)pixel[2];
                }
                float *out = intermediate +
                    ((size_t)y * (size_t)output_width + (size_t)x) * 3;
                out[0] = sum[0]; out[1] = sum[1]; out[2] = sum[2];
            }
        }

        for (int y = 0; y < output_height; y++) {
            const h3_filter *filter = &vertical[y];
            for (int x = 0; x < output_width; x++) {
                float sum[3] = {0.0f, 0.0f, 0.0f};
                for (int tap = 0; tap < filter->count; tap++) {
                    int source_y = clamp_index(filter->first + tap, input_height);
                    const float *pixel = intermediate +
                        ((size_t)source_y * (size_t)output_width + (size_t)x) * 3;
                    float weight = filter->weights[tap];
                    sum[0] += weight * pixel[0];
                    sum[1] += weight * pixel[1];
                    sum[2] += weight * pixel[2];
                }
                uint8_t *out = destination +
                    ((size_t)y * (size_t)output_width + (size_t)x) * 3;
                out[0] = quantize(sum[0]);
                out[1] = quantize(sum[1]);
                out[2] = quantize(sum[2]);
            }
        }
    }

    free(intermediate);
    free(horizontal);
    free(vertical);
    *output = pixels;
    return 1;
}
