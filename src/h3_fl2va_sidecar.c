#include "h3_fl2va_sidecar.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t *values;
    size_t count;
    size_t capacity;
} h3_sidecar_ids;

static int ids_reserve(h3_sidecar_ids *ids, size_t extra) {
    if (extra > SIZE_MAX - ids->count) return 0;
    size_t needed = ids->count + extra;
    if (needed <= ids->capacity) return 1;
    size_t capacity = ids->capacity ? ids->capacity : 64;
    while (capacity < needed) {
        if (capacity > SIZE_MAX / 2) { capacity = needed; break; }
        capacity *= 2;
    }
    if (capacity > SIZE_MAX / sizeof(*ids->values)) return 0;
    uint32_t *grown = realloc(ids->values, capacity * sizeof(*grown));
    if (!grown) return 0;
    ids->values = grown;
    ids->capacity = capacity;
    return 1;
}

static int ids_append(h3_sidecar_ids *ids, const uint32_t *values, size_t count) {
    if (!ids_reserve(ids, count)) return 0;
    if (count) memcpy(ids->values + ids->count, values, count * sizeof(*values));
    ids->count += count;
    return 1;
}

static int ids_text(const h3_tokenizer *tokenizer, const char *text,
                    h3_sidecar_ids *ids, char *error, size_t error_size) {
    uint32_t *values = NULL;
    size_t count = 0;
    if (!h3_tokenizer_encode(tokenizer, text, 0, &values, &count, error, error_size)) return 0;
    int ok = ids_append(ids, values, count);
    h3_tokenizer_ids_free(values);
    if (!ok && error && error_size)
        (void)snprintf(error, error_size, "out of memory constructing I2V token IDs");
    return ok;
}

int h3cspeed_fl2va_build_token_ids(
    const h3_tokenizer *tokenizer, const char *prompt,
    const int *widths, const int *heights, size_t image_count,
    uint32_t **output, size_t *output_count, char *error, size_t error_size) {
    h3_sidecar_ids ids = {0};
    static const uint32_t vision_start = UINT32_C(151652);
    static const uint32_t vision_pad = UINT32_C(151655);
    static const uint32_t vision_end = UINT32_C(151653);
    if (error && error_size) error[0] = '\0';
    if (output) *output = NULL;
    if (output_count) *output_count = 0;
    if (!tokenizer || !prompt || !widths || !heights || image_count == 0 ||
        image_count > 2 || !output || !output_count) {
        if (error && error_size) (void)snprintf(error, error_size, "invalid I2V token builder arguments");
        return 0;
    }
    for (size_t image = 0; image < image_count; image++) {
        char label[64];
        int grid_h = heights[image] / 16;
        int grid_w = widths[image] / 16;
        size_t patches;
        if (grid_h < 2 || grid_w < 2 || (grid_h % 2) || (grid_w % 2) ||
            (widths[image] % 32) != 0 || (heights[image] % 32) != 0) {
            if (error && error_size) (void)snprintf(error, error_size,
                "I2V render geometry is not divisible by Qwen patch grid");
            free(ids.values);
            return 0;
        }
        patches = (size_t)(grid_h / 2) * (size_t)(grid_w / 2);
        (void)snprintf(label, sizeof(label), "<Picture %zu>: ", image + 1);
        if (!ids_text(tokenizer, label, &ids, error, error_size) ||
            !ids_append(&ids, &vision_start, 1) || !ids_reserve(&ids, patches + 1)) {
            free(ids.values);
            return 0;
        }
        for (size_t patch = 0; patch < patches; patch++) ids.values[ids.count++] = vision_pad;
        ids.values[ids.count++] = vision_end;
    }
    if (!ids_text(tokenizer, prompt, &ids, error, error_size)) {
        free(ids.values);
        return 0;
    }
    *output = ids.values;
    *output_count = ids.count;
    return 1;
}
