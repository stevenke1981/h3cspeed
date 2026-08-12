#ifndef H3CSPEED_FL2VA_SIDECAR_H
#define H3CSPEED_FL2VA_SIDECAR_H

#include "h3_tokenizer.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Reconstruct the exact MiniMax FL2VA token sequence used by ComfyUI's
 * image-aware tokenizer. The returned IDs are owned by the caller. */
int h3cspeed_fl2va_build_token_ids(
    const h3_tokenizer *tokenizer,
    const char *prompt,
    const int *widths,
    const int *heights,
    size_t image_count,
    uint32_t **output,
    size_t *output_count,
    char *error,
    size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
