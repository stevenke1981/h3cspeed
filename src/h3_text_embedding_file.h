#ifndef H3CSPEED_TEXT_EMBEDDING_FILE_H
#define H3CSPEED_TEXT_EMBEDDING_FILE_H

#include "h3_text_encoder.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Fixed sidecar format version emitted by scripts/encode_h3_quantized_prompt.py.
 * The loader is intentionally additive and does not alter the public H3 API. */
#define H3CSPEED_TEXT_EMBEDDING_FILE_HEADER_SIZE 128u
#define H3CSPEED_TEXT_EMBEDDING_FILE_WIDTH H3_TEXT_HIDDEN_SIZE
#define H3CSPEED_TEXT_EMBEDDING_FILE_RECIPE "h3cspeed-conditioning-v1"

/* Little-endian header offsets (the 32-byte whole-model SHA-256 is 72..103
 * and the 24-byte reserved tail is 104..127). Payload order is prompt UTF-8, recipe
 * UTF-8, uint32 token IDs, BF16 values, then uint8 tags. */

/* Load a ComfyUI GPU conditioning sidecar.  The prompt bytes and token IDs in
 * the file must match the caller's expected values byte-for-byte before any
 * output is committed.  On success output owns values/tags and must be freed
 * with h3_text_embedding_free(). */
int h3cspeed_text_embedding_load_file(
    const char *path,
    const char *expected_prompt,
    const uint32_t *expected_token_ids,
    size_t expected_token_count,
    const uint8_t expected_model_sha256[32],
    h3_text_embedding *output,
    char *error,
    size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
