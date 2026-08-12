#ifndef H3CSPEED_TEXT_EMBEDDING_FILE_H
#define H3CSPEED_TEXT_EMBEDDING_FILE_H

#include "h3_text_encoder.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Sidecar format emitted by scripts/encode_h3_quantized_prompt.py. Version 1
 * remains readable for existing T2V artifacts; version 2 adds fail-closed
 * FL2VA keyframe metadata without changing the public H3 API. */
#define H3CSPEED_TEXT_EMBEDDING_FILE_HEADER_SIZE 128u
#define H3CSPEED_TEXT_EMBEDDING_FILE_WIDTH H3_TEXT_HIDDEN_SIZE
#define H3CSPEED_TEXT_EMBEDDING_FILE_RECIPE "h3cspeed-conditioning-v1"
#define H3CSPEED_TEXT_EMBEDDING_FILE_RECIPE_V2 "h3cspeed-conditioning-v2"

#define H3CSPEED_TEXT_EMBEDDING_MODE_T2V 0u
#define H3CSPEED_TEXT_EMBEDDING_MODE_FL2VA_I2V 1u
#define H3CSPEED_TEXT_EMBEDDING_ROLE_FIRST 1u
#define H3CSPEED_TEXT_EMBEDDING_ROLE_LAST 2u
#define H3CSPEED_TEXT_EMBEDDING_RESIZE_STRETCH 0u
#define H3CSPEED_TEXT_EMBEDDING_RESIZE_COVER 1u

typedef struct {
    uint32_t mode;
    uint32_t keyframe_role;
    uint32_t keyframe_count;
    uint32_t keyframe_order;
    uint32_t first_resize_policy;
    uint32_t last_resize_policy;
    uint32_t render_width;
    uint32_t render_height;
    const uint8_t *first_image_sha256;
    const uint8_t *last_image_sha256;
} h3_text_embedding_file_expectation;

typedef struct {
    char directory[512];
    char path[512];
    uint8_t sha256[32];
} h3cspeed_keyframe_snapshot;

/* Copy a keyframe into a private, uniquely named directory before either
 * hashing or decoding it. The returned path and digest therefore describe
 * the same immutable snapshot even if the caller's original path changes. */
int h3cspeed_keyframe_snapshot_create(
    const char *source,
    h3cspeed_keyframe_snapshot *snapshot,
    char *error,
    size_t error_size);
void h3cspeed_keyframe_snapshot_discard(h3cspeed_keyframe_snapshot *snapshot);

/* Hash the exact on-disk bytes of a keyframe. The helper is shared by the
 * native loader so an I2V sidecar cannot be replayed with a substituted image. */
int h3cspeed_sha256_file(const char *path, uint8_t output[32],
                         char *error, size_t error_size);

/* Little-endian header offsets (the 32-byte whole-model SHA-256 is 72..103).
 * Version 2 stores mode/role/count/order/first+last resize in bytes 104..109,
 * render geometry at 112..119 and metadata byte count at 120..123. Version 2 payload
 * order is prompt UTF-8, recipe UTF-8, image SHA-256 metadata, uint32 token
 * IDs, BF16 values, then uint8 tags. */

int h3cspeed_text_embedding_load_file_ex(
    const char *path,
    const char *expected_prompt,
    const uint32_t *expected_token_ids,
    size_t expected_token_count,
    const uint8_t expected_model_sha256[32],
    const h3_text_embedding_file_expectation *expectation,
    h3_text_embedding *output,
    char *error,
    size_t error_size);

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
