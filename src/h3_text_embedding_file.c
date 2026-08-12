#include "h3_text_embedding_file.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define H3CSPEED_TEXT_FLAG_TAGS UINT32_C(1)
#define H3CSPEED_TEXT_VERSION UINT32_C(1)
#define H3CSPEED_TEXT_MAX_FILE (UINT64_C(64) * UINT64_C(1024) * UINT64_C(1024))

static void set_error(char *error, size_t error_size, const char *message) {
    if (error && error_size) {
        (void)snprintf(error, error_size, "%s", message);
    }
}

static void set_errorf(char *error, size_t error_size, const char *format,
                       unsigned long long value) {
    if (error && error_size) (void)snprintf(error, error_size, format, value);
}

static int add_u64(uint64_t left, uint64_t right, uint64_t *result) {
    if (left > UINT64_MAX - right) return 0;
    *result = left + right;
    return 1;
}

static int mul_u64(uint64_t left, uint64_t right, uint64_t *result) {
    if (left && right > UINT64_MAX / left) return 0;
    *result = left * right;
    return 1;
}

static uint32_t read_u32(const unsigned char *bytes) {
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
           ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static uint64_t read_u64(const unsigned char *bytes) {
    uint64_t value = 0;
    for (unsigned index = 0; index < 8; index++)
        value |= (uint64_t)bytes[index] << (index * 8u);
    return value;
}

/* Reject malformed UTF-8 in recipe and prompt metadata without normalizing it.
 * Expected prompt matching below remains byte-exact. */
static int valid_utf8(const unsigned char *bytes, size_t length) {
    size_t index = 0;
    while (index < length) {
        unsigned char first = bytes[index++];
        if (first <= 0x7fu) continue;
        unsigned count;
        uint32_t codepoint;
        if (first >= 0xc2u && first <= 0xdfu) {
            count = 1;
            codepoint = first & 0x1fu;
        } else if (first >= 0xe0u && first <= 0xefu) {
            count = 2;
            codepoint = first & 0x0fu;
        } else if (first >= 0xf0u && first <= 0xf4u) {
            count = 3;
            codepoint = first & 0x07u;
        } else {
            return 0;
        }
        if (length - index < count) return 0;
        for (unsigned part = 0; part < count; part++) {
            unsigned char next = bytes[index++];
            if ((next & 0xc0u) != 0x80u) return 0;
            codepoint = (codepoint << 6) | (next & 0x3fu);
        }
        if ((count == 2 && codepoint < 0x800u) ||
            (count == 3 && codepoint < 0x10000u) || codepoint > 0x10ffffu ||
            (codepoint >= 0xd800u && codepoint <= 0xdfffu)) return 0;
    }
    return 1;
}

static int read_exact(FILE *file, unsigned char *buffer, size_t size) {
    return size == 0 || fread(buffer, 1, size, file) == size;
}

int h3cspeed_text_embedding_load_file(
    const char *path, const char *expected_prompt,
    const uint32_t *expected_token_ids, size_t expected_token_count,
    const uint8_t expected_model_sha256[32],
    h3_text_embedding *output, char *error, size_t error_size) {
    FILE *file = NULL;
    unsigned char *bytes = NULL;
    uint16_t *values = NULL;
    uint8_t *tags = NULL;
    uint64_t file_size_u64;
    size_t file_size;
    uint64_t prompt_bytes, token_count, recipe_bytes, embedding_bytes;
    uint64_t tags_bytes, token_ids_bytes, expected_size;
    uint64_t header_embedding_bytes, header_tags_bytes, header_token_ids_bytes;
    uint64_t offset;
    uint64_t calculated_embedding_bytes, calculated_token_ids_bytes;
    size_t prompt_size, recipe_size, token_ids_size, embedding_size, tags_size;
    uint32_t header_size, version, width, flags;
    uint64_t elements;
    int ok = 0;

    if (error && error_size) error[0] = '\0';
    if (output) memset(output, 0, sizeof(*output));
    if (!path || !*path || !expected_prompt || !expected_model_sha256 || !output) {
        set_error(error, error_size, "sidecar path, expected prompt, model hash, and output are required");
        return 0;
    }
    if (expected_token_count && !expected_token_ids) {
        set_error(error, error_size, "expected token IDs are required");
        return 0;
    }
    file = fopen(path, "rb");
    if (!file) {
        if (error && error_size) (void)snprintf(error, error_size,
            "cannot open sidecar %s: %s", path, strerror(errno));
        return 0;
    }
    if (
#if defined(_WIN32)
        _fseeki64(file, 0, SEEK_END) != 0
#else
        fseek(file, 0, SEEK_END) != 0
#endif
    ) {
        set_error(error, error_size, "cannot seek sidecar");
        goto cleanup;
    }
    {
#if defined(_WIN32)
        __int64 end = _ftelli64(file);
#else
        long end = ftell(file);
#endif
        if (end < 0) {
            set_error(error, error_size, "cannot determine sidecar size");
            goto cleanup;
        }
        file_size_u64 = (uint64_t)end;
    }
    if (file_size_u64 < H3CSPEED_TEXT_EMBEDDING_FILE_HEADER_SIZE ||
        file_size_u64 > H3CSPEED_TEXT_MAX_FILE ||
        file_size_u64 > (uint64_t)SIZE_MAX) {
        set_error(error, error_size, "sidecar size is invalid or too large");
        goto cleanup;
    }
    file_size = (size_t)file_size_u64;
    bytes = (unsigned char *)malloc(file_size);
    if (!bytes) {
        set_error(error, error_size, "out of memory reading sidecar");
        goto cleanup;
    }
    if (
#if defined(_WIN32)
        _fseeki64(file, 0, SEEK_SET) != 0
#else
        fseek(file, 0, SEEK_SET) != 0
#endif
        || !read_exact(file, bytes, file_size)) {
        set_error(error, error_size, "cannot read complete sidecar");
        goto cleanup;
    }
    if (memcmp(bytes, "H3CSEV01", 8) != 0) {
        set_error(error, error_size, "sidecar magic is invalid");
        goto cleanup;
    }
    version = read_u32(bytes + 8);
    header_size = read_u32(bytes + 12);
    if (version != H3CSPEED_TEXT_VERSION ||
        header_size != H3CSPEED_TEXT_EMBEDDING_FILE_HEADER_SIZE) {
        set_error(error, error_size, "sidecar version or header size is invalid");
        goto cleanup;
    }
    prompt_bytes = read_u64(bytes + 16);
    token_count = read_u64(bytes + 24);
    recipe_bytes = read_u64(bytes + 32);
    width = read_u32(bytes + 40);
    flags = read_u32(bytes + 44);
    header_embedding_bytes = read_u64(bytes + 48);
    header_tags_bytes = read_u64(bytes + 56);
    header_token_ids_bytes = read_u64(bytes + 64);
    /* bytes 104..127 are reserved; bytes 72..103 carry the whole-model SHA-256. */
    for (size_t index = 104; index < H3CSPEED_TEXT_EMBEDDING_FILE_HEADER_SIZE; index++) {
        if (bytes[index] != 0) {
            set_error(error, error_size, "sidecar reserved header bytes are non-zero");
            goto cleanup;
        }
    }
    if (memcmp(bytes + 72, expected_model_sha256, 32) != 0) {
        set_error(error, error_size, "sidecar model fingerprint does not match expected SHA-256");
        goto cleanup;
    }
    if (width != H3CSPEED_TEXT_EMBEDDING_FILE_WIDTH ||
        flags != H3CSPEED_TEXT_FLAG_TAGS || token_count == 0 ||
        recipe_bytes == 0 || recipe_bytes > UINT32_C(65536)) {
        set_error(error, error_size, "sidecar dimensions or flags are invalid");
        goto cleanup;
    }
    if (!mul_u64(token_count, UINT64_C(4), &calculated_token_ids_bytes) ||
        !mul_u64(token_count, (uint64_t)width, &elements) ||
        !mul_u64(elements, UINT64_C(2), &calculated_embedding_bytes) ||
        token_count != (uint64_t)expected_token_count ||
        calculated_token_ids_bytes != header_token_ids_bytes ||
        calculated_embedding_bytes != header_embedding_bytes ||
        header_tags_bytes != token_count) {
        set_error(error, error_size, "sidecar tensor byte lengths are invalid");
        goto cleanup;
    }
    offset = H3CSPEED_TEXT_EMBEDDING_FILE_HEADER_SIZE;
    if (!add_u64(offset, prompt_bytes, &offset) ||
        !add_u64(offset, recipe_bytes, &offset) ||
        !add_u64(offset, header_token_ids_bytes, &offset) ||
        !add_u64(offset, header_embedding_bytes, &offset) ||
        !add_u64(offset, header_tags_bytes, &expected_size) ||
        expected_size != file_size_u64 || prompt_bytes > (uint64_t)SIZE_MAX ||
        recipe_bytes > (uint64_t)SIZE_MAX || header_token_ids_bytes > (uint64_t)SIZE_MAX ||
        header_embedding_bytes > (uint64_t)SIZE_MAX || header_tags_bytes > (uint64_t)SIZE_MAX) {
        set_error(error, error_size, "sidecar has trailing bytes or truncated payload");
        goto cleanup;
    }
    token_ids_bytes = header_token_ids_bytes;
    embedding_bytes = header_embedding_bytes;
    tags_bytes = header_tags_bytes;
    prompt_size = (size_t)prompt_bytes;
    recipe_size = (size_t)recipe_bytes;
    token_ids_size = (size_t)token_ids_bytes;
    embedding_size = (size_t)embedding_bytes;
    tags_size = (size_t)tags_bytes;
    if (prompt_bytes != (uint64_t)strlen(expected_prompt) ||
        memcmp(bytes + H3CSPEED_TEXT_EMBEDDING_FILE_HEADER_SIZE, expected_prompt,
               prompt_size) != 0 ||
        !valid_utf8(bytes + H3CSPEED_TEXT_EMBEDDING_FILE_HEADER_SIZE,
                    prompt_size)) {
        set_error(error, error_size, "sidecar prompt does not match expected UTF-8 bytes");
        goto cleanup;
    }
    {
        const unsigned char *recipe = bytes + H3CSPEED_TEXT_EMBEDDING_FILE_HEADER_SIZE + prompt_size;
        const char expected_recipe[] = H3CSPEED_TEXT_EMBEDDING_FILE_RECIPE;
        if (recipe_size != sizeof(expected_recipe) - 1u ||
            memcmp(recipe, expected_recipe, sizeof(expected_recipe) - 1u) != 0 ||
            !valid_utf8(recipe, recipe_size)) {
            set_error(error, error_size, "sidecar recipe is invalid or unsupported");
            goto cleanup;
        }
    }
    {
        const unsigned char *ids = bytes + H3CSPEED_TEXT_EMBEDDING_FILE_HEADER_SIZE + prompt_size + recipe_size;
        for (size_t index = 0; index < expected_token_count; index++) {
            uint32_t actual = read_u32(ids + index * 4u);
            if (actual != expected_token_ids[index]) {
                set_errorf(error, error_size, "sidecar token ID mismatch at index %llu",
                           (unsigned long long)index);
                goto cleanup;
            }
        }
    }
    if (elements > (uint64_t)(SIZE_MAX / sizeof(*values))) {
        set_error(error, error_size, "sidecar embedding allocation overflows size_t");
        goto cleanup;
    }
    values = (uint16_t *)malloc(embedding_size);
    tags = (uint8_t *)malloc(tags_size);
    if (!values || !tags) {
        set_error(error, error_size, "out of memory allocating sidecar embedding");
        goto cleanup;
    }
    {
        const unsigned char *payload = bytes + H3CSPEED_TEXT_EMBEDDING_FILE_HEADER_SIZE +
            prompt_size + recipe_size + token_ids_size;
        memcpy(values, payload, embedding_size);
        payload += embedding_size;
        for (size_t index = 0; index < tags_size; index++) {
            if (payload[index] > 1u) {
                set_error(error, error_size, "sidecar tag is not 0 or 1");
                goto cleanup;
            }
        }
        memcpy(tags, payload, tags_size);
    }
    output->tokens = (size_t)token_count;
    output->width = (size_t)width;
    output->values = values;
    output->tags = tags;
    values = NULL;
    tags = NULL;
    ok = 1;
cleanup:
    free(values);
    free(tags);
    free(bytes);
    if (file) (void)fclose(file);
    if (!ok && output) h3_text_embedding_free(output);
    return ok;
}
