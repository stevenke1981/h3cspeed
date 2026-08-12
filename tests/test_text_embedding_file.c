#include "h3_text_embedding_file.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void put32(unsigned char *p, uint32_t v) {
    for (unsigned i = 0; i < 4; i++) p[i] = (unsigned char)(v >> (i * 8u));
}
static void put64(unsigned char *p, uint64_t v) {
    for (unsigned i = 0; i < 8; i++) p[i] = (unsigned char)(v >> (i * 8u));
}

static int write_sidecar(const char *path, const char *prompt,
                         const uint32_t *ids, size_t count, int bad_tag,
                         int trailing, int bad_recipe, int bad_hash) {
    const char *recipe = bad_recipe ? "wrong-recipe" :
        H3CSPEED_TEXT_EMBEDDING_FILE_RECIPE;
    const size_t prompt_bytes = strlen(prompt), recipe_bytes = strlen(recipe);
    const size_t embedding_bytes = count * 5120u * sizeof(uint16_t);
    const size_t ids_bytes = count * sizeof(uint32_t);
    const size_t tags_bytes = count;
    const size_t total = 128u + prompt_bytes + recipe_bytes + ids_bytes +
                         embedding_bytes + tags_bytes + (trailing ? 1u : 0u);
    unsigned char *buffer = (unsigned char *)calloc(total, 1);
    if (!buffer) return 0;
    memcpy(buffer, "H3CSEV01", 8);
    put32(buffer + 8, 1u);
    put32(buffer + 12, 128u);
    put64(buffer + 16, prompt_bytes);
    put64(buffer + 24, count);
    put64(buffer + 32, recipe_bytes);
    put32(buffer + 40, 5120u);
    put32(buffer + 44, 1u);
    put64(buffer + 48, embedding_bytes);
    put64(buffer + 56, tags_bytes);
    put64(buffer + 64, ids_bytes);
    buffer[72] = bad_hash ? 0x43 : 0x42;
    size_t offset = 128u;
    memcpy(buffer + offset, prompt, prompt_bytes); offset += prompt_bytes;
    memcpy(buffer + offset, recipe, recipe_bytes); offset += recipe_bytes;
    for (size_t i = 0; i < count; i++) {
        put32(buffer + offset, ids[i]); offset += 4u;
    }
    /* Deterministic BF16 values: all zero except the first element. */
    buffer[offset] = 0x80; buffer[offset + 1] = 0x3f; offset += embedding_bytes;
    memset(buffer + offset, bad_tag ? 2 : 1, tags_bytes); offset += tags_bytes;
    if (trailing) buffer[offset] = 0xa5;
    FILE *file = fopen(path, "wb");
    int ok = 0;
    if (file) {
        int wrote = fwrite(buffer, 1, total, file) == total;
        int closed = fclose(file) == 0;
        ok = wrote && closed;
    }
    free(buffer);
    return ok;
}

static int write_sidecar_v2(const char *path, const char *prompt,
                            const uint32_t *ids, size_t count,
                            const uint8_t model_hash[32],
                            const uint8_t image_hash[32]) {
    const char *recipe = H3CSPEED_TEXT_EMBEDDING_FILE_RECIPE_V2;
    const size_t prompt_bytes = strlen(prompt), recipe_bytes = strlen(recipe);
    const size_t metadata_bytes = 32u;
    const size_t embedding_bytes = count * 5120u * sizeof(uint16_t);
    const size_t ids_bytes = count * sizeof(uint32_t);
    const size_t tags_bytes = count;
    const size_t total = 128u + prompt_bytes + recipe_bytes + metadata_bytes +
                         ids_bytes + embedding_bytes + tags_bytes;
    unsigned char *buffer = (unsigned char *)calloc(total, 1);
    if (!buffer) return 0;
    memcpy(buffer, "H3CSEV01", 8);
    put32(buffer + 8, 2u);
    put32(buffer + 12, 128u);
    put64(buffer + 16, prompt_bytes);
    put64(buffer + 24, count);
    put64(buffer + 32, recipe_bytes);
    put32(buffer + 40, 5120u);
    put32(buffer + 44, 1u);
    put64(buffer + 48, embedding_bytes);
    put64(buffer + 56, tags_bytes);
    put64(buffer + 64, ids_bytes);
    memcpy(buffer + 72, model_hash, 32u);
    buffer[104] = H3CSPEED_TEXT_EMBEDDING_MODE_FL2VA_I2V;
    buffer[105] = H3CSPEED_TEXT_EMBEDDING_ROLE_FIRST;
    buffer[106] = 1u;
    buffer[107] = H3CSPEED_TEXT_EMBEDDING_ROLE_FIRST;
    buffer[108] = H3CSPEED_TEXT_EMBEDDING_RESIZE_STRETCH;
    buffer[109] = H3CSPEED_TEXT_EMBEDDING_RESIZE_COVER;
    put32(buffer + 112, 64u);
    put32(buffer + 116, 64u);
    put32(buffer + 120, (uint32_t)metadata_bytes);
    size_t offset = 128u;
    memcpy(buffer + offset, prompt, prompt_bytes); offset += prompt_bytes;
    memcpy(buffer + offset, recipe, recipe_bytes); offset += recipe_bytes;
    memcpy(buffer + offset, image_hash, metadata_bytes); offset += metadata_bytes;
    for (size_t i = 0; i < count; i++) {
        put32(buffer + offset, ids[i]); offset += 4u;
    }
    buffer[offset] = 0x80; buffer[offset + 1] = 0x3f;
    buffer[offset + 2] = 0x00; buffer[offset + 3] = 0x40;
    offset += embedding_bytes;
    memset(buffer + offset, 1, tags_bytes);
    FILE *file = fopen(path, "wb");
    int ok = 0;
    if (file) {
        int wrote = fwrite(buffer, 1, total, file) == total;
        int closed = fclose(file) == 0;
        ok = wrote && closed;
    }
    free(buffer);
    return ok;
}

int main(void) {
    const char *path = "text_embedding_file_test.bin";
    const char *v2_path = "text_embedding_file_v2_test.bin";
    const char *sha_path = "text_embedding_sha_test.bin";
    const uint32_t ids[] = {11u, 22u};
    uint8_t expected_hash[32] = {0};
    expected_hash[0] = 0x42;
    char error[256];
    h3_text_embedding output;
    {
        static const uint8_t expected_empty[32] = {
            0xe3,0xb0,0xc4,0x42,0x98,0xfc,0x1c,0x14,0x9a,0xfb,0xf4,0xc8,0x99,0x6f,0xb9,0x24,
            0x27,0xae,0x41,0xe4,0x64,0x9b,0x93,0x4c,0xa4,0x95,0x99,0x1b,0x78,0x52,0xb8,0x55};
        static const uint8_t expected_abc[32] = {
            0xba,0x78,0x16,0xbf,0x8f,0x01,0xcf,0xea,0x41,0x41,0x40,0xde,0x5d,0xae,0x22,0x23,
            0xb0,0x03,0x61,0xa3,0x96,0x17,0x7a,0x9c,0xb4,0x10,0xff,0x61,0xf2,0x00,0x15,0xad};
        uint8_t digest[32];
        FILE *sha_file = fopen(sha_path, "wb");
        if (!sha_file || fclose(sha_file) != 0 ||
            !h3cspeed_sha256_file(sha_path, digest, error, sizeof(error)) ||
            memcmp(digest, expected_empty, sizeof(digest)) != 0) return 13;
        sha_file = fopen(sha_path, "wb");
        if (!sha_file || fwrite("abc", 1, 3, sha_file) != 3 || fclose(sha_file) != 0 ||
            !h3cspeed_sha256_file(sha_path, digest, error, sizeof(error)) ||
            memcmp(digest, expected_abc, sizeof(digest)) != 0) return 14;
        (void)remove(sha_path);
    }
    if (!write_sidecar(path, "GPU 測試", ids, 2, 0, 0, 0, 0)) return 1;
    if (!h3cspeed_text_embedding_load_file(path, "GPU 測試", ids, 2,
                                           expected_hash,
                                           &output, error, sizeof(error))) return 2;
    if (output.tokens != 2 || output.width != 5120 || !output.values ||
        !output.tags || output.tags[0] != 1 || output.tags[1] != 1) return 3;
    h3_text_embedding_free(&output);
    if (h3cspeed_text_embedding_load_file(path, "wrong", ids, 2,
                                          expected_hash,
                                          &output, error, sizeof(error))) return 4;
    if (!write_sidecar(path, "GPU 測試", ids, 2, 0, 1, 0, 0)) return 5;
    if (h3cspeed_text_embedding_load_file(path, "GPU 測試", ids, 2,
                                          expected_hash,
                                          &output, error, sizeof(error))) return 6;
    if (!write_sidecar(path, "GPU 測試", ids, 2, 1, 0, 0, 0)) return 7;
    if (h3cspeed_text_embedding_load_file(path, "GPU 測試", ids, 2,
                                          expected_hash,
                                          &output, error, sizeof(error))) return 8;
    if (!write_sidecar(path, "GPU 測試", ids, 2, 0, 0, 1, 0)) return 9;
    if (h3cspeed_text_embedding_load_file(path, "GPU 測試", ids, 2,
                                          expected_hash,
                                          &output, error, sizeof(error))) return 10;
    if (!write_sidecar(path, "GPU 測試", ids, 2, 0, 0, 0, 1)) return 11;
    if (h3cspeed_text_embedding_load_file(path, "GPU 測試", ids, 2,
                                          expected_hash,
                                          &output, error, sizeof(error))) return 12;
    {
        uint8_t image_hash[32];
        h3_text_embedding_file_expectation expectation = {0};
        memset(image_hash, 0xa5, sizeof(image_hash));
        expectation.mode = H3CSPEED_TEXT_EMBEDDING_MODE_FL2VA_I2V;
        expectation.keyframe_role = H3CSPEED_TEXT_EMBEDDING_ROLE_FIRST;
        expectation.keyframe_count = 1u;
        expectation.keyframe_order = H3CSPEED_TEXT_EMBEDDING_ROLE_FIRST;
        expectation.first_resize_policy = H3CSPEED_TEXT_EMBEDDING_RESIZE_STRETCH;
        expectation.last_resize_policy = H3CSPEED_TEXT_EMBEDDING_RESIZE_COVER;
        expectation.render_width = 64u;
        expectation.render_height = 64u;
        expectation.first_image_sha256 = image_hash;
        if (!write_sidecar(path, "GPU 測試", ids, 2, 0, 0, 0, 0)) return 15;
        if (h3cspeed_text_embedding_load_file_ex(
                path, "GPU 測試", ids, 2, expected_hash, &expectation,
                &output, error, sizeof(error))) return 16;
        if (!write_sidecar_v2(v2_path, "GPU 測試", ids, 2,
                              expected_hash, image_hash)) return 17;
        if (!h3cspeed_text_embedding_load_file_ex(
                v2_path, "GPU 測試", ids, 2, expected_hash, &expectation,
                &output, error, sizeof(error))) return 18;
        if (output.tokens != 2 || output.width != 5120 || !output.values ||
            !output.tags || output.values[0] != 0x3f80u ||
            output.values[1] != 0x4000u || output.tags[0] != 1u ||
            output.tags[1] != 1u) return 19;
        h3_text_embedding_free(&output);
        image_hash[0] ^= 1u;
        if (h3cspeed_text_embedding_load_file_ex(
                v2_path, "GPU 測試", ids, 2, expected_hash, &expectation,
                &output, error, sizeof(error))) return 20;
    }
    {
        const char *snapshot_source = "text_embedding_snapshot_source.bin";
        h3cspeed_keyframe_snapshot snapshot;
        uint8_t source_digest[32];
        uint8_t snapshot_digest[32];
        char snapshot_path[sizeof(snapshot.path)];
        FILE *snapshot_file = fopen(snapshot_source, "wb");
        if (!snapshot_file || fwrite("snapshot-A", 1, 10, snapshot_file) != 10 ||
            fclose(snapshot_file) != 0) return 21;
        if (!h3cspeed_sha256_file(snapshot_source, source_digest, error, sizeof(error)) ||
            !h3cspeed_keyframe_snapshot_create(
                snapshot_source, &snapshot, error, sizeof(error))) return 22;
        if (memcmp(source_digest, snapshot.sha256, sizeof(source_digest)) != 0)
            return 23;
        (void)snprintf(snapshot_path, sizeof(snapshot_path), "%s", snapshot.path);
        snapshot_file = fopen(snapshot_source, "wb");
        if (!snapshot_file || fwrite("snapshot-B", 1, 10, snapshot_file) != 10 ||
            fclose(snapshot_file) != 0) return 24;
        if (!h3cspeed_sha256_file(snapshot.path, snapshot_digest, error, sizeof(error)) ||
            memcmp(source_digest, snapshot_digest, sizeof(source_digest)) != 0)
            return 25;
        h3cspeed_keyframe_snapshot_discard(&snapshot);
        snapshot_file = fopen(snapshot_path, "rb");
        if (snapshot_file) { (void)fclose(snapshot_file); return 26; }
        (void)remove(snapshot_source);
    }
    (void)remove(path);
    (void)remove(v2_path);
    puts("text_embedding_file: ok");
    return 0;
}
