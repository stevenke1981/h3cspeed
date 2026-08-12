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

int main(void) {
    const char *path = "text_embedding_file_test.bin";
    const uint32_t ids[] = {11u, 22u};
    uint8_t expected_hash[32] = {0};
    expected_hash[0] = 0x42;
    char error[256];
    h3_text_embedding output;
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
    (void)remove(path);
    puts("text_embedding_file: ok");
    return 0;
}
