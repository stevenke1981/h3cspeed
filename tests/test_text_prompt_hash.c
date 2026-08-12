#include "h3_text_encoder.h"
#include "h3_tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *message) {
    fprintf(stderr, "text prompt hash failed: %s\n", message);
    exit(1);
}

static char *join_path(const char *root, const char *suffix) {
    size_t size = strlen(root) + strlen(suffix) + 2;
    char *result = (char *)malloc(size);
    if (result) snprintf(result, size, "%s/%s", root, suffix);
    return result;
}

static uint64_t hash_bf16(const uint16_t *values, size_t count) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < count; index++) {
        hash ^= values[index] & 255u;
        hash *= UINT64_C(1099511628211);
        hash ^= values[index] >> 8;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void progress(int completed, int total, void *opaque) {
    (void)completed;
    (void)total;
    (void)opaque;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL_ROOT PROMPT\n", argv[0]);
        return 2;
    }
    char *tokenizer_path = join_path(argv[1], "FL2VA/tokenizer/tokenizer.json");
    char *weights_path = join_path(argv[1], "FL2VA/text_encoder");
    if (!tokenizer_path || !weights_path) fail("path allocation failed");
    char error[512] = {0};
    h3_tokenizer *tokenizer = h3_tokenizer_load(
        tokenizer_path, error, sizeof(error));
    if (!tokenizer) fail(error);
    uint32_t *ids = NULL;
    size_t token_count = 0;
    if (!h3_tokenizer_encode(tokenizer, argv[2], 1, &ids, &token_count,
                             error, sizeof(error))) fail(error);
    h3_text_embedding embedding;
    if (!h3_text_encode_bf16(weights_path, "h3_shaders.metal", ids,
                              token_count, progress, NULL, &embedding,
                              error, sizeof(error))) fail(error);
    printf("prompt-hash %016llx tokens=%zu\n",
           (unsigned long long)hash_bf16(
               embedding.values, embedding.tokens * embedding.width),
           token_count);
    h3_text_embedding_free(&embedding);
    h3_tokenizer_ids_free(ids);
    h3_tokenizer_free(tokenizer);
    free(tokenizer_path);
    free(weights_path);
    return 0;
}
