#include "h3_text_encoder.h"
#include "h3_tokenizer.h"

#include <stdint.h>
#include <math.h>
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

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
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
    double sum = 0.0, square_sum = 0.0;
    float absolute_max = 0.0f;
    size_t embedding_count = embedding.tokens * embedding.width;
    for (size_t index = 0; index < embedding_count; index++) {
        float value = bf16_to_f32(embedding.values[index]);
        sum += value;
        square_sum += (double)value * value;
        if (fabsf(value) > absolute_max) absolute_max = fabsf(value);
    }
    double mean = embedding_count ? sum / (double)embedding_count : 0.0;
    double variance = embedding_count ?
        square_sum / (double)embedding_count - mean * mean : 0.0;
    printf("prompt-stats mean=%.9g std=%.9g absmax=%.9g shape=%zux%zu\n",
           mean, sqrt(variance > 0.0 ? variance : 0.0), absolute_max,
           embedding.tokens, embedding.width);
    const char *dump_path = getenv("H3_TEXT_DUMP");
    if (dump_path && *dump_path) {
        FILE *dump = fopen(dump_path, "wb");
        if (!dump || fwrite(embedding.values, sizeof(*embedding.values),
                            embedding_count, dump) != embedding_count ||
            fclose(dump) != 0) fail("cannot write text embedding dump");
    }
    h3_text_embedding_free(&embedding);
    h3_tokenizer_ids_free(ids);
    h3_tokenizer_free(tokenizer);
    free(tokenizer_path);
    free(weights_path);
    return 0;
}
