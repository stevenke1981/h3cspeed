#include "h3_model_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TENSORS 128
#define MAX_NAME 160
#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        return 0; \
    } \
} while (0)

typedef struct {
    h3cspeed_model_tensor_info tensor[MAX_TENSORS];
    char name[MAX_TENSORS][MAX_NAME];
    size_t count;
} fixture;

static h3cspeed_model_tensor_info *add(
        fixture *f, const char *prefix, const char *suffix,
        int ndim, uint64_t d0, uint64_t d1) {
    if (f->count >= MAX_TENSORS) abort();
    size_t i = f->count++;
    if (prefix && *prefix)
        (void)snprintf(f->name[i], sizeof(f->name[i]), "%s.%s", prefix, suffix);
    else
        (void)snprintf(f->name[i], sizeof(f->name[i]), "%s", suffix);
    f->tensor[i].name = f->name[i];
    f->tensor[i].ndim = ndim;
    memset(f->tensor[i].shape, 0, sizeof(f->tensor[i].shape));
    f->tensor[i].shape[0] = d0;
    if (ndim > 1) f->tensor[i].shape[1] = d1;
    return &f->tensor[i];
}

static void make_fixture(
        fixture *f, const char *prefix, uint64_t hidden,
        unsigned layers, int skip_layer, int adaln) {
    memset(f, 0, sizeof(*f));
    add(f, prefix, "video_patch_proj.weight", 2, hidden, 96);
    add(f, prefix, "audio_patch_proj.weight", 2, hidden, 32);
    add(f, prefix, "blocks.0.attn.q_norm.weight", 1, 128, 0);
    add(f, prefix, "blocks.0.attn.qkv_proj.weight", 2, 3 * 56 * 128, hidden);
    add(f, prefix, "blocks.0.mlp.fc1.weight", 2, 2 * 14336, hidden);
    add(f, prefix, "condition_proj.weight", 2, hidden, 5120);
    add(f, prefix, "rope.inv_freq", 1, 16, 0);
    if (adaln) {
        add(f, prefix, "adaln_t_table", 2, 24, 2688);
    } else {
        add(f, prefix, "time_embedder.proj_in.weight", 2, hidden, 256);
        add(f, prefix, "time_embedder.proj_out.weight", 2, 2688, hidden);
    }
    for (unsigned block = 1; block < layers; block++) {
        if ((int)block == skip_layer) continue;
        char suffix[80];
        (void)snprintf(suffix, sizeof(suffix), "blocks.%u.norm1.weight", block);
        add(f, prefix, suffix, 1, hidden, 0);
    }
    add(f, prefix, "token_refiner.blocks.0.norm1.weight", 1, hidden, 0);
    add(f, prefix, "token_refiner.blocks.1.norm1.weight", 1, hidden, 0);
}

static h3cspeed_model_tensor_info *find_tensor(fixture *f, const char *suffix) {
    size_t suffix_len = strlen(suffix);
    for (size_t i = 0; i < f->count; i++) {
        size_t name_len = strlen(f->tensor[i].name);
        if (name_len >= suffix_len &&
            strcmp(f->tensor[i].name + name_len - suffix_len, suffix) == 0)
            return &f->tensor[i];
    }
    return NULL;
}

static int detect(fixture *f, const char *prefix,
                  h3cspeed_h3_model_config *config, char *error) {
    return h3cspeed_h3_model_config_detect(
        f->tensor, f->count, prefix, config, error, 256);
}

static int standard_model(void) {
    fixture f;
    h3cspeed_h3_model_config c;
    char error[256] = {0};
    make_fixture(&f, "", 5376, 50, -1, 0);
    CHECK(detect(&f, "", &c, error));
    CHECK(c.variant == H3CSPEED_H3_VARIANT_TIME_EMBEDDER);
    CHECK(c.hidden_size == 5376 && c.num_layers == 50);
    CHECK(c.token_refiner_num_layers == 2);
    CHECK(c.num_attention_heads == 56 && c.attention_head_dim == 128);
    CHECK(c.ffn_hidden_size == 14336);
    CHECK(c.video_latent_channels == 24 && c.audio_latent_channels == 32);
    CHECK(c.text_dim == 5120 && c.rope_inv_freq_len == 16);
    CHECK(h3cspeed_h3_model_compatibility(&c) == 0);
    return 1;
}

static int prefixed_model(void) {
    fixture f;
    h3cspeed_h3_model_config c;
    char error[256] = {0};
    const char *prefix = "model.diffusion_model";
    make_fixture(&f, prefix, 5376, 50, -1, 0);
    CHECK(detect(&f, prefix, &c, error));
    CHECK(h3cspeed_h3_model_compatibility(&c) == 0);
    return 1;
}

static int adaln_is_detected_but_rejected(void) {
    fixture f;
    h3cspeed_h3_model_config c;
    char error[256] = {0};
    char report[256] = {0};
    make_fixture(&f, "", 5376, 50, -1, 1);
    CHECK(detect(&f, "", &c, error));
    CHECK(c.variant == H3CSPEED_H3_VARIANT_ADALN_CURVES);
    CHECK(c.adaln_curve_grid == 24 && c.time_embed_dim == 2688);
    CHECK(h3cspeed_h3_model_compatibility(&c) == H3CSPEED_H3_INCOMPAT_VARIANT);
    CHECK(h3cspeed_h3_model_format_compatibility(&c, report, sizeof(report)));
    CHECK(strstr(report, "adaln-curves") != NULL);
    return 1;
}

static int sparse_blocks_fail(void) {
    fixture f;
    h3cspeed_h3_model_config c;
    char error[256] = {0};
    make_fixture(&f, "", 5376, 4, 2, 0);
    CHECK(!detect(&f, "", &c, error));
    CHECK(strstr(error, "not contiguous") != NULL);
    return 1;
}

static int malformed_qkv_fails(void) {
    fixture f;
    h3cspeed_h3_model_config c;
    char error[256] = {0};
    make_fixture(&f, "", 5376, 50, -1, 0);
    h3cspeed_model_tensor_info *qkv =
        find_tensor(&f, "blocks.0.attn.qkv_proj.weight");
    CHECK(qkv != NULL);
    qkv->shape[0]++;
    CHECK(!detect(&f, "", &c, error));
    CHECK(strstr(error, "not divisible") != NULL);
    return 1;
}

static int alternate_hidden_is_reported(void) {
    fixture f;
    h3cspeed_h3_model_config c;
    char error[256] = {0};
    char report[512] = {0};
    make_fixture(&f, "", 4096, 50, -1, 0);
    CHECK(detect(&f, "", &c, error));
    h3cspeed_h3_compatibility mask = h3cspeed_h3_model_compatibility(&c);
    CHECK(mask & H3CSPEED_H3_INCOMPAT_HIDDEN_SIZE);
    CHECK(mask & H3CSPEED_H3_INCOMPAT_TIME_HIDDEN_SIZE);
    CHECK(h3cspeed_h3_model_format_compatibility(&c, report, sizeof(report)));
    CHECK(strstr(report, "hidden_size=4096") != NULL);
    return 1;
}

static int duplicate_tensor_fails(void) {
    fixture f;
    h3cspeed_h3_model_config c;
    char error[256] = {0};
    make_fixture(&f, "", 5376, 50, -1, 0);
    add(&f, "", "rope.inv_freq", 1, 16, 0);
    CHECK(!detect(&f, "", &c, error));
    CHECK(strstr(error, "duplicate") != NULL);
    return 1;
}

int main(void) {
    int passed = 0;
    passed += standard_model();
    passed += prefixed_model();
    passed += adaln_is_detected_but_rejected();
    passed += sparse_blocks_fail();
    passed += malformed_qkv_fails();
    passed += alternate_hidden_is_reported();
    passed += duplicate_tensor_fails();
    if (passed != 7) return 1;
    printf("h3 model config tests: %d passed\n", passed);
    return 0;
}
