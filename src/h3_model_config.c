#include "h3_model_config.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define H3_EXPECTED_HIDDEN_SIZE UINT64_C(5376)
#define H3_EXPECTED_LAYERS UINT64_C(50)
#define H3_EXPECTED_REFINER_LAYERS UINT64_C(2)
#define H3_EXPECTED_ATTENTION_HEADS UINT64_C(56)
#define H3_EXPECTED_ATTENTION_HEAD_DIM UINT64_C(128)
#define H3_EXPECTED_FFN_SIZE UINT64_C(14336)
#define H3_EXPECTED_VIDEO_CHANNELS UINT64_C(24)
#define H3_EXPECTED_AUDIO_CHANNELS UINT64_C(32)
#define H3_EXPECTED_TEXT_DIM UINT64_C(5120)
#define H3_EXPECTED_TIMESTEP_INPUT_DIM UINT64_C(256)
#define H3_EXPECTED_TIME_HIDDEN_SIZE UINT64_C(5376)
#define H3_EXPECTED_TIME_EMBED_DIM UINT64_C(2688)
#define H3_EXPECTED_ROPE_FREQS UINT64_C(16)
#define H3_VIDEO_PATCH_AREA UINT64_C(4)
#define H3_MAX_DETECTED_BLOCK_INDEX 4095U

static void set_error(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    (void)vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int qualified_name(char *output, size_t output_size,
                          const char *prefix, const char *suffix) {
    if (!output || !output_size || !suffix || !*suffix) return 0;
    prefix = prefix ? prefix : "";
    size_t prefix_length = strlen(prefix);
    int written;
    if (!prefix_length) {
        written = snprintf(output, output_size, "%s", suffix);
    } else if (prefix[prefix_length - 1] == '.') {
        written = snprintf(output, output_size, "%s%s", prefix, suffix);
    } else {
        written = snprintf(output, output_size, "%s.%s", prefix, suffix);
    }
    return written >= 0 && (size_t)written < output_size;
}

static const h3cspeed_model_tensor_info *find_unique_tensor(
        const h3cspeed_model_tensor_info *tensors, size_t tensor_count,
        const char *name, char *error, size_t error_size) {
    const h3cspeed_model_tensor_info *found = NULL;
    for (size_t index = 0; index < tensor_count; index++) {
        if (!tensors[index].name || strcmp(tensors[index].name, name) != 0)
            continue;
        if (found) {
            set_error(error, error_size, "duplicate tensor metadata: %s", name);
            return NULL;
        }
        found = &tensors[index];
    }
    if (!found) set_error(error, error_size, "required tensor is absent: %s", name);
    return found;
}

static int require_rank(const h3cspeed_model_tensor_info *tensor,
                        int rank, char *error, size_t error_size) {
    if (!tensor) return 0;
    if (tensor->ndim != rank || rank < 1 || rank > H3CSPEED_MODEL_MAX_DIMS) {
        set_error(error, error_size, "tensor %s has rank %d, expected %d",
                  tensor->name ? tensor->name : "(unnamed)", tensor->ndim, rank);
        return 0;
    }
    for (int dimension = 0; dimension < rank; dimension++) {
        if (!tensor->shape[dimension]) {
            set_error(error, error_size,
                      "tensor %s has an empty dimension at index %d",
                      tensor->name ? tensor->name : "(unnamed)", dimension);
            return 0;
        }
    }
    return 1;
}

static int parse_block_index(const char *name, const char *base,
                             unsigned *index) {
    size_t base_length = strlen(base);
    if (strncmp(name, base, base_length) != 0) return 0;
    const char *cursor = name + base_length;
    if (*cursor < '0' || *cursor > '9') return 0;
    unsigned value = 0;
    while (*cursor >= '0' && *cursor <= '9') {
        unsigned digit = (unsigned)(*cursor - '0');
        if (value > (H3_MAX_DETECTED_BLOCK_INDEX - digit) / 10U) return -1;
        value = value * 10U + digit;
        cursor++;
    }
    if (*cursor != '.') return 0;
    *index = value;
    return 1;
}

static int count_contiguous_blocks(
        const h3cspeed_model_tensor_info *tensors, size_t tensor_count,
        const char *prefix, const char *block_suffix, int required,
        uint64_t *count, char *error, size_t error_size) {
    char base[512];
    if (!qualified_name(base, sizeof(base), prefix, block_suffix)) {
        set_error(error, error_size, "model prefix is too long");
        return 0;
    }

    unsigned char *seen = NULL;
    size_t capacity = 0;
    unsigned maximum = 0;
    int any = 0;
    for (size_t tensor_index = 0; tensor_index < tensor_count; tensor_index++) {
        const char *name = tensors[tensor_index].name;
        if (!name) continue;
        unsigned block_index = 0;
        int parsed = parse_block_index(name, base, &block_index);
        if (parsed < 0) {
            free(seen);
            set_error(error, error_size,
                      "block index exceeds the supported metadata bound: %s", name);
            return 0;
        }
        if (!parsed) continue;
        if ((size_t)block_index >= capacity) {
            size_t next = capacity ? capacity : 16;
            while (next <= (size_t)block_index) next *= 2;
            unsigned char *grown = (unsigned char *)realloc(seen, next);
            if (!grown) {
                free(seen);
                set_error(error, error_size,
                          "out of memory while indexing model blocks");
                return 0;
            }
            memset(grown + capacity, 0, next - capacity);
            seen = grown;
            capacity = next;
        }
        seen[block_index] = 1;
        if (!any || block_index > maximum) maximum = block_index;
        any = 1;
    }

    if (!any) {
        free(seen);
        if (required) {
            set_error(error, error_size, "no tensors found below %s", base);
            return 0;
        }
        *count = 0;
        return 1;
    }
    for (unsigned block_index = 0; block_index <= maximum; block_index++) {
        if (!seen[block_index]) {
            free(seen);
            set_error(error, error_size,
                      "model block indices below %s are not contiguous; missing %u",
                      base, block_index);
            return 0;
        }
    }
    free(seen);
    *count = (uint64_t)maximum + 1;
    return 1;
}

static const h3cspeed_model_tensor_info *required_tensor(
        const h3cspeed_model_tensor_info *tensors, size_t tensor_count,
        const char *prefix, const char *suffix,
        char *error, size_t error_size) {
    char name[512];
    if (!qualified_name(name, sizeof(name), prefix, suffix)) {
        set_error(error, error_size, "model prefix is too long");
        return NULL;
    }
    return find_unique_tensor(tensors, tensor_count, name, error, error_size);
}

static const h3cspeed_model_tensor_info *optional_tensor(
        const h3cspeed_model_tensor_info *tensors, size_t tensor_count,
        const char *prefix, const char *suffix,
        int *duplicate, char *error, size_t error_size) {
    char name[512];
    if (duplicate) *duplicate = 0;
    if (!qualified_name(name, sizeof(name), prefix, suffix)) {
        set_error(error, error_size, "model prefix is too long");
        if (duplicate) *duplicate = 1;
        return NULL;
    }
    const h3cspeed_model_tensor_info *found = NULL;
    for (size_t index = 0; index < tensor_count; index++) {
        if (!tensors[index].name || strcmp(tensors[index].name, name) != 0)
            continue;
        if (found) {
            set_error(error, error_size, "duplicate tensor metadata: %s", name);
            if (duplicate) *duplicate = 1;
            return NULL;
        }
        found = &tensors[index];
    }
    return found;
}

int h3cspeed_h3_model_config_detect(
        const h3cspeed_model_tensor_info *tensors, size_t tensor_count,
        const char *prefix, h3cspeed_h3_model_config *config,
        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!tensors || !tensor_count || !config) {
        set_error(error, error_size, "tensor metadata and output config are required");
        return 0;
    }

    h3cspeed_h3_model_config detected;
    memset(&detected, 0, sizeof(detected));
    /* Match stable-diffusion.cpp's defaults for fields that are not used by
     * the AdaLN curve-table variant. Required architecture anchors below are
     * still validated instead of silently accepting missing metadata. */
    detected.hidden_size = H3_EXPECTED_HIDDEN_SIZE;
    detected.num_layers = H3_EXPECTED_LAYERS;
    detected.token_refiner_num_layers = H3_EXPECTED_REFINER_LAYERS;
    detected.num_attention_heads = H3_EXPECTED_ATTENTION_HEADS;
    detected.attention_head_dim = H3_EXPECTED_ATTENTION_HEAD_DIM;
    detected.ffn_hidden_size = H3_EXPECTED_FFN_SIZE;
    detected.video_latent_channels = H3_EXPECTED_VIDEO_CHANNELS;
    detected.audio_latent_channels = H3_EXPECTED_AUDIO_CHANNELS;
    detected.text_dim = H3_EXPECTED_TEXT_DIM;
    detected.timestep_input_dim = H3_EXPECTED_TIMESTEP_INPUT_DIM;
    detected.time_embed_hidden_size = H3_EXPECTED_TIME_HIDDEN_SIZE;
    detected.time_embed_dim = H3_EXPECTED_TIME_EMBED_DIM;
    detected.rope_inv_freq_len = H3_EXPECTED_ROPE_FREQS;

    const h3cspeed_model_tensor_info *video_patch = required_tensor(
        tensors, tensor_count, prefix, "video_patch_proj.weight", error, error_size);
    const h3cspeed_model_tensor_info *audio_patch = required_tensor(
        tensors, tensor_count, prefix, "audio_patch_proj.weight", error, error_size);
    const h3cspeed_model_tensor_info *q_norm = required_tensor(
        tensors, tensor_count, prefix, "blocks.0.attn.q_norm.weight", error, error_size);
    const h3cspeed_model_tensor_info *qkv = required_tensor(
        tensors, tensor_count, prefix, "blocks.0.attn.qkv_proj.weight", error, error_size);
    const h3cspeed_model_tensor_info *fc1 = required_tensor(
        tensors, tensor_count, prefix, "blocks.0.mlp.fc1.weight", error, error_size);
    const h3cspeed_model_tensor_info *condition = required_tensor(
        tensors, tensor_count, prefix, "condition_proj.weight", error, error_size);
    const h3cspeed_model_tensor_info *rope = required_tensor(
        tensors, tensor_count, prefix, "rope.inv_freq", error, error_size);
    if (!video_patch || !audio_patch || !q_norm || !qkv || !fc1 ||
        !condition || !rope) return 0;

    if (!require_rank(video_patch, 2, error, error_size) ||
        video_patch->shape[1] % H3_VIDEO_PATCH_AREA != 0) {
        if (error && error_size && !error[0])
            set_error(error, error_size,
                      "tensor %s input width is not divisible by the 2x2 patch area",
                      video_patch->name);
        return 0;
    }
    detected.hidden_size = video_patch->shape[0];
    detected.video_latent_channels = video_patch->shape[1] / H3_VIDEO_PATCH_AREA;

    if (!require_rank(audio_patch, 2, error, error_size)) return 0;
    if (audio_patch->shape[0] != detected.hidden_size) {
        set_error(error, error_size,
                  "tensor %s output width does not match video hidden size",
                  audio_patch->name);
        return 0;
    }
    detected.audio_latent_channels = audio_patch->shape[1];

    if (!require_rank(q_norm, 1, error, error_size)) return 0;
    detected.attention_head_dim = q_norm->shape[0];

    if (!require_rank(qkv, 2, error, error_size)) return 0;
    if (qkv->shape[1] != detected.hidden_size) {
        set_error(error, error_size,
                  "tensor %s input width does not match hidden size", qkv->name);
        return 0;
    }
    if (detected.attention_head_dim > UINT64_MAX / UINT64_C(3)) {
        set_error(error, error_size, "attention head dimension overflows");
        return 0;
    }
    uint64_t qkv_group = UINT64_C(3) * detected.attention_head_dim;
    if (!qkv_group || qkv->shape[0] % qkv_group != 0) {
        set_error(error, error_size,
                  "tensor %s output width is not divisible by 3 * head_dim",
                  qkv->name);
        return 0;
    }
    detected.num_attention_heads = qkv->shape[0] / qkv_group;

    if (!require_rank(fc1, 2, error, error_size)) return 0;
    if (fc1->shape[1] != detected.hidden_size || fc1->shape[0] % 2 != 0) {
        set_error(error, error_size,
                  "tensor %s is not a coherent fused SwiGLU projection", fc1->name);
        return 0;
    }
    detected.ffn_hidden_size = fc1->shape[0] / 2;

    if (!require_rank(condition, 2, error, error_size)) return 0;
    if (condition->shape[0] != detected.hidden_size) {
        set_error(error, error_size,
                  "tensor %s output width does not match hidden size",
                  condition->name);
        return 0;
    }
    detected.text_dim = condition->shape[1];

    if (!require_rank(rope, 1, error, error_size)) return 0;
    detected.rope_inv_freq_len = rope->shape[0];

    if (!count_contiguous_blocks(tensors, tensor_count, prefix, "blocks.", 1,
                                 &detected.num_layers, error, error_size) ||
        !count_contiguous_blocks(tensors, tensor_count, prefix,
                                 "token_refiner.blocks.", 0,
                                 &detected.token_refiner_num_layers,
                                 error, error_size)) return 0;

    int duplicate = 0;
    const h3cspeed_model_tensor_info *adaln = optional_tensor(
        tensors, tensor_count, prefix, "adaln_t_table", &duplicate,
        error, error_size);
    if (duplicate) return 0;
    if (adaln) {
        if (!require_rank(adaln, 2, error, error_size)) return 0;
        detected.variant = H3CSPEED_H3_VARIANT_ADALN_CURVES;
        detected.adaln_curve_grid = adaln->shape[0];
        detected.time_embed_dim = adaln->shape[1];
    } else {
        const h3cspeed_model_tensor_info *proj_in = required_tensor(
            tensors, tensor_count, prefix, "time_embedder.proj_in.weight",
            error, error_size);
        const h3cspeed_model_tensor_info *proj_out = required_tensor(
            tensors, tensor_count, prefix, "time_embedder.proj_out.weight",
            error, error_size);
        if (!proj_in || !proj_out ||
            !require_rank(proj_in, 2, error, error_size) ||
            !require_rank(proj_out, 2, error, error_size)) return 0;
        if (proj_out->shape[1] != proj_in->shape[0]) {
            set_error(error, error_size,
                      "time embedder projections have inconsistent hidden widths");
            return 0;
        }
        detected.variant = H3CSPEED_H3_VARIANT_TIME_EMBEDDER;
        detected.timestep_input_dim = proj_in->shape[1];
        detected.time_embed_hidden_size = proj_in->shape[0];
        detected.time_embed_dim = proj_out->shape[0];
    }

    *config = detected;
    return 1;
}

