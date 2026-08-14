#include "h3_model_config.h"

#include <stdarg.h>
#include <stdio.h>

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

h3cspeed_h3_compatibility h3cspeed_h3_model_compatibility(
        const h3cspeed_h3_model_config *config) {
    if (!config) return UINT64_MAX;
    h3cspeed_h3_compatibility result = 0;
    if (config->variant != H3CSPEED_H3_VARIANT_TIME_EMBEDDER)
        result |= H3CSPEED_H3_INCOMPAT_VARIANT;
    if (config->hidden_size != H3_EXPECTED_HIDDEN_SIZE)
        result |= H3CSPEED_H3_INCOMPAT_HIDDEN_SIZE;
    if (config->num_layers != H3_EXPECTED_LAYERS)
        result |= H3CSPEED_H3_INCOMPAT_LAYER_COUNT;
    if (config->token_refiner_num_layers != H3_EXPECTED_REFINER_LAYERS)
        result |= H3CSPEED_H3_INCOMPAT_REFINER_LAYER_COUNT;
    if (config->num_attention_heads != H3_EXPECTED_ATTENTION_HEADS)
        result |= H3CSPEED_H3_INCOMPAT_ATTENTION_HEADS;
    if (config->attention_head_dim != H3_EXPECTED_ATTENTION_HEAD_DIM)
        result |= H3CSPEED_H3_INCOMPAT_ATTENTION_HEAD_DIM;
    if (config->ffn_hidden_size != H3_EXPECTED_FFN_SIZE)
        result |= H3CSPEED_H3_INCOMPAT_FFN_SIZE;
    if (config->video_latent_channels != H3_EXPECTED_VIDEO_CHANNELS)
        result |= H3CSPEED_H3_INCOMPAT_VIDEO_CHANNELS;
    if (config->audio_latent_channels != H3_EXPECTED_AUDIO_CHANNELS)
        result |= H3CSPEED_H3_INCOMPAT_AUDIO_CHANNELS;
    if (config->text_dim != H3_EXPECTED_TEXT_DIM)
        result |= H3CSPEED_H3_INCOMPAT_TEXT_DIM;
    if (config->timestep_input_dim != H3_EXPECTED_TIMESTEP_INPUT_DIM)
        result |= H3CSPEED_H3_INCOMPAT_TIMESTEP_INPUT_DIM;
    if (config->time_embed_hidden_size != H3_EXPECTED_TIME_HIDDEN_SIZE)
        result |= H3CSPEED_H3_INCOMPAT_TIME_HIDDEN_SIZE;
    if (config->time_embed_dim != H3_EXPECTED_TIME_EMBED_DIM)
        result |= H3CSPEED_H3_INCOMPAT_TIME_EMBED_DIM;
    if (config->rope_inv_freq_len != H3_EXPECTED_ROPE_FREQS)
        result |= H3CSPEED_H3_INCOMPAT_ROPE_FREQS;
    return result;
}

const char *h3cspeed_h3_variant_name(h3cspeed_h3_variant variant) {
    switch (variant) {
        case H3CSPEED_H3_VARIANT_TIME_EMBEDDER: return "time-embedder";
        case H3CSPEED_H3_VARIANT_ADALN_CURVES: return "adaln-curves";
        default: return "unknown";
    }
}

static int append_text(char *output, size_t output_size, size_t *used,
                       const char *format, ...) {
    if (!output || !output_size || !used || *used >= output_size) return 0;
    va_list arguments;
    va_start(arguments, format);
    int written = vsnprintf(output + *used, output_size - *used,
                            format, arguments);
    va_end(arguments);
    if (written < 0 || (size_t)written >= output_size - *used) {
        output[output_size - 1] = '\0';
        return 0;
    }
    *used += (size_t)written;
    return 1;
}

#define APPEND_ISSUE(mask_value, flag, field, expected, label) do {             \
    if ((mask_value) & (flag)) {                                                \
        if (!append_text(output, output_size, &used, "%s%s=%llu (expected %llu)", \
                         used ? "; " : "", (label),                            \
                         (unsigned long long)(field),                            \
                         (unsigned long long)(expected))) return 0;              \
    }                                                                           \
} while (0)

int h3cspeed_h3_model_format_compatibility(
        const h3cspeed_h3_model_config *config,
        char *output, size_t output_size) {
    if (!config || !output || !output_size) return 0;
    output[0] = '\0';
    size_t used = 0;
    h3cspeed_h3_compatibility mask =
        h3cspeed_h3_model_compatibility(config);
    if (!mask) {
        return append_text(output, output_size, &used,
                           "compatible with current h3cspeed CUDA kernels");
    }
    if (mask & H3CSPEED_H3_INCOMPAT_VARIANT) {
        if (!append_text(output, output_size, &used,
                         "variant=%s (expected time-embedder)",
                         h3cspeed_h3_variant_name(config->variant))) return 0;
    }
    APPEND_ISSUE(mask, H3CSPEED_H3_INCOMPAT_HIDDEN_SIZE,
                 config->hidden_size, H3_EXPECTED_HIDDEN_SIZE, "hidden_size");
    APPEND_ISSUE(mask, H3CSPEED_H3_INCOMPAT_LAYER_COUNT,
                 config->num_layers, H3_EXPECTED_LAYERS, "layers");
    APPEND_ISSUE(mask, H3CSPEED_H3_INCOMPAT_REFINER_LAYER_COUNT,
                 config->token_refiner_num_layers,
                 H3_EXPECTED_REFINER_LAYERS, "token_refiner_layers");
    APPEND_ISSUE(mask, H3CSPEED_H3_INCOMPAT_ATTENTION_HEADS,
                 config->num_attention_heads,
                 H3_EXPECTED_ATTENTION_HEADS, "attention_heads");
    APPEND_ISSUE(mask, H3CSPEED_H3_INCOMPAT_ATTENTION_HEAD_DIM,
                 config->attention_head_dim,
                 H3_EXPECTED_ATTENTION_HEAD_DIM, "attention_head_dim");
    APPEND_ISSUE(mask, H3CSPEED_H3_INCOMPAT_FFN_SIZE,
                 config->ffn_hidden_size, H3_EXPECTED_FFN_SIZE, "ffn_hidden_size");
    APPEND_ISSUE(mask, H3CSPEED_H3_INCOMPAT_VIDEO_CHANNELS,
                 config->video_latent_channels,
                 H3_EXPECTED_VIDEO_CHANNELS, "video_latent_channels");
    APPEND_ISSUE(mask, H3CSPEED_H3_INCOMPAT_AUDIO_CHANNELS,
                 config->audio_latent_channels,
                 H3_EXPECTED_AUDIO_CHANNELS, "audio_latent_channels");
    APPEND_ISSUE(mask, H3CSPEED_H3_INCOMPAT_TEXT_DIM,
                 config->text_dim, H3_EXPECTED_TEXT_DIM, "text_dim");
    APPEND_ISSUE(mask, H3CSPEED_H3_INCOMPAT_TIMESTEP_INPUT_DIM,
                 config->timestep_input_dim,
                 H3_EXPECTED_TIMESTEP_INPUT_DIM, "timestep_input_dim");
    APPEND_ISSUE(mask, H3CSPEED_H3_INCOMPAT_TIME_HIDDEN_SIZE,
                 config->time_embed_hidden_size,
                 H3_EXPECTED_TIME_HIDDEN_SIZE, "time_embed_hidden_size");
    APPEND_ISSUE(mask, H3CSPEED_H3_INCOMPAT_TIME_EMBED_DIM,
                 config->time_embed_dim,
                 H3_EXPECTED_TIME_EMBED_DIM, "time_embed_dim");
    APPEND_ISSUE(mask, H3CSPEED_H3_INCOMPAT_ROPE_FREQS,
                 config->rope_inv_freq_len,
                 H3_EXPECTED_ROPE_FREQS, "rope_inv_freq_len");
    return 1;
}

#undef APPEND_ISSUE
