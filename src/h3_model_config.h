#ifndef H3CSPEED_MODEL_CONFIG_H
#define H3CSPEED_MODEL_CONFIG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define H3CSPEED_MODEL_MAX_DIMS 8

/* Tensor metadata in safetensors/PyTorch shape order. For a linear weight,
 * shape[0] is output features and shape[1] is input features. No payload is
 * required; this module deliberately operates on headers only. */
typedef struct {
    const char *name;
    int ndim;
    uint64_t shape[H3CSPEED_MODEL_MAX_DIMS];
} h3cspeed_model_tensor_info;

typedef enum {
    H3CSPEED_H3_VARIANT_UNKNOWN = 0,
    H3CSPEED_H3_VARIANT_TIME_EMBEDDER,
    H3CSPEED_H3_VARIANT_ADALN_CURVES
} h3cspeed_h3_variant;

typedef struct {
    uint64_t hidden_size;
    uint64_t num_layers;
    uint64_t token_refiner_num_layers;
    uint64_t num_attention_heads;
    uint64_t attention_head_dim;
    uint64_t ffn_hidden_size;
    uint64_t video_latent_channels;
    uint64_t audio_latent_channels;
    uint64_t text_dim;
    uint64_t timestep_input_dim;
    uint64_t time_embed_hidden_size;
    uint64_t time_embed_dim;
    uint64_t rope_inv_freq_len;
    uint64_t adaln_curve_grid;
    h3cspeed_h3_variant variant;
} h3cspeed_h3_model_config;

typedef uint64_t h3cspeed_h3_compatibility;

enum {
    H3CSPEED_H3_INCOMPAT_VARIANT = UINT64_C(1) << 0,
    H3CSPEED_H3_INCOMPAT_HIDDEN_SIZE = UINT64_C(1) << 1,
    H3CSPEED_H3_INCOMPAT_LAYER_COUNT = UINT64_C(1) << 2,
    H3CSPEED_H3_INCOMPAT_REFINER_LAYER_COUNT = UINT64_C(1) << 3,
    H3CSPEED_H3_INCOMPAT_ATTENTION_HEADS = UINT64_C(1) << 4,
    H3CSPEED_H3_INCOMPAT_ATTENTION_HEAD_DIM = UINT64_C(1) << 5,
    H3CSPEED_H3_INCOMPAT_FFN_SIZE = UINT64_C(1) << 6,
    H3CSPEED_H3_INCOMPAT_VIDEO_CHANNELS = UINT64_C(1) << 7,
    H3CSPEED_H3_INCOMPAT_AUDIO_CHANNELS = UINT64_C(1) << 8,
    H3CSPEED_H3_INCOMPAT_TEXT_DIM = UINT64_C(1) << 9,
    H3CSPEED_H3_INCOMPAT_TIMESTEP_INPUT_DIM = UINT64_C(1) << 10,
    H3CSPEED_H3_INCOMPAT_TIME_HIDDEN_SIZE = UINT64_C(1) << 11,
    H3CSPEED_H3_INCOMPAT_TIME_EMBED_DIM = UINT64_C(1) << 12,
    H3CSPEED_H3_INCOMPAT_ROPE_FREQS = UINT64_C(1) << 13
};

/* Detect the H3 architecture from tensor names and shapes. Prefix is the
 * optional namespace before names such as video_patch_proj.weight; pass an
 * empty string for native h3.c checkpoints. Returns 1 on a coherent model and
 * 0 for missing, duplicate, sparse, or internally inconsistent metadata. */
int h3cspeed_h3_model_config_detect(
    const h3cspeed_model_tensor_info *tensors, size_t tensor_count,
    const char *prefix, h3cspeed_h3_model_config *config,
    char *error, size_t error_size);

/* Compare detected metadata with the CUDA execution contract implemented by
 * h3cspeed 0.2. A zero mask is directly executable by current kernels. */
h3cspeed_h3_compatibility h3cspeed_h3_model_compatibility(
    const h3cspeed_h3_model_config *config);

/* Render the compatibility mask as a stable, human-readable list. */
int h3cspeed_h3_model_format_compatibility(
    const h3cspeed_h3_model_config *config,
    char *output, size_t output_size);

const char *h3cspeed_h3_variant_name(h3cspeed_h3_variant variant);

#ifdef __cplusplus
}
#endif

#endif
