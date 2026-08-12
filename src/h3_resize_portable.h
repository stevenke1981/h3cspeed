#ifndef H3CSPEED_RESIZE_PORTABLE_H
#define H3CSPEED_RESIZE_PORTABLE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Portable replacement for Apple's vImage high-quality RGB24 scaler.
 * The caller owns *output and releases it with free(). */
int h3cspeed_resize_rgb24_lanczos(const uint8_t *input, int frames,
                                 int input_width, int input_height,
                                 int output_width, int output_height,
                                 uint8_t **output);

#ifdef __cplusplus
}
#endif

#endif
