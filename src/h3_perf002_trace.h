#ifndef H3CSPEED_PERF002_TRACE_H
#define H3CSPEED_PERF002_TRACE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Private PERF-002 evidence hooks.  These are intentionally outside the
 * installed/public H3 API and are inert unless both trace paths are supplied
 * through the opt-in environment variables. */
int h3cspeed_perf002_trace_begin(int width, int height, int frames,
                                 int layers, int steps, uint64_t seed,
                                 const float *sigma_video,
                                 const float *sigma_audio);
void h3cspeed_perf002_trace_note_bf16_attention(int sage_hit,
                                                 int expected_native,
                                                 int unexpected_fallback);
void h3cspeed_perf002_trace_note_audio_euler_step(void);
int h3cspeed_perf002_trace_finish(int raw_audio_protocol_verified);
void h3cspeed_perf002_trace_abort(void);

#ifdef __cplusplus
}
#endif

#endif
