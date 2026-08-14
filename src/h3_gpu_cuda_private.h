#ifndef H3_GPU_CUDA_PRIVATE_H
#define H3_GPU_CUDA_PRIVATE_H

#include "h3_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Reserve device storage before the current command chain is enqueued.  This
 * avoids cudaMalloc's synchronization cost in the later upload helper.  The
 * reservation is deliberately not pinned and remains reclaimable.
 */
int h3cspeed_cuda_reserve_prefetch_weight(h3_gpu *gpu,
                                          h3_gpu_tensor *tensor);

/* Enqueue one offloadable weight upload on the private CUDA upload stream.
 * This is intentionally outside h3_gpu.h: callers must use it only from the
 * owning host enqueue thread, and the prefetched tensor is not pinned as
 * current-operation use.  The helper is a no-op when H3_CUDA_DIT_PREFETCH is
 * not exactly enabled by the caller's scheduling policy. */
int h3cspeed_cuda_prefetch_weight(h3_gpu *gpu, h3_gpu_tensor *tensor);

#ifdef __cplusplus
}
#endif

#endif
