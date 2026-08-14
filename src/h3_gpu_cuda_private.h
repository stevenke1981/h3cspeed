#ifndef H3_GPU_CUDA_PRIVATE_H
#define H3_GPU_CUDA_PRIVATE_H

#include "h3_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Reserve device storage before the current command chain is enqueued.  This
 * avoids cudaMalloc's synchronization cost in the later upload helper.  The
 * reservation is deliberately not current-epoch pinned, but scheduler
 * ownership protects it from eviction until consume or explicit cancel.
 */
int h3cspeed_cuda_reserve_prefetch_weight(h3_gpu *gpu,
                                          h3_gpu_tensor *tensor);

/* Enqueue one offloadable weight upload on the private CUDA upload stream.
 * This is intentionally outside h3_gpu.h: callers must use it only from the
 * owning host enqueue thread, and the prefetched tensor is not pinned as
 * current-operation use.  The helper is a no-op when H3_CUDA_DIT_PREFETCH is
 * not exactly enabled by the caller's scheduling policy. */
int h3cspeed_cuda_prefetch_weight(h3_gpu *gpu, h3_gpu_tensor *tensor);

/* Prime a weight that belongs to the next current block.  Re-entering a
 * sampler window may find the same ready tensor still pinned by the active
 * operation; that case is an intentional no-op rather than a future-reserve
 * contract violation. */
int h3cspeed_cuda_prime_prefetch_weight(h3_gpu *gpu,
                                        h3_gpu_tensor *tensor);

/* Drop scheduler ownership after a failed/abandoned future-block batch.  This
 * does not free a valid resident tensor; it merely makes it reclaimable. */
void h3cspeed_cuda_cancel_prefetch_weight(h3_gpu *gpu,
                                          h3_gpu_tensor *tensor);

#ifdef __cplusplus
}
#endif

#endif
