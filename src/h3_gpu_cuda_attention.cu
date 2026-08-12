#include "h3_cuda_common.cuh"

#include <algorithm>
#include <cmath>

__global__ static void sdpa_online_kernel(
        void *output, h3_gpu_dtype output_dtype,
        const void *query, h3_gpu_dtype query_dtype,
        const void *key, h3_gpu_dtype key_dtype,
        const void *value, h3_gpu_dtype value_dtype,
        uint32_t batch, uint32_t sequence, uint32_t query_heads,
        uint32_t kv_heads, uint32_t head_dim, float scale,
        int causal, int output_head_major, int input_head_major,
        int scale_query_bf16) {
    uint32_t block = blockIdx.x;
    uint32_t row = block % sequence;
    uint32_t query_head = (block / sequence) % query_heads;
    uint32_t batch_index = block / (sequence * query_heads);
    if (batch_index >= batch) return;
    uint32_t kv_head = query_heads == kv_heads ? query_head :
        (uint32_t)(((uint64_t)query_head * kv_heads) / query_heads);
    if (kv_head >= kv_heads) kv_head = kv_heads - 1;

    extern __shared__ float shared[];
    float accumulator = 0.0f;
    float running_max = -INFINITY;
    float running_denominator = 0.0f;
    uint32_t key_limit = causal ? row + 1 : sequence;
    uint32_t dimension = threadIdx.x;
    size_t query_base = input_head_major ?
        (((size_t)batch_index * query_heads + query_head) * sequence + row) *
            head_dim :
        (((size_t)batch_index * sequence + row) * query_heads + query_head) *
            head_dim;

    /* Online softmax: each key contributes one Q.K reduction. The running
     * maximum rescales both the prior value accumulator and denominator, so no
     * score/probability matrix and no second dot-product pass are required. */
    for (uint32_t key_row = 0; key_row < key_limit; key_row++) {
        size_t key_base = input_head_major ?
            (((size_t)batch_index * kv_heads + kv_head) * sequence + key_row) *
                head_dim :
            (((size_t)batch_index * sequence + key_row) * kv_heads + kv_head) *
                head_dim;
        float partial = 0.0f;
        for (uint32_t inner = threadIdx.x; inner < head_dim;
             inner += blockDim.x) {
            float query_value = h3cspeed_device_load(query, query_dtype,
                                                     query_base + inner);
            if (scale_query_bf16)
                query_value = __bfloat162float(__float2bfloat16_rn(
                    query_value * scale));
            partial += query_value *
                       h3cspeed_device_load(key, key_dtype,
                                             key_base + inner);
        }
        shared[threadIdx.x] = partial;
        __syncthreads();
        for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
            if (threadIdx.x < stride)
                shared[threadIdx.x] += shared[threadIdx.x + stride];
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            float score = scale_query_bf16 ? shared[0] : shared[0] * scale;
            float new_max = fmaxf(running_max, score);
            float alpha = running_denominator > 0.0f ?
                expf(running_max - new_max) : 0.0f;
            float beta = expf(score - new_max);
            running_denominator = running_denominator * alpha + beta;
            running_max = new_max;
            shared[0] = alpha;
            shared[1] = beta;
            shared[2] = running_denominator;
        }
        __syncthreads();

        if (dimension < head_dim) {
            size_t value_base = input_head_major ?
                (((size_t)batch_index * kv_heads + kv_head) * sequence + key_row) *
                    head_dim :
                (((size_t)batch_index * sequence + key_row) * kv_heads + kv_head) *
                    head_dim;
            accumulator = accumulator * shared[0] + shared[1] *
                h3cspeed_device_load(value, value_dtype,
                                     value_base + dimension);
        }
        __syncthreads();
    }

    if (dimension < head_dim) {
        size_t destination;
        if (output_head_major) {
            destination = (((size_t)batch_index * query_heads + query_head) *
                           sequence + row) * head_dim + dimension;
        } else {
            destination = ((size_t)batch_index * sequence + row) *
                          query_heads * head_dim +
                          (size_t)query_head * head_dim + dimension;
        }
        float denominator = shared[2];
        h3cspeed_device_store(output, output_dtype, destination,
                              denominator > 0.0f ?
                              accumulator / denominator : 0.0f);
    }
}

int h3cspeed_sdpa(h3_gpu *gpu, h3_gpu_tensor *output,
                  const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                  const h3_gpu_tensor *value, uint32_t batch,
                  uint32_t sequence, uint32_t query_heads,
                  uint32_t kv_heads, uint32_t head_dim, float scale,
                  int causal, int output_head_major, int input_head_major,
                  int scale_query_bf16) {
    size_t query_elements = (size_t)batch * query_heads * sequence * head_dim;
    size_t kv_elements = (size_t)batch * kv_heads * sequence * head_dim;
    if (!gpu || !output || !query || !key || !value || !batch || !sequence ||
        !query_heads || !kv_heads || !head_dim || head_dim > 256 ||
        query->elements < query_elements || key->elements < kv_elements ||
        value->elements < kv_elements || output->elements < query_elements ||
        !h3cspeed_tensor_wait(gpu, query) || !h3cspeed_tensor_wait(gpu, key) ||
        !h3cspeed_tensor_wait(gpu, value)) return 0;
    unsigned threads = 1;
    while (threads < head_dim) threads <<= 1;
    threads = std::max(threads, 32u);
    uint64_t blocks64 = (uint64_t)batch * query_heads * sequence;
    if (blocks64 > UINT32_MAX) return 0;
    sdpa_online_kernel<<<(unsigned)blocks64, threads,
                          threads * sizeof(float), gpu->compute_stream>>>(
        output->data, output->dtype, query->data, query->dtype,
        key->data, key->dtype, value->data, value->dtype,
        batch, sequence, query_heads, kv_heads, head_dim, scale,
        causal, output_head_major, input_head_major, scale_query_bf16);
    h3cspeed_count_sdpa(gpu);
    return h3cspeed_launch_ok(gpu, "memory-bounded SDPA CUDA kernel");
}

extern "C" int h3_gpu_sdpa_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                 const h3_gpu_tensor *query,
                                 const h3_gpu_tensor *key,
                                 const h3_gpu_tensor *value,
                                 uint32_t sequence, uint32_t heads,
                                 uint32_t head_dim, float scale) {
    return h3cspeed_sdpa(gpu, output, query, key, value, 1, sequence,
                         heads, heads, head_dim, scale, 0, 0, 1, 0);
}

extern "C" int h3_gpu_sdpa_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                                  const h3_gpu_tensor *query,
                                  const h3_gpu_tensor *key,
                                  const h3_gpu_tensor *value,
                                  uint32_t sequence, uint32_t heads,
                                  uint32_t head_dim, float scale) {
    return h3cspeed_sdpa(gpu, output, query, key, value, 1, sequence,
                         heads, heads, head_dim, scale, 0, 0, 1, 0);
}

extern "C" int h3_gpu_sdpa_bf16_head_major_output(
        h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *query, const h3_gpu_tensor *key,
        const h3_gpu_tensor *value, uint32_t sequence, uint32_t heads,
        uint32_t head_dim, float scale) {
    return h3cspeed_sdpa(gpu, output, query, key, value, 1, sequence,
                         heads, heads, head_dim, scale, 0, 1, 1, 0);
}

extern "C" int h3_gpu_sdpa_causal_f32(
        h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *query, const h3_gpu_tensor *key,
        const h3_gpu_tensor *value, uint32_t batch, uint32_t sequence,
        uint32_t heads, uint32_t head_dim, float scale) {
    return h3cspeed_sdpa(gpu, output, query, key, value, batch, sequence,
                         heads, heads, head_dim, scale, 1, 0, 1, 0);
}

extern "C" int h3_gpu_gqa_causal_bf16(
        h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *query, const h3_gpu_tensor *key,
        const h3_gpu_tensor *value, uint32_t sequence,
        uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim,
        float scale) {
    return h3cspeed_sdpa(gpu, output, query, key, value, 1, sequence,
                         query_heads, kv_heads, head_dim, scale, 1, 0, 0, 1);
}

__global__ static void attention_pool_kernel(
        void *output, h3_gpu_dtype output_dtype,
        const void *attended, h3_gpu_dtype attended_dtype,
        uint32_t batch, uint32_t length, uint32_t heads,
        uint32_t head_dim, uint32_t output_dim) {
    size_t total = (size_t)batch * length * output_dim;
    uint32_t natural_width = heads * head_dim;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += (size_t)gridDim.x * blockDim.x) {
        uint32_t column = (uint32_t)(index % output_dim);
        size_t row_index = index / output_dim;
        uint32_t row = (uint32_t)(row_index % length);
        uint32_t batch_index = (uint32_t)(row_index / length);
        float value = 0.0f;
        if (column < natural_width) {
            uint32_t head = column / head_dim;
            uint32_t dimension = column % head_dim;
            size_t source = (((size_t)batch_index * heads + head) * length + row) *
                            head_dim + dimension;
            value = h3cspeed_device_load(attended, attended_dtype, source);
        }
        h3cspeed_device_store(output, output_dtype, index, value);
    }
}

extern "C" int h3_gpu_audio_attention_pool_f32(
        h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *attended, uint32_t batch, uint32_t length,
        uint32_t heads, uint32_t head_dim, uint32_t output_dim) {
    size_t output_elements = (size_t)batch * length * output_dim;
    size_t input_elements = (size_t)batch * length * heads * head_dim;
    if (!gpu || !output || !attended || output->elements < output_elements ||
        attended->elements < input_elements ||
        !h3cspeed_tensor_wait(gpu, attended)) return 0;
    attention_pool_kernel<<<h3cspeed_blocks(output_elements), 256, 0,
                             gpu->compute_stream>>>(
        output->data, output->dtype, attended->data, attended->dtype,
        batch, length, heads, head_dim, output_dim);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "audio attention pool CUDA kernel");
}
