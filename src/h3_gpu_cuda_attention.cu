#include "h3_cuda_common.cuh"

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdlib>

/* Native SageAttention-compatible QK INT8 path.
 *
 * SageAttention 2 uses low-precision Q/K with an unchanged FP16/BF16 P*V
 * stage.  The upstream sm_86 implementation is a Python/Triton extension and
 * cannot be linked into this standalone C runtime, so this backend implements
 * the same capacity contract directly: BF16 Q/K are quantized per token,
 * INT8 dot products use DP4A, softmax is accumulated in FP32, and V/output
 * retain BF16 boundaries.  It is opt-in through H3_CUDA_ATTENTION=sage; the
 * existing exact BF16 online kernel remains the default. */

__global__ static void sage_quantize_rows_kernel(
        const void *input, h3_gpu_dtype input_dtype, int8_t *quantized,
        float *scales, uint32_t batch, uint32_t sequence, uint32_t heads,
        uint32_t head_dim, int input_head_major, float input_scale,
        int scale_bf16) {
    uint32_t canonical_row = blockIdx.x;
    uint32_t token = canonical_row % sequence;
    uint32_t head = (canonical_row / sequence) % heads;
    uint32_t batch_index = canonical_row / (sequence * heads);
    if (batch_index >= batch) return;
    size_t source_base = input_head_major ?
        (((size_t)batch_index * heads + head) * sequence + token) * head_dim :
        (((size_t)batch_index * sequence + token) * heads + head) * head_dim;
    size_t destination_base = (size_t)canonical_row * head_dim;
    extern __shared__ float reductions[];
    float maximum = 1.0e-7f;
    for (uint32_t dimension = threadIdx.x; dimension < head_dim;
         dimension += blockDim.x) {
        float value = h3cspeed_device_load(input, input_dtype,
                                           source_base + dimension);
        if (scale_bf16)
            value = __bfloat162float(__float2bfloat16_rn(value * input_scale));
        maximum = fmaxf(maximum, fabsf(value));
    }
    reductions[threadIdx.x] = maximum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            reductions[threadIdx.x] = fmaxf(
                reductions[threadIdx.x], reductions[threadIdx.x + stride]);
        __syncthreads();
    }
    float scale = reductions[0] / 127.0f;
    if (threadIdx.x == 0) scales[canonical_row] = scale;
    __syncthreads();
    float inverse = 1.0f / scale;
    for (uint32_t dimension = threadIdx.x; dimension < head_dim;
         dimension += blockDim.x) {
        float value = h3cspeed_device_load(input, input_dtype,
                                           source_base + dimension);
        if (scale_bf16)
            value = __bfloat162float(__float2bfloat16_rn(value * input_scale));
        int rounded = (int)nearbyintf(value * inverse);
        if (rounded < -127) rounded = -127;
        if (rounded > 127) rounded = 127;
        quantized[destination_base + dimension] = (int8_t)rounded;
    }
}

__global__ static void sage_sdpa_int8_kernel(
        void *output, h3_gpu_dtype output_dtype,
        const int8_t *query, const int8_t *key,
        const void *value, h3_gpu_dtype value_dtype,
        const float *query_scales, const float *key_scales,
        uint32_t batch, uint32_t sequence, uint32_t query_heads,
        uint32_t kv_heads, uint32_t head_dim, float score_scale,
        int causal, int output_head_major, int input_head_major) {
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
    float running_max = -FLT_MAX;
    float running_denominator = 0.0f;
    uint32_t key_limit = causal ? row + 1 : sequence;
    uint32_t dimension = threadIdx.x;
    size_t query_row = ((size_t)batch_index * query_heads + query_head) *
                       sequence + row;
    size_t query_base = query_row * head_dim;

    for (uint32_t key_row = 0; key_row < key_limit; key_row++) {
        size_t key_scale_row = ((size_t)batch_index * kv_heads + kv_head) *
                               sequence + key_row;
        size_t key_base = key_scale_row * head_dim;
        int partial = 0;
        for (uint32_t inner = threadIdx.x * 4u; inner + 3u < head_dim;
             inner += blockDim.x * 4u) {
            int q4 = *reinterpret_cast<const int *>(query + query_base + inner);
            int k4 = *reinterpret_cast<const int *>(key + key_base + inner);
            partial = __dp4a(q4, k4, partial);
        }
        for (uint32_t inner = (head_dim & ~3u) + threadIdx.x;
             inner < head_dim; inner += blockDim.x) {
            partial += (int)query[query_base + inner] *
                       (int)key[key_base + inner];
        }
        shared[threadIdx.x] = (float)partial;
        __syncthreads();
        for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
            if (threadIdx.x < stride)
                shared[threadIdx.x] += shared[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            float score = shared[0] * query_scales[query_row] *
                          key_scales[key_scale_row] * score_scale;
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
        size_t destination = output_head_major ?
            (((size_t)batch_index * query_heads + query_head) * sequence + row) *
                head_dim + dimension :
            ((size_t)batch_index * sequence + row) * query_heads * head_dim +
                (size_t)query_head * head_dim + dimension;
        float denominator = shared[2];
        h3cspeed_device_store(output, output_dtype, destination,
                              denominator > 0.0f ?
                              accumulator / denominator : 0.0f);
    }
}

static size_t sage_align(size_t value) {
    return (value + 255u) & ~(size_t)255u;
}

static int h3cspeed_sage_sdpa(
        h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *query, const h3_gpu_tensor *key,
        const h3_gpu_tensor *value, uint32_t batch, uint32_t sequence,
        uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim,
        float scale, int causal, int output_head_major,
        int input_head_major, int scale_query_bf16) {
    if (gpu->properties.major < 8 || query->dtype != H3_GPU_BF16 ||
        key->dtype != H3_GPU_BF16 || value->dtype != H3_GPU_BF16 ||
        output->dtype != H3_GPU_BF16 || head_dim < 4 || head_dim > 128 ||
        head_dim % 4u != 0 ||
        query_heads % kv_heads != 0) {
        h3cspeed_set_error(gpu, "SageAttention",
            "requires sm_80+, BF16 Q/K/V/output, head_dim 4..128 divisible by 4, and divisible GQA heads");
        return 0;
    }
    uint64_t query_rows64 = (uint64_t)batch * query_heads;
    uint64_t key_rows64 = (uint64_t)batch * kv_heads;
    if (sequence > 0 &&
        (query_rows64 > UINT64_MAX / sequence ||
         key_rows64 > UINT64_MAX / sequence)) {
        h3cspeed_set_error(gpu, "SageAttention", "shape overflows row count");
        return 0;
    }
    query_rows64 *= sequence;
    key_rows64 *= sequence;
    if (query_rows64 > INT_MAX || key_rows64 > INT_MAX ||
        query_rows64 > SIZE_MAX / head_dim || key_rows64 > SIZE_MAX / head_dim) {
        h3cspeed_set_error(gpu, "SageAttention", "shape is too large");
        return 0;
    }
    size_t query_rows = (size_t)query_rows64;
    size_t key_rows = (size_t)key_rows64;
    size_t query_bytes = query_rows * head_dim;
    size_t key_bytes = key_rows * head_dim;
    size_t query_scale_bytes = query_rows * sizeof(float);
    size_t key_scale_bytes = key_rows * sizeof(float);
    size_t key_offset = sage_align(query_bytes);
    size_t query_scale_offset = sage_align(key_offset + key_bytes);
    size_t key_scale_offset = sage_align(query_scale_offset + query_scale_bytes);
    size_t total = sage_align(key_scale_offset + key_scale_bytes);
    unsigned char *scratch = static_cast<unsigned char *>(
        h3cspeed_scratch_reserve(gpu, total));
    if (!scratch) return 0;
    int8_t *quantized_query = reinterpret_cast<int8_t *>(scratch);
    int8_t *quantized_key = reinterpret_cast<int8_t *>(scratch + key_offset);
    float *query_scales = reinterpret_cast<float *>(scratch + query_scale_offset);
    float *key_scales = reinterpret_cast<float *>(scratch + key_scale_offset);
    unsigned threads = 32;
    while (threads < head_dim) threads <<= 1;
    sage_quantize_rows_kernel<<<(unsigned)query_rows, threads,
        threads * sizeof(float), gpu->compute_stream>>>(
            query->data, query->dtype, quantized_query, query_scales,
            batch, sequence, query_heads, head_dim, input_head_major,
            scale, scale_query_bf16);
    sage_quantize_rows_kernel<<<(unsigned)key_rows, threads,
        threads * sizeof(float), gpu->compute_stream>>>(
            key->data, key->dtype, quantized_key, key_scales,
            batch, sequence, kv_heads, head_dim, input_head_major, 1.0f, 0);
    sage_sdpa_int8_kernel<<<(unsigned)query_rows, threads,
        threads * sizeof(float), gpu->compute_stream>>>(
            output->data, output->dtype, quantized_query, quantized_key,
            value->data, value->dtype, query_scales, key_scales,
            batch, sequence, query_heads, kv_heads, head_dim,
            scale_query_bf16 ? 1.0f : scale, causal,
            output_head_major, input_head_major);
    h3cspeed_count_sdpa(gpu);
    return h3cspeed_launch_ok(gpu, "SageAttention-compatible INT8 QK kernel");
}

static int h3cspeed_sage_eligible(
        const h3_gpu *gpu, const h3_gpu_tensor *output,
        const h3_gpu_tensor *query, const h3_gpu_tensor *key,
        const h3_gpu_tensor *value, uint32_t query_heads,
        uint32_t kv_heads, uint32_t head_dim) {
    return gpu->properties.major >= 8 && query->dtype == H3_GPU_BF16 &&
        key->dtype == H3_GPU_BF16 && value->dtype == H3_GPU_BF16 &&
        output->dtype == H3_GPU_BF16 && head_dim >= 4 && head_dim <= 128 &&
        head_dim % 4u == 0 && query_heads % kv_heads == 0;
}

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
    float running_max = -FLT_MAX;
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
    const char *backend = getenv("H3_CUDA_ATTENTION");
    int bf16_scope = query->dtype == H3_GPU_BF16 &&
        key->dtype == H3_GPU_BF16 && value->dtype == H3_GPU_BF16 &&
        output->dtype == H3_GPU_BF16;
    if (backend && *backend && strcmp(backend, "native") != 0) {
        if (strcmp(backend, "sage") != 0) {
            h3cspeed_set_error(gpu, "H3_CUDA_ATTENTION",
                               "expected native or sage");
            return 0;
        }
        /* Sage quantization is defined only for the released BF16 DP4A
         * shapes. VAE and other F32 attention remains on the existing CUDA
         * kernel; this is a GPU-to-GPU dispatch choice, never a CPU fallback. */
        if (h3cspeed_sage_eligible(
                gpu, output, query, key, value, query_heads, kv_heads,
                head_dim)) {
            int sage_ok = h3cspeed_sage_sdpa(
                gpu, output, query, key, value, batch, sequence,
                query_heads, kv_heads, head_dim, scale, causal,
                output_head_major, input_head_major, scale_query_bf16);
            if (sage_ok && bf16_scope)
                h3cspeed_perf002_trace_note_bf16_attention(1, 0, 0);
            return sage_ok;
        }
        if (bf16_scope)
            h3cspeed_perf002_trace_note_bf16_attention(0, 0, 1);
    } else if (bf16_scope) {
        /* A trace-enabled native run records the deliberate native calls as
         * expected-native, while F32 VAE attention never enters this branch. */
        h3cspeed_perf002_trace_note_bf16_attention(0, 1, 0);
    }
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
