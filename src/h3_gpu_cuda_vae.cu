#include "h3_cuda_common.cuh"

#include <algorithm>
#include <climits>
#include <cmath>

__global__ static void conv1d_kernel(
        void *output, h3_gpu_dtype output_dtype,
        const void *input, h3_gpu_dtype input_dtype,
        const void *weight, h3_gpu_dtype weight_dtype,
        const void *bias, h3_gpu_dtype bias_dtype,
        uint32_t batch, uint32_t length, uint32_t input_channels,
        uint32_t output_channels, uint32_t kernel, uint32_t stride,
        uint32_t padding, uint32_t dilation, uint32_t output_length) {
    size_t total = (size_t)batch * output_length * output_channels;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += (size_t)gridDim.x * blockDim.x) {
        uint32_t channel = (uint32_t)(index % output_channels);
        size_t row_index = index / output_channels;
        uint32_t out_x = (uint32_t)(row_index % output_length);
        uint32_t batch_index = (uint32_t)(row_index / output_length);
        float accumulator = bias ? h3cspeed_device_load(bias, bias_dtype, channel) : 0.0f;
        for (uint32_t input_channel = 0; input_channel < input_channels;
             input_channel++) {
            for (uint32_t tap = 0; tap < kernel; tap++) {
                int64_t source = (int64_t)out_x * stride - padding +
                                 (int64_t)tap * dilation;
                if (source < 0 || source >= length) continue;
                size_t input_index = ((size_t)batch_index * length +
                                      (size_t)source) * input_channels + input_channel;
                size_t weight_index = ((size_t)channel * input_channels +
                                       input_channel) * kernel + tap;
                accumulator += h3cspeed_device_load(input, input_dtype, input_index) *
                               h3cspeed_device_load(weight, weight_dtype, weight_index);
            }
        }
        h3cspeed_device_store(output, output_dtype, index, accumulator);
    }
}

static int conv1d_dispatch(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *bias, uint32_t batch,
                           uint32_t length, uint32_t input_channels,
                           uint32_t output_channels, uint32_t kernel,
                           uint32_t stride, uint32_t padding,
                           uint32_t dilation) {
    if (!gpu || !output || !input || !weight || !batch || !length ||
        !input_channels || !output_channels || !kernel || !stride ||
        !dilation) return 0;
    int64_t numerator = (int64_t)length + 2 * (int64_t)padding -
                        (int64_t)dilation * (kernel - 1) - 1;
    if (numerator < 0) return 0;
    int64_t calculated_length = numerator / stride + 1;
    if (calculated_length < 1 || calculated_length > UINT32_MAX) return 0;
    uint32_t output_length = (uint32_t)calculated_length;
    size_t output_elements = 0;
    size_t input_elements = 0;
    size_t weight_elements = 0;
    if (!h3cspeed_size_mul3(batch, output_length, output_channels,
                            &output_elements) ||
        !h3cspeed_size_mul3(batch, length, input_channels, &input_elements) ||
        !h3cspeed_size_mul3(output_channels, input_channels, kernel,
                            &weight_elements) ||
        output->elements < output_elements || input->elements < input_elements ||
        weight->elements < weight_elements || (bias && bias->elements < output_channels) ||
        !h3cspeed_tensor_wait(gpu, input) || !h3cspeed_tensor_wait(gpu, weight) ||
        (bias && !h3cspeed_tensor_wait(gpu, bias))) return 0;
    conv1d_kernel<<<h3cspeed_blocks(output_elements), 256, 0,
                    gpu->compute_stream>>>(
        output->data, output->dtype, input->data, input->dtype,
        weight->data, weight->dtype,
        bias ? bias->data : nullptr, bias ? bias->dtype : H3_GPU_F32,
        batch, length, input_channels, output_channels, kernel, stride,
        padding, dilation, output_length);
    h3cspeed_count_conv(gpu);
    return h3cspeed_launch_ok(gpu, "Conv1d CUDA kernel");
}

extern "C" int h3_gpu_conv1d_f32(
        h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
        const h3_gpu_tensor *weight, const h3_gpu_tensor *bias,
        uint32_t batch, uint32_t length, uint32_t input_channels,
        uint32_t output_channels, uint32_t kernel, uint32_t padding,
        uint32_t dilation) {
    return conv1d_dispatch(gpu, output, input, weight, bias, batch, length,
                           input_channels, output_channels, kernel, 1,
                           padding, dilation);
}

extern "C" int h3_gpu_conv1d_stride_f32(
        h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
        const h3_gpu_tensor *weight, const h3_gpu_tensor *bias,
        uint32_t batch, uint32_t length, uint32_t input_channels,
        uint32_t output_channels, uint32_t kernel, uint32_t stride,
        uint32_t padding, uint32_t dilation) {
    return conv1d_dispatch(gpu, output, input, weight, bias, batch, length,
                           input_channels, output_channels, kernel, stride,
                           padding, dilation);
}

__global__ static void conv_transpose1d_kernel(
        void *output, h3_gpu_dtype output_dtype,
        const void *input, h3_gpu_dtype input_dtype,
        const void *weight, h3_gpu_dtype weight_dtype,
        const void *bias, h3_gpu_dtype bias_dtype,
        uint32_t batch, uint32_t length, uint32_t input_channels,
        uint32_t output_channels, uint32_t kernel, uint32_t stride,
        uint32_t padding, uint32_t output_length) {
    size_t total = (size_t)batch * output_length * output_channels;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += (size_t)gridDim.x * blockDim.x) {
        uint32_t output_channel = (uint32_t)(index % output_channels);
        size_t row_index = index / output_channels;
        uint32_t out_x = (uint32_t)(row_index % output_length);
        uint32_t batch_index = (uint32_t)(row_index / output_length);
        float accumulator = bias ?
            h3cspeed_device_load(bias, bias_dtype, output_channel) : 0.0f;
        for (uint32_t input_channel = 0; input_channel < input_channels;
             input_channel++) {
            for (uint32_t tap = 0; tap < kernel; tap++) {
                int64_t numerator = (int64_t)out_x + padding - tap;
                if (numerator < 0 || numerator % stride) continue;
                int64_t source = numerator / stride;
                if (source < 0 || source >= length) continue;
                size_t input_index = ((size_t)batch_index * length +
                                      (size_t)source) * input_channels + input_channel;
                /* PyTorch ConvTranspose1d weight order: [in, out, kernel]. */
                size_t weight_index = ((size_t)input_channel * output_channels +
                                       output_channel) * kernel + tap;
                accumulator += h3cspeed_device_load(input, input_dtype, input_index) *
                               h3cspeed_device_load(weight, weight_dtype, weight_index);
            }
        }
        h3cspeed_device_store(output, output_dtype, index, accumulator);
    }
}

extern "C" int h3_gpu_conv_transpose1d_f32(
        h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
        const h3_gpu_tensor *weight, const h3_gpu_tensor *bias,
        uint32_t batch, uint32_t length, uint32_t input_channels,
        uint32_t output_channels, uint32_t kernel, uint32_t stride,
        uint32_t padding) {
    if (!gpu || !output || !input || !weight || !batch || !length ||
        !input_channels || !output_channels || !kernel || !stride) return 0;
    int64_t calculated = ((int64_t)length - 1) * stride -
                         2 * (int64_t)padding + kernel;
    if (calculated < 1 || calculated > UINT32_MAX) return 0;
    uint32_t output_length = (uint32_t)calculated;
    size_t output_elements = 0;
    size_t input_elements = 0;
    size_t weight_elements = 0;
    if (!h3cspeed_size_mul3(batch, output_length, output_channels,
                            &output_elements) ||
        !h3cspeed_size_mul3(batch, length, input_channels, &input_elements) ||
        !h3cspeed_size_mul3(input_channels, output_channels, kernel,
                            &weight_elements) ||
        output->elements < output_elements ||
        input->elements < input_elements || weight->elements < weight_elements ||
        (bias && bias->elements < output_channels) ||
        !h3cspeed_tensor_wait(gpu, input) || !h3cspeed_tensor_wait(gpu, weight) ||
        (bias && !h3cspeed_tensor_wait(gpu, bias))) return 0;
    conv_transpose1d_kernel<<<h3cspeed_blocks(output_elements), 256, 0,
                              gpu->compute_stream>>>(
        output->data, output->dtype, input->data, input->dtype,
        weight->data, weight->dtype,
        bias ? bias->data : nullptr, bias ? bias->dtype : H3_GPU_F32,
        batch, length, input_channels, output_channels, kernel, stride,
        padding, output_length);
    h3cspeed_count_conv(gpu);
    return h3cspeed_launch_ok(gpu, "ConvTranspose1d CUDA kernel");
}

__global__ static void weight_norm_kernel(
        void *output, h3_gpu_dtype output_dtype,
        const void *vector, h3_gpu_dtype vector_dtype,
        const void *magnitude, h3_gpu_dtype magnitude_dtype,
        uint32_t outer, uint32_t inner) {
    uint32_t row = blockIdx.x;
    if (row >= outer) return;
    extern __shared__ float shared[];
    float square = 0.0f;
    for (uint32_t column = threadIdx.x; column < inner; column += blockDim.x) {
        float value = h3cspeed_device_load(vector, vector_dtype,
                                            (size_t)row * inner + column);
        square += value * value;
    }
    shared[threadIdx.x] = square;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = h3cspeed_device_load(magnitude, magnitude_dtype, row) /
                  sqrtf(shared[0] + 1e-12f);
    for (uint32_t column = threadIdx.x; column < inner; column += blockDim.x) {
        float value = h3cspeed_device_load(vector, vector_dtype,
                                            (size_t)row * inner + column) * scale;
        h3cspeed_device_store(output, output_dtype,
                              (size_t)row * inner + column, value);
    }
}

extern "C" int h3_gpu_weight_norm_f32(
        h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *vector,
        const h3_gpu_tensor *magnitude, uint32_t outer, uint32_t inner) {
    size_t elements = 0;
    if (!gpu || !output || !vector || !magnitude || !outer || !inner ||
        !h3cspeed_size_mul(outer, inner, &elements) ||
        output->elements < elements ||
        vector->elements < elements || magnitude->elements < outer ||
        !h3cspeed_tensor_wait(gpu, vector) ||
        !h3cspeed_tensor_wait(gpu, magnitude)) return 0;
    weight_norm_kernel<<<outer, 256, 256 * sizeof(float), gpu->compute_stream>>>(
        output->data, output->dtype, vector->data, vector->dtype,
        magnitude->data, magnitude->dtype, outer, inner);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "weight norm CUDA kernel");
}

__global__ static void snake_kernel(
        void *output, h3_gpu_dtype output_dtype,
        const void *input, h3_gpu_dtype input_dtype,
        const void *alpha, h3_gpu_dtype alpha_dtype,
        const void *beta, h3_gpu_dtype beta_dtype,
        uint32_t batch, uint32_t length, uint32_t channels,
        int logarithmic, int separate_beta) {
    size_t total = (size_t)batch * length * channels;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += (size_t)gridDim.x * blockDim.x) {
        uint32_t channel = (uint32_t)(index % channels);
        float a = h3cspeed_device_load(alpha, alpha_dtype, channel);
        float b = separate_beta ? h3cspeed_device_load(beta, beta_dtype, channel) : a;
        if (logarithmic) {
            a = expf(a);
            b = expf(b);
        }
        a = fmaxf(a, 1e-6f);
        b = fmaxf(b, 1e-6f);
        float value = h3cspeed_device_load(input, input_dtype, index);
        float sine = sinf(a * value);
        value += (sine * sine) / b;
        h3cspeed_device_store(output, output_dtype, index, value);
    }
}

extern "C" int h3_gpu_snake1d_f32(
        h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
        const h3_gpu_tensor *alpha, uint32_t batch, uint32_t length,
        uint32_t channels) {
    size_t elements = 0;
    if (!gpu || !output || !input || !alpha || !batch || !length ||
        !channels || !h3cspeed_size_mul3(batch, length, channels, &elements) ||
        output->elements < elements ||
        input->elements < elements || alpha->elements < channels ||
        !h3cspeed_tensor_wait(gpu, input) || !h3cspeed_tensor_wait(gpu, alpha)) return 0;
    snake_kernel<<<h3cspeed_blocks(elements), 256, 0, gpu->compute_stream>>>(
        output->data, output->dtype, input->data, input->dtype,
        alpha->data, alpha->dtype, nullptr, H3_GPU_F32,
        batch, length, channels, 0, 0);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "Snake1d CUDA kernel");
}

/* Exact CUDA translation of the released Metal implementation. BigVGAN's
 * fixed 2x upsample -> SnakeBeta -> 2x low-pass downsample is evaluated as a
 * fused polyphase FIR at the original sample rate, avoiding a doubled
 * intermediate waveform for every activation. */
__global__ static void alias_free_snake_kernel(
        float *output, const float *input, const float *alpha_log,
        const float *beta_log, const float *upsample_filter,
        const float *downsample_filter, uint32_t batch, uint32_t length,
        uint32_t channels) {
    size_t total = (size_t)batch * length * channels;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += (size_t)gridDim.x * blockDim.x) {
        uint32_t channel = (uint32_t)(index % channels);
        size_t row = index / channels;
        uint32_t time = (uint32_t)(row % length);
        uint32_t batch_index = (uint32_t)(row / length);
        float alpha = expf(alpha_log[channel]);
        float beta = expf(beta_log[channel]);
        float result = 0.0f;
        for (int down_k = 0; down_k < 12; down_k++) {
            int up_time = (int)(time * 2u) + down_k - 5;
            if (up_time < 0) up_time = 0;
            int maximum_up_time = (int)(length * 2u) - 1;
            if (up_time > maximum_up_time) up_time = maximum_up_time;
            int raw_time = up_time + 15;
            float upsampled = 0.0f;
            for (int up_k = 0; up_k < 12; up_k++) {
                int numerator = raw_time - up_k;
                if (numerator < 0 || (numerator & 1)) continue;
                int padded_time = numerator / 2;
                int source_time = padded_time - 5;
                if (source_time < 0) source_time = 0;
                int maximum_source_time = (int)length - 1;
                if (source_time > maximum_source_time)
                    source_time = maximum_source_time;
                size_t source = ((size_t)batch_index * length +
                                 (uint32_t)source_time) * channels + channel;
                upsampled = fmaf(input[source],
                                 2.0f * upsample_filter[up_k], upsampled);
            }
            float sine = sinf(alpha * upsampled);
            float activated = upsampled + sine * sine / (beta + 1e-9f);
            result = fmaf(activated, downsample_filter[down_k], result);
        }
        output[index] = result;
    }
}

extern "C" int h3_gpu_alias_free_snake_f32(
        h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
        const h3_gpu_tensor *alpha_log, const h3_gpu_tensor *beta_log,
        const h3_gpu_tensor *upsample_filter,
        const h3_gpu_tensor *downsample_filter, uint32_t batch,
        uint32_t length, uint32_t channels) {
    size_t elements = 0;
    if (!gpu || !output || !input || !alpha_log || !beta_log ||
        !upsample_filter || !downsample_filter || !batch || !length ||
        !channels || length > (uint32_t)(INT_MAX / 2) ||
        !h3cspeed_size_mul3(batch, length, channels, &elements) ||
        output->dtype != H3_GPU_F32 || input->dtype != H3_GPU_F32 ||
        alpha_log->dtype != H3_GPU_F32 || beta_log->dtype != H3_GPU_F32 ||
        upsample_filter->dtype != H3_GPU_F32 ||
        downsample_filter->dtype != H3_GPU_F32 ||
        output->elements < elements || input->elements < elements ||
        alpha_log->elements < channels || beta_log->elements < channels ||
        upsample_filter->elements < 12 || downsample_filter->elements < 12 ||
        !h3cspeed_tensor_wait(gpu, input) ||
        !h3cspeed_tensor_wait(gpu, alpha_log) ||
        !h3cspeed_tensor_wait(gpu, beta_log) ||
        !h3cspeed_tensor_wait(gpu, upsample_filter) ||
        !h3cspeed_tensor_wait(gpu, downsample_filter)) return 0;
    alias_free_snake_kernel<<<h3cspeed_blocks(elements), 256, 0,
                              gpu->compute_stream>>>(
        static_cast<float *>(output->data),
        static_cast<const float *>(input->data),
        static_cast<const float *>(alpha_log->data),
        static_cast<const float *>(beta_log->data),
        static_cast<const float *>(upsample_filter->data),
        static_cast<const float *>(downsample_filter->data),
        batch, length, channels);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "alias-free Snake CUDA kernel");
}

__global__ static void audio_qkv_split_kernel(
        void *query, h3_gpu_dtype query_dtype,
        void *key, h3_gpu_dtype key_dtype,
        void *value, h3_gpu_dtype value_dtype,
        const void *qkv, h3_gpu_dtype qkv_dtype,
        const void *q_bias, h3_gpu_dtype q_bias_dtype,
        const void *k_bias, h3_gpu_dtype k_bias_dtype,
        const void *v_bias, h3_gpu_dtype v_bias_dtype,
        uint32_t batch, uint32_t length, uint32_t heads,
        uint32_t head_dim) {
    size_t total = (size_t)batch * length * heads * head_dim;
    uint32_t width = heads * head_dim;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += (size_t)gridDim.x * blockDim.x) {
        uint32_t dimension = (uint32_t)(index % head_dim);
        size_t vector_index = index / head_dim;
        uint32_t head = (uint32_t)(vector_index % heads);
        size_t row_index = vector_index / heads;
        uint32_t row = (uint32_t)(row_index % length);
        uint32_t batch_index = (uint32_t)(row_index / length);
        size_t source_base = ((size_t)batch_index * length + row) * width * 3;
        size_t projection = (size_t)head * head_dim + dimension;
        size_t destination = (((size_t)batch_index * heads + head) * length + row) *
                             head_dim + dimension;
        float q = h3cspeed_device_load(qkv, qkv_dtype, source_base + projection) +
                  (q_bias ? h3cspeed_device_load(q_bias, q_bias_dtype, projection) : 0.0f);
        float k = h3cspeed_device_load(qkv, qkv_dtype, source_base + width + projection) +
                  (k_bias ? h3cspeed_device_load(k_bias, k_bias_dtype, projection) : 0.0f);
        float v = h3cspeed_device_load(qkv, qkv_dtype, source_base + width * 2 + projection) +
                  (v_bias ? h3cspeed_device_load(v_bias, v_bias_dtype, projection) : 0.0f);
        h3cspeed_device_store(query, query_dtype, destination, q);
        h3cspeed_device_store(key, key_dtype, destination, k);
        h3cspeed_device_store(value, value_dtype, destination, v);
    }
}

extern "C" int h3_gpu_audio_qkv_split_f32(
        h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
        h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
        const h3_gpu_tensor *q_bias, const h3_gpu_tensor *k_bias,
        const h3_gpu_tensor *v_bias, uint32_t batch, uint32_t length,
        uint32_t heads, uint32_t head_dim) {
    size_t elements = 0;
    size_t qkv_elements = 0;
    size_t width = 0;
    if (!gpu || !query || !key || !value || !qkv || !batch || !length ||
        !heads || !head_dim || heads > UINT32_MAX / head_dim ||
        !h3cspeed_size_mul(heads, head_dim, &width) ||
        !h3cspeed_size_mul5(batch, length, heads, head_dim, 1, &elements) ||
        !h3cspeed_size_mul(elements, 3, &qkv_elements) ||
        query->elements < elements || key->elements < elements ||
        value->elements < elements || qkv->elements < qkv_elements ||
        (q_bias && q_bias->elements < width) ||
        (k_bias && k_bias->elements < width) ||
        (v_bias && v_bias->elements < width) ||
        !h3cspeed_tensor_wait(gpu, qkv) ||
        (q_bias && !h3cspeed_tensor_wait(gpu, q_bias)) ||
        (k_bias && !h3cspeed_tensor_wait(gpu, k_bias)) ||
        (v_bias && !h3cspeed_tensor_wait(gpu, v_bias))) return 0;
    audio_qkv_split_kernel<<<h3cspeed_blocks(elements), 256, 0,
                             gpu->compute_stream>>>(
        query->data, query->dtype, key->data, key->dtype,
        value->data, value->dtype, qkv->data, qkv->dtype,
        q_bias ? q_bias->data : nullptr, q_bias ? q_bias->dtype : H3_GPU_F32,
        k_bias ? k_bias->data : nullptr, k_bias ? k_bias->dtype : H3_GPU_F32,
        v_bias ? v_bias->data : nullptr, v_bias ? v_bias->dtype : H3_GPU_F32,
        batch, length, heads, head_dim);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "audio QKV split CUDA kernel");
}

static __device__ int reflect_coordinate(int coordinate, int length) {
    if (length <= 1) return 0;
    while (coordinate < 0 || coordinate >= length) {
        if (coordinate < 0) coordinate = -coordinate;
        if (coordinate >= length) coordinate = 2 * length - coordinate - 2;
    }
    return coordinate;
}

__global__ static void vae_pad_kernel(
        void *output, h3_gpu_dtype output_dtype,
        const void *input, h3_gpu_dtype input_dtype,
        uint32_t batch, uint32_t depth, uint32_t height, uint32_t width,
        uint32_t channels, uint32_t depth_front,
        uint32_t height_before, uint32_t height_after,
        uint32_t width_before, uint32_t width_after) {
    uint32_t output_depth = depth + depth_front;
    uint32_t output_height = height + height_before + height_after;
    uint32_t output_width = width + width_before + width_after;
    size_t total = (size_t)batch * output_depth * output_height * output_width * channels;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += (size_t)gridDim.x * blockDim.x) {
        uint32_t channel = (uint32_t)(index % channels);
        size_t cursor = index / channels;
        uint32_t x = (uint32_t)(cursor % output_width); cursor /= output_width;
        uint32_t y = (uint32_t)(cursor % output_height); cursor /= output_height;
        uint32_t z = (uint32_t)(cursor % output_depth);
        uint32_t batch_index = (uint32_t)(cursor / output_depth);
        float value = 0.0f;
        if (z >= depth_front) {
            int source_z = (int)(z - depth_front);
            int source_y = reflect_coordinate((int)y - (int)height_before,
                                               (int)height);
            int source_x = reflect_coordinate((int)x - (int)width_before,
                                               (int)width);
            size_t source = ((((size_t)batch_index * depth + (uint32_t)source_z) *
                              height + (uint32_t)source_y) * width +
                              (uint32_t)source_x) * channels + channel;
            value = h3cspeed_device_load(input, input_dtype, source);
        }
        h3cspeed_device_store(output, output_dtype, index, value);
    }
}

extern "C" int h3_gpu_vae_encoder_pad_f32(
        h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
        uint32_t batch, uint32_t depth, uint32_t height, uint32_t width,
        uint32_t channels, uint32_t depth_front,
        uint32_t height_before, uint32_t height_after,
        uint32_t width_before, uint32_t width_after) {
    if (!gpu || !output || !input || !batch || !depth || !height || !width ||
        !channels ||
        ((height_before || height_after) &&
         (height < 2 || height_before >= height || height_after >= height)) ||
        ((width_before || width_after) &&
         (width < 2 || width_before >= width || width_after >= width)) ||
        depth_front > UINT32_MAX - depth ||
        height_before > UINT32_MAX - height ||
        height_after > UINT32_MAX - height - height_before ||
        width_before > UINT32_MAX - width ||
        width_after > UINT32_MAX - width - width_before) return 0;
    uint32_t output_depth = depth + depth_front;
    uint32_t output_height = height + height_before + height_after;
    uint32_t output_width = width + width_before + width_after;
    size_t output_elements = 0;
    size_t input_elements = 0;
    if (!h3cspeed_size_mul5(batch, output_depth, output_height, output_width,
                            channels, &output_elements) ||
        !h3cspeed_size_mul5(batch, depth, height, width, channels,
                            &input_elements) ||
        output->elements < output_elements ||
        input->elements < input_elements || !h3cspeed_tensor_wait(gpu, input)) return 0;
    vae_pad_kernel<<<h3cspeed_blocks(output_elements), 256, 0,
                     gpu->compute_stream>>>(
        output->data, output->dtype, input->data, input->dtype,
        batch, depth, height, width, channels, depth_front,
        height_before, height_after, width_before, width_after);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "VAE padding CUDA kernel");
}

__global__ static void conv3d_kernel(
        void *output, h3_gpu_dtype output_dtype,
        const void *input, h3_gpu_dtype input_dtype,
        const void *weight, h3_gpu_dtype weight_dtype,
        const void *bias, h3_gpu_dtype bias_dtype,
        uint32_t batch, uint32_t depth, uint32_t height, uint32_t width,
        uint32_t input_channels, uint32_t output_channels,
        uint32_t kernel_depth, uint32_t kernel_height, uint32_t kernel_width,
        uint32_t stride_depth, uint32_t stride_height, uint32_t stride_width,
        uint32_t output_depth, uint32_t output_height, uint32_t output_width) {
    size_t total = (size_t)batch * output_depth * output_height *
                   output_width * output_channels;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += (size_t)gridDim.x * blockDim.x) {
        uint32_t output_channel = (uint32_t)(index % output_channels);
        size_t cursor = index / output_channels;
        uint32_t out_x = (uint32_t)(cursor % output_width); cursor /= output_width;
        uint32_t out_y = (uint32_t)(cursor % output_height); cursor /= output_height;
        uint32_t out_z = (uint32_t)(cursor % output_depth);
        uint32_t batch_index = (uint32_t)(cursor / output_depth);
        float accumulator = bias ? h3cspeed_device_load(bias, bias_dtype,
                                                         output_channel) : 0.0f;
        for (uint32_t input_channel = 0; input_channel < input_channels;
             input_channel++) {
            for (uint32_t kz = 0; kz < kernel_depth; kz++) {
                uint32_t source_z = out_z * stride_depth + kz;
                if (source_z >= depth) continue;
                for (uint32_t ky = 0; ky < kernel_height; ky++) {
                    uint32_t source_y = out_y * stride_height + ky;
                    if (source_y >= height) continue;
                    for (uint32_t kx = 0; kx < kernel_width; kx++) {
                        uint32_t source_x = out_x * stride_width + kx;
                        if (source_x >= width) continue;
                        size_t source = ((((size_t)batch_index * depth + source_z) *
                                          height + source_y) * width + source_x) *
                                        input_channels + input_channel;
                        size_t weight_index = (((((size_t)output_channel * input_channels +
                            input_channel) * kernel_depth + kz) * kernel_height + ky) *
                            kernel_width + kx);
                        accumulator += h3cspeed_device_load(input, input_dtype, source) *
                                       h3cspeed_device_load(weight, weight_dtype,
                                                            weight_index);
                    }
                }
            }
        }
        h3cspeed_device_store(output, output_dtype, index, accumulator);
    }
}

extern "C" int h3_gpu_conv3d_f32(
        h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
        const h3_gpu_tensor *weight, const h3_gpu_tensor *bias,
        uint32_t batch, uint32_t depth, uint32_t height, uint32_t width,
        uint32_t input_channels, uint32_t output_channels,
        uint32_t kernel_depth, uint32_t kernel_height, uint32_t kernel_width,
        uint32_t stride_depth, uint32_t stride_height, uint32_t stride_width) {
    if (!gpu || !output || !input || !weight || !batch || !depth ||
        !height || !width || !input_channels || !output_channels ||
        !kernel_depth || !kernel_height || !kernel_width || !stride_depth ||
        !stride_height || !stride_width || depth < kernel_depth ||
        height < kernel_height || width < kernel_width) return 0;
    uint32_t output_depth = (depth - kernel_depth) / stride_depth + 1;
    uint32_t output_height = (height - kernel_height) / stride_height + 1;
    uint32_t output_width = (width - kernel_width) / stride_width + 1;
    size_t output_elements = 0;
    size_t input_elements = 0;
    size_t weight_elements = 0;
    if (!h3cspeed_size_mul5(batch, output_depth, output_height, output_width,
                            output_channels, &output_elements) ||
        !h3cspeed_size_mul5(batch, depth, height, width, input_channels,
                            &input_elements) ||
        !h3cspeed_size_mul5(output_channels, input_channels, kernel_depth,
                            kernel_height, kernel_width, &weight_elements) ||
        output->elements < output_elements || input->elements < input_elements ||
        weight->elements < weight_elements || (bias && bias->elements < output_channels) ||
        !h3cspeed_tensor_wait(gpu, input) || !h3cspeed_tensor_wait(gpu, weight) ||
        (bias && !h3cspeed_tensor_wait(gpu, bias))) return 0;
    conv3d_kernel<<<h3cspeed_blocks(output_elements), 256, 0,
                    gpu->compute_stream>>>(
        output->data, output->dtype, input->data, input->dtype,
        weight->data, weight->dtype,
        bias ? bias->data : nullptr, bias ? bias->dtype : H3_GPU_F32,
        batch, depth, height, width, input_channels, output_channels,
        kernel_depth, kernel_height, kernel_width,
        stride_depth, stride_height, stride_width,
        output_depth, output_height, output_width);
    h3cspeed_count_conv(gpu);
    return h3cspeed_launch_ok(gpu, "Conv3d CUDA kernel");
}

__global__ static void group_norm_silu_kernel(
        void *output, h3_gpu_dtype output_dtype,
        const void *input, h3_gpu_dtype input_dtype,
        const void *weight, h3_gpu_dtype weight_dtype,
        const void *bias, h3_gpu_dtype bias_dtype,
        uint32_t batch, uint32_t depth, uint32_t height, uint32_t width,
        uint32_t channels, uint32_t groups, float epsilon) {
    uint32_t group_index = blockIdx.x;
    uint32_t batch_index = group_index / groups;
    uint32_t group = group_index % groups;
    if (batch_index >= batch) return;
    uint32_t channels_per_group = channels / groups;
    size_t spatial = (size_t)depth * height * width;
    size_t values = spatial * channels_per_group;
    extern __shared__ float shared[];
    float sum = 0.0f;
    float square = 0.0f;
    for (size_t local = threadIdx.x; local < values; local += blockDim.x) {
        size_t spatial_index = local / channels_per_group;
        uint32_t local_channel = (uint32_t)(local % channels_per_group);
        uint32_t channel = group * channels_per_group + local_channel;
        size_t source = ((size_t)batch_index * spatial + spatial_index) * channels +
                        channel;
        float value = h3cspeed_device_load(input, input_dtype, source);
        sum += value;
        square += value * value;
    }
    shared[threadIdx.x] = sum;
    shared[blockDim.x + threadIdx.x] = square;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
            shared[blockDim.x + threadIdx.x] +=
                shared[blockDim.x + threadIdx.x + stride];
        }
        __syncthreads();
    }
    float mean = shared[0] / (float)values;
    float variance = fmaxf(shared[blockDim.x] / (float)values - mean * mean, 0.0f);
    float inverse = rsqrtf(variance + epsilon);
    for (size_t local = threadIdx.x; local < values; local += blockDim.x) {
        size_t spatial_index = local / channels_per_group;
        uint32_t local_channel = (uint32_t)(local % channels_per_group);
        uint32_t channel = group * channels_per_group + local_channel;
        size_t destination = ((size_t)batch_index * spatial + spatial_index) * channels +
                             channel;
        float value = (h3cspeed_device_load(input, input_dtype, destination) - mean) *
                      inverse;
        value = value * h3cspeed_device_load(weight, weight_dtype, channel) +
                h3cspeed_device_load(bias, bias_dtype, channel);
        value = value / (1.0f + expf(-value));
        h3cspeed_device_store(output, output_dtype, destination, value);
    }
}

extern "C" int h3_gpu_vae_encoder_group_norm_silu_f32(
        h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
        const h3_gpu_tensor *weight, const h3_gpu_tensor *bias,
        uint32_t batch, uint32_t depth, uint32_t height, uint32_t width,
        uint32_t channels, uint32_t groups, float epsilon) {
    size_t elements = 0;
    if (!gpu || !output || !input || !weight || !bias || !batch || !depth ||
        !height || !width || !channels || !groups || channels % groups ||
        !(epsilon > 0.0f) || !std::isfinite(epsilon) ||
        batch > UINT32_MAX / groups ||
        !h3cspeed_size_mul5(batch, depth, height, width, channels, &elements) ||
        output->elements < elements || input->elements < elements ||
        weight->elements < channels || bias->elements < channels ||
        !h3cspeed_tensor_wait(gpu, input) || !h3cspeed_tensor_wait(gpu, weight) ||
        !h3cspeed_tensor_wait(gpu, bias)) return 0;
    group_norm_silu_kernel<<<batch * groups, 256, 512 * sizeof(float),
                             gpu->compute_stream>>>(
        output->data, output->dtype, input->data, input->dtype,
        weight->data, weight->dtype, bias->data, bias->dtype,
        batch, depth, height, width, channels, groups, epsilon);
    h3cspeed_count_direct(gpu);
    return h3cspeed_launch_ok(gpu, "VAE group norm SiLU CUDA kernel");
}
