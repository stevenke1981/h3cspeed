#include "h3_weights.h"
#include "h3_quantized_weights.h"

#include <dirent.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct h3_weight_store {
    h3_st_header *headers;
    size_t count;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int safetensors_name(const char *name) {
    static const char suffix[] = ".safetensors";
    size_t length = strlen(name);
    return length > sizeof(suffix) - 1 &&
           strcmp(name + length - (sizeof(suffix) - 1), suffix) == 0;
}

static int compare_paths(const void *left, const void *right) {
    const char *const *a = left;
    const char *const *b = right;
    return strcmp(*a, *b);
}

static void free_paths(char **paths, size_t count) {
    if (!paths) return;
    for (size_t index = 0; index < count; index++) free(paths[index]);
    free(paths);
}

h3_weight_store *h3_weight_store_open(const char *directory,
                                      char *error, size_t error_size) {
    if (!directory || !*directory) {
        fail(error, error_size, "weight directory is required");
        return NULL;
    }
    DIR *stream = opendir(directory);
    if (!stream) {
        fail(error, error_size, "cannot open weight directory: %s", directory);
        return NULL;
    }
    char **paths = NULL;
    size_t count = 0;
    size_t capacity = 0;
    struct dirent *entry;
    while ((entry = readdir(stream)) != NULL) {
        if (!safetensors_name(entry->d_name)) continue;
        if (count == capacity) {
            size_t next = capacity ? capacity * 2 : 8;
            char **grown = realloc(paths, next * sizeof(*grown));
            if (!grown) {
                fail(error, error_size, "out of memory listing weight shards");
                closedir(stream);
                free_paths(paths, count);
                return NULL;
            }
            paths = grown;
            capacity = next;
        }
        size_t length = strlen(directory) + strlen(entry->d_name) + 2;
        paths[count] = malloc(length);
        if (!paths[count]) {
            fail(error, error_size, "out of memory resolving a weight shard");
            closedir(stream);
            free_paths(paths, count);
            return NULL;
        }
        snprintf(paths[count], length, "%s/%s", directory, entry->d_name);
        count++;
    }
    closedir(stream);
    if (!count) {
        fail(error, error_size, "no safetensors shards in %s", directory);
        free(paths);
        return NULL;
    }
    qsort(paths, count, sizeof(*paths), compare_paths);
    h3_weight_store *store = calloc(1, sizeof(*store));
    if (!store) {
        fail(error, error_size, "out of memory creating weight store");
        free_paths(paths, count);
        return NULL;
    }
    store->headers = calloc(count, sizeof(*store->headers));
    if (!store->headers) {
        fail(error, error_size, "out of memory allocating weight headers");
        free(store);
        free_paths(paths, count);
        return NULL;
    }
    store->count = count;
    for (size_t index = 0; index < count; index++) {
        char detail[384];
        if (!h3_st_read_header(paths[index], &store->headers[index], detail,
                               sizeof(detail))) {
            fail(error, error_size, "%s", detail);
            free_paths(paths, count);
            h3_weight_store_free(store);
            return NULL;
        }
    }
    free_paths(paths, count);
    return store;
}

void h3_weight_store_free(h3_weight_store *store) {
    if (!store) return;
    for (size_t index = 0; index < store->count; index++) {
        h3_st_free_header(&store->headers[index]);
    }
    free(store->headers);
    free(store);
}

size_t h3_weight_store_shards(const h3_weight_store *store) {
    return store ? store->count : 0;
}

const h3_st_tensor *h3_weight_find(const h3_weight_store *store,
                                   const char *name,
                                   const h3_st_header **header) {
    if (header) *header = NULL;
    if (!store || !name) return NULL;
    for (size_t index = 0; index < store->count; index++) {
        const h3_st_tensor *tensor = h3_st_find(&store->headers[index], name);
        if (tensor) {
            if (header) *header = &store->headers[index];
            return tensor;
        }
    }
    /* Comfy-Org's consolidated Qwen checkpoint omits the upstream wrapper
     * prefixes. Keep the alias virtual so the 15 GiB file is never copied. */
    const char *alias = NULL;
    if (!strncmp(name, "model.language_model.", 21)) alias = name + 15;
    else if (!strncmp(name, "model.visual.", 13)) alias = name + 6;
    if (alias) {
        for (size_t index = 0; index < store->count; index++) {
            const h3_st_tensor *tensor = h3_st_find(&store->headers[index], alias);
            if (tensor) {
                if (header) *header = &store->headers[index];
                return tensor;
            }
        }
    }
    return NULL;
}

static int tensor_shape(const h3_st_tensor *tensor, const char *name,
                        int ndim, const uint64_t *shape, uint64_t *elements,
                        char *error, size_t error_size) {
    if (!tensor || tensor->ndim != ndim) {
        fail(error, error_size, "weight %s has rank %d, expected %d", name,
             tensor ? tensor->ndim : -1, ndim);
        return 0;
    }
    uint64_t count = 1;
    for (int dimension = 0; dimension < ndim; dimension++) {
        if (tensor->shape[dimension] != shape[dimension] ||
            (shape[dimension] && count > UINT64_MAX / shape[dimension])) {
            fail(error, error_size, "weight %s shape mismatch at dimension %d",
                 name, dimension);
            return 0;
        }
        count *= shape[dimension];
    }
    if (count > SIZE_MAX) {
        fail(error, error_size, "weight %s is too large for this process", name);
        return 0;
    }
    *elements = count;
    return 1;
}

static int quant_marker(const h3_weight_store *store,
                        const h3_st_tensor *weight, const char *required,
                        const h3_st_header **marker_header,
                        const h3_st_tensor **marker_tensor) {
    char name[256];
    if (!weight) return 0;
    size_t base = strlen(weight->name);
    if (base >= 7 && !strcmp(weight->name + base - 7, ".weight")) base -= 7;
    if (base + strlen(".comfy_quant") + 1 > sizeof(name)) return 0;
    memcpy(name, weight->name, base);
    strcpy(name + base, ".comfy_quant");
    const h3_st_tensor *marker = h3_weight_find(store, name, marker_header);
    if (!marker || marker->dtype != H3_DTYPE_U8 || marker->ndim != 1 ||
        marker->shape[0] < 2 || marker->shape[0] > 512) return 0;
    char json[513] = {0};
    char ignored[128] = {0};
    if (!h3_st_read_data(*marker_header, marker, json, (size_t)marker->shape[0],
                         ignored, sizeof(ignored)))
        return 0;
    size_t write = 0;
    for (size_t read = 0; json[read]; read++)
        if (json[read] != ' ' && json[read] != '\t' &&
            json[read] != '\r' && json[read] != '\n')
            json[write++] = json[read];
    json[write] = '\0';
    if (!strstr(json, required)) return 0;
    if (marker_tensor) *marker_tensor = marker;
    return 1;
}

int h3_weight_is_convrot_i8(const h3_weight_store *store, const char *name) {
    const h3_st_header *header = NULL, *marker_header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    (void)header;
    return tensor && tensor->dtype == H3_DTYPE_I8 &&
           quant_marker(store, tensor, "\"convrot\":true", &marker_header, NULL) &&
           quant_marker(store, tensor, "\"convrot_groupsize\":256",
                        &marker_header, NULL);
}

h3_gpu_tensor *h3_weight_load_i8_convrot(
        const h3_weight_store *store, h3_gpu *gpu, const char *name,
        int ndim, const uint64_t *shape, h3_gpu_tensor **scales,
        char *error, size_t error_size) {
    const h3_st_header *header = NULL, *scale_header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    uint64_t elements = 0;
    if (scales) *scales = NULL;
    if (!tensor || tensor->dtype != H3_DTYPE_I8 ||
        !tensor_shape(tensor, name, ndim, shape, &elements, error, error_size) ||
        !h3_weight_is_convrot_i8(store, name)) {
        if (error && error_size && !error[0])
            fail(error, error_size, "weight %s is not ConvRot INT8/group-256", name);
        return NULL;
    }
    char scale_name[256];
    size_t base = strlen(tensor->name);
    if (base >= 7 && !strcmp(tensor->name + base - 7, ".weight")) base -= 7;
    if (base + strlen(".weight_scale") + 1 > sizeof(scale_name)) {
        fail(error, error_size, "weight name is too long: %s", name);
        return NULL;
    }
    memcpy(scale_name, tensor->name, base);
    strcpy(scale_name + base, ".weight_scale");
    const h3_st_tensor *scale = h3_weight_find(store, scale_name, &scale_header);
    if (!scale || scale->dtype != H3_DTYPE_F32 || scale->ndim != 2 ||
        scale->shape[0] != shape[0] || scale->shape[1] != 1) {
        fail(error, error_size, "weight %s has invalid ConvRot scale", name);
        return NULL;
    }
    h3_gpu_tensor *weight = h3cspeed_gpu_tensor_load_i8_convrot(
        gpu, header->path, tensor->file_offset, (size_t)elements, 256);
    h3_gpu_tensor *loaded_scale = h3_gpu_tensor_load_f32(
        gpu, scale_header->path, scale->file_offset, (size_t)shape[0]);
    if (!weight || !loaded_scale) {
        h3_gpu_tensor_free(weight);
        h3_gpu_tensor_free(loaded_scale);
        fail(error, error_size, "cannot load ConvRot INT8 %s: %s", name,
             h3_gpu_error(gpu));
        return NULL;
    }
    *scales = loaded_scale;
    return weight;
}

static h3_gpu_tensor *load_tensor(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape, h3_dtype dtype,
                                  char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor) {
        fail(error, error_size, "required weight is absent: %s", name);
        return NULL;
    }
    int convertible = (dtype == H3_DTYPE_BF16 && tensor->dtype == H3_DTYPE_F16) ||
                      (dtype == H3_DTYPE_F32 && tensor->dtype == H3_DTYPE_F16);
    if ((!convertible && tensor->dtype != dtype) || tensor->ndim != ndim) {
        fail(error, error_size, "weight %s has dtype/rank %s/%d, expected %s/%d",
             name, h3_dtype_name(tensor->dtype), tensor->ndim,
             h3_dtype_name(dtype), ndim);
        return NULL;
    }
    uint64_t elements = 1;
    for (int dimension = 0; dimension < ndim; dimension++) {
        if (tensor->shape[dimension] != shape[dimension]) {
            fail(error, error_size, "weight %s shape mismatch at dimension %d",
                 name, dimension);
            return NULL;
        }
        if (shape[dimension] && elements > UINT64_MAX / shape[dimension]) {
            fail(error, error_size, "weight %s shape overflows", name);
            return NULL;
        }
        elements *= shape[dimension];
    }
    if (elements > SIZE_MAX) {
        fail(error, error_size, "weight %s is too large for this process", name);
        return NULL;
    }
    h3_gpu_tensor *result = NULL;
    if (tensor->dtype == H3_DTYPE_F16 && dtype == H3_DTYPE_BF16)
        result = h3cspeed_gpu_tensor_load_f16_as_bf16(
            gpu, header->path, tensor->file_offset, (size_t)elements);
    else if (tensor->dtype == H3_DTYPE_F16 && dtype == H3_DTYPE_F32)
        result = h3cspeed_gpu_tensor_load_f16_as_f32(
            gpu, header->path, tensor->file_offset, (size_t)elements);
    else if (dtype == H3_DTYPE_BF16)
        result = h3_gpu_tensor_load_bf16(gpu, header->path, tensor->file_offset,
                                        (size_t)elements);
    else
        result = h3_gpu_tensor_load_f32(gpu, header->path, tensor->file_offset,
                                       (size_t)elements);
    if (!result) {
        fail(error, error_size, "cannot load %s: %s", name, h3_gpu_error(gpu));
    }
    return result;
}

static const h3_st_tensor *companion(const h3_weight_store *store,
                                      const h3_st_tensor *weight,
                                      const char *suffix,
                                      const h3_st_header **header) {
    char name[320];
    if (!weight) return NULL;
    size_t base = strlen(weight->name);
    if (base >= 7 && !strcmp(weight->name + base - 7, ".weight")) base -= 7;
    if (base + strlen(suffix) + 1 > sizeof(name)) return NULL;
    memcpy(name, weight->name, base);
    strcpy(name + base, suffix);
    return h3_weight_find(store, name, header);
}

static int read_f32_scalar(const h3_st_header *header,
                           const h3_st_tensor *tensor, float *value) {
    char error[128] = {0};
    return header && tensor && value && tensor->dtype == H3_DTYPE_F32 &&
           h3_st_tensor_elements(tensor) == 1 &&
           h3_st_read_data(header, tensor, value, sizeof(*value),
                           error, sizeof(error));
}

int h3_weight_read_bf16(const h3_weight_store *store, const char *name,
                        int ndim, const uint64_t *shape,
                        h3_gpu_tensor *destination,
                        char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    uint64_t elements = 0;
    if (!tensor) {
        fail(error, error_size, "required weight is absent: %s", name);
        return 0;
    }
    if (tensor->dtype == H3_DTYPE_U8) {
        if (ndim != 2 || tensor->ndim != 2 || shape[1] % 2 ||
            tensor->shape[0] != shape[0] || tensor->shape[1] != shape[1] / 2 ||
            shape[0] > UINT64_MAX / shape[1]) {
            fail(error, error_size, "weight %s NVFP4 packed shape mismatch", name);
            return 0;
        }
        elements = shape[0] * shape[1];
    } else if (!tensor_shape(tensor, name, ndim, shape, &elements,
                             error, error_size)) {
        return 0;
    }
    if (tensor->dtype == H3_DTYPE_BF16)
        return h3_gpu_tensor_read_file_bf16(destination, header->path,
            tensor->file_offset, (size_t)elements, error, error_size);
    if (tensor->dtype == H3_DTYPE_F16)
        return h3cspeed_gpu_tensor_read_f16_as_bf16(destination, header->path,
            tensor->file_offset, (size_t)elements, error, error_size);
    if (tensor->dtype == H3_DTYPE_I8) {
        const h3_st_header *scale_header = NULL, *marker_header = NULL;
        const h3_st_tensor *scale = companion(store, tensor, ".weight_scale",
                                               &scale_header);
        if (!scale || scale->dtype != H3_DTYPE_F32 || scale->ndim != 2 ||
            scale->shape[0] != shape[0] || scale->shape[1] != 1 ||
            !quant_marker(store, tensor, "\"format\":\"int8_tensorwise\"",
                          &marker_header, NULL)) {
            fail(error, error_size, "weight %s has invalid INT8 metadata", name);
            return 0;
        }
        return h3cspeed_gpu_tensor_read_i8_as_bf16(
            destination, header->path, tensor->file_offset,
            scale_header->path, scale->file_offset, (uint32_t)shape[0],
            (uint32_t)shape[1], error, error_size);
    }
    if (tensor->dtype == H3_DTYPE_U8) {
        const h3_st_header *scale_header = NULL, *scale2_header = NULL;
        const h3_st_header *pre_header = NULL, *marker_header = NULL;
        const h3_st_tensor *scale = companion(store, tensor, ".weight_scale",
                                               &scale_header);
        const h3_st_tensor *scale2 = companion(store, tensor, ".weight_scale_2",
                                                &scale2_header);
        const h3_st_tensor *pre = companion(store, tensor, ".pre_quant_scale",
                                             &pre_header);
        float tensor_scale = 0.0f;
        uint64_t blocks = (shape[1] + 15) / 16;
        if (shape[1] % 2 || tensor->shape[0] != shape[0] ||
            tensor->shape[1] * 2 != shape[1] || !scale ||
            scale->dtype != H3_DTYPE_F8_E4M3 || scale->ndim != 2 ||
            scale->shape[0] != shape[0] || scale->shape[1] != blocks ||
            !read_f32_scalar(scale2_header, scale2, &tensor_scale) ||
            !quant_marker(store, tensor, "\"format\":\"nvfp4\"",
                          &marker_header, NULL) ||
            (pre && (pre->dtype != H3_DTYPE_BF16 || pre->ndim != 1 ||
                     pre->shape[0] != shape[1]))) {
            fail(error, error_size, "weight %s has invalid NVFP4 metadata", name);
            return 0;
        }
        return h3cspeed_gpu_tensor_read_nvfp4_as_bf16(
            destination, header->path, tensor->file_offset,
            scale_header->path, scale->file_offset, tensor_scale,
            NULL, 0,
            (uint32_t)shape[0], (uint32_t)shape[1], error, error_size);
    }
    fail(error, error_size, "weight %s has unsupported quantized dtype %s",
         name, h3_dtype_name(tensor->dtype));
    return 0;
}

h3_gpu_tensor *h3_weight_load_optional_bf16(
        const h3_weight_store *store, h3_gpu *gpu, const char *name,
        int ndim, const uint64_t *shape, char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    if (!h3_weight_find(store, name, &header)) return NULL;
    return h3_weight_load_bf16(store, gpu, name, ndim, shape,
                               error, error_size);
}

h3_gpu_tensor *h3_weight_load_bf16(const h3_weight_store *store, h3_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (tensor && (tensor->dtype == H3_DTYPE_I8 ||
                   tensor->dtype == H3_DTYPE_U8)) {
        uint64_t elements = 0;
        if (ndim < 0 || ndim > 8) return NULL;
        elements = 1;
        for (int dimension = 0; dimension < ndim; dimension++) {
            if (shape[dimension] && elements > UINT64_MAX / shape[dimension]) {
                fail(error, error_size, "weight %s shape overflows", name);
                return NULL;
            }
            elements *= shape[dimension];
        }
        if (elements > SIZE_MAX) return NULL;
        h3_gpu_tensor *result = h3_gpu_tensor_new_bf16(gpu, (size_t)elements);
        if (!result) {
            fail(error, error_size,
                 "cannot allocate quantized destination for %s: %s", name,
                 h3_gpu_error(gpu));
            return NULL;
        }
        if (!h3_weight_read_bf16(store, name, ndim, shape, result,
                                 error, error_size)) {
            if (error && error_size && !error[0])
                fail(error, error_size,
                     "cannot decode quantized weight %s: %s", name,
                     h3_gpu_error(gpu));
            h3_gpu_tensor_free(result);
            return NULL;
        }
        return result;
    }
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_BF16,
                       error, error_size);
}

h3_gpu_tensor *h3_weight_load_f32(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_F32,
                       error, error_size);
}
