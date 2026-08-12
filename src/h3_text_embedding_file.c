#include "h3_text_embedding_file.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(_WIN32)
#include <sddl.h>
#include <windows.h>
#endif

#define H3CSPEED_TEXT_FLAG_TAGS UINT32_C(1)
#define H3CSPEED_TEXT_VERSION_V1 UINT32_C(1)
#define H3CSPEED_TEXT_VERSION_V2 UINT32_C(2)
#define H3CSPEED_TEXT_MAX_FILE (UINT64_C(64) * UINT64_C(1024) * UINT64_C(1024))

static void set_error(char *error, size_t error_size, const char *message);

static int create_private_snapshot_directory(
        char *directory, size_t directory_size,
        char *error, size_t error_size) {
#if defined(_WIN32)
    char temporary[MAX_PATH + 1];
    DWORD length = GetTempPathA((DWORD)sizeof(temporary), temporary);
    PSECURITY_DESCRIPTOR descriptor = NULL;
    SECURITY_ATTRIBUTES attributes;
    if (length == 0 || length >= sizeof(temporary)) {
        set_error(error, error_size, "cannot resolve the per-user temporary directory");
        return 0;
    }
    if (!ConvertStringSecurityDescriptorToSecurityDescriptorA(
            "D:P(A;;FA;;;SY)(A;;FA;;;BA)(A;;FA;;;OW)", SDDL_REVISION_1,
            &descriptor, NULL)) {
        set_error(error, error_size, "cannot create the private snapshot ACL");
        return 0;
    }
    attributes.nLength = sizeof(attributes);
    attributes.lpSecurityDescriptor = descriptor;
    attributes.bInheritHandle = FALSE;
    for (unsigned attempt = 0; attempt < 100u; attempt++) {
        unsigned int first = 0;
        unsigned int second = 0;
        if (rand_s(&first) != 0 || rand_s(&second) != 0) {
            LocalFree(descriptor);
            set_error(error, error_size, "cannot generate a private snapshot name");
            return 0;
        }
        int written = snprintf(
            directory, directory_size, "%sh3cspeed-keyframe-%08x%08x",
            temporary, first, second);
        if (written < 0 || (size_t)written >= directory_size) {
            LocalFree(descriptor);
            set_error(error, error_size, "private snapshot path is too long");
            return 0;
        }
        if (CreateDirectoryA(directory, &attributes)) {
            LocalFree(descriptor);
            return 1;
        }
        if (GetLastError() != ERROR_ALREADY_EXISTS) break;
    }
    LocalFree(descriptor);
    set_error(error, error_size, "cannot create private keyframe snapshot directory");
    return 0;
#else
    int written = snprintf(directory, directory_size,
                           "/tmp/h3cspeed-keyframe-XXXXXX");
    if (written < 0 || (size_t)written >= directory_size ||
        !mkdtemp(directory)) {
        set_error(error, error_size, "cannot create private keyframe snapshot directory");
        return 0;
    }
    return 1;
#endif
}

typedef struct {
    uint32_t state[8];
    uint64_t bits;
    unsigned char block[64];
    size_t used;
} h3_sha256;

static uint32_t sha_rotr(uint32_t value, unsigned count) {
    return (value >> count) | (value << (32u - count));
}

static void sha_transform(h3_sha256 *sha, const unsigned char block[64]) {
    static const uint32_t k[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
    uint32_t w[64];
    for (unsigned i = 0; i < 16; i++)
        w[i] = ((uint32_t)block[i * 4] << 24) | ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) | (uint32_t)block[i * 4 + 3];
    for (unsigned i = 16; i < 64; i++) {
        uint32_t s0 = sha_rotr(w[i - 15], 7) ^ sha_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = sha_rotr(w[i - 2], 17) ^ sha_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = sha->state[0], b = sha->state[1], c = sha->state[2], d = sha->state[3];
    uint32_t e = sha->state[4], f = sha->state[5], g = sha->state[6], h = sha->state[7];
    for (unsigned i = 0; i < 64; i++) {
        uint32_t s1 = sha_rotr(e, 6) ^ sha_rotr(e, 11) ^ sha_rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + s1 + ch + k[i] + w[i];
        uint32_t s0 = sha_rotr(a, 2) ^ sha_rotr(a, 13) ^ sha_rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = s0 + maj;
        h = g; g = f; f = e; e = d + temp1; d = c; c = b; b = a; a = temp1 + temp2;
    }
    sha->state[0] += a; sha->state[1] += b; sha->state[2] += c; sha->state[3] += d;
    sha->state[4] += e; sha->state[5] += f; sha->state[6] += g; sha->state[7] += h;
}

static void sha_init(h3_sha256 *sha) {
    static const uint32_t initial[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    memcpy(sha->state, initial, sizeof(initial));
    sha->bits = 0;
    sha->used = 0;
}

static void sha_update(h3_sha256 *sha, const unsigned char *data, size_t length) {
    sha->bits += (uint64_t)length * 8u;
    while (length) {
        size_t take = sizeof(sha->block) - sha->used;
        if (take > length) take = length;
        memcpy(sha->block + sha->used, data, take);
        sha->used += take; data += take; length -= take;
        if (sha->used == sizeof(sha->block)) { sha_transform(sha, sha->block); sha->used = 0; }
    }
}

static void sha_final(h3_sha256 *sha, uint8_t output[32]) {
    size_t used = sha->used;
    sha->block[used++] = 0x80;
    if (used > 56) { memset(sha->block + used, 0, 64 - used); sha_transform(sha, sha->block); used = 0; }
    memset(sha->block + used, 0, 56 - used);
    for (unsigned i = 0; i < 8; i++) sha->block[56 + i] = (unsigned char)(sha->bits >> (56u - i * 8u));
    sha_transform(sha, sha->block);
    for (unsigned i = 0; i < 8; i++) {
        output[i * 4] = (unsigned char)(sha->state[i] >> 24);
        output[i * 4 + 1] = (unsigned char)(sha->state[i] >> 16);
        output[i * 4 + 2] = (unsigned char)(sha->state[i] >> 8);
        output[i * 4 + 3] = (unsigned char)sha->state[i];
    }
}

int h3cspeed_sha256_file(const char *path, uint8_t output[32], char *error, size_t error_size) {
    unsigned char buffer[64 * 1024];
    h3_sha256 sha;
    FILE *file;
    if (error && error_size) error[0] = '\0';
    if (!path || !*path || !output) { set_error(error, error_size, "image path and SHA-256 output are required"); return 0; }
    file = fopen(path, "rb");
    if (!file) { if (error && error_size) (void)snprintf(error, error_size, "cannot open image for SHA-256: %s", strerror(errno)); return 0; }
    sha_init(&sha);
    for (;;) {
        size_t count = fread(buffer, 1, sizeof(buffer), file);
        if (count) sha_update(&sha, buffer, count);
        if (count < sizeof(buffer)) {
            if (ferror(file)) { set_error(error, error_size, "cannot read image for SHA-256"); (void)fclose(file); return 0; }
            break;
        }
    }
    if (fclose(file) != 0) { set_error(error, error_size, "cannot close image after SHA-256"); return 0; }
    sha_final(&sha, output);
    return 1;
}

void h3cspeed_keyframe_snapshot_discard(h3cspeed_keyframe_snapshot *snapshot) {
    if (!snapshot) return;
#if defined(_WIN32)
    if (snapshot->path[0]) (void)DeleteFileA(snapshot->path);
    if (snapshot->directory[0]) (void)RemoveDirectoryA(snapshot->directory);
#else
    if (snapshot->path[0]) (void)unlink(snapshot->path);
    if (snapshot->directory[0]) (void)rmdir(snapshot->directory);
#endif
    memset(snapshot, 0, sizeof(*snapshot));
}

int h3cspeed_keyframe_snapshot_create(
        const char *source, h3cspeed_keyframe_snapshot *snapshot,
        char *error, size_t error_size) {
    unsigned char buffer[64 * 1024];
    FILE *input = NULL;
    FILE *output = NULL;
    int ok = 0;
    if (error && error_size) error[0] = '\0';
    if (!source || !*source || !snapshot) {
        set_error(error, error_size, "keyframe snapshot source is required");
        return 0;
    }
    memset(snapshot, 0, sizeof(*snapshot));
    if (!create_private_snapshot_directory(
            snapshot->directory, sizeof(snapshot->directory),
            error, error_size)) goto cleanup;
#if defined(_WIN32)
    if ((size_t)snprintf(snapshot->path, sizeof(snapshot->path), "%s\\image.png",
                         snapshot->directory) >= sizeof(snapshot->path)) {
#else
    if ((size_t)snprintf(snapshot->path, sizeof(snapshot->path), "%s/image.png",
                         snapshot->directory) >= sizeof(snapshot->path)) {
#endif
        set_error(error, error_size, "keyframe snapshot path is too long");
        goto cleanup;
    }
    input = fopen(source, "rb");
    if (!input) {
        set_error(error, error_size, "cannot open keyframe for snapshot");
        goto cleanup;
    }
    output = fopen(snapshot->path, "wb");
    if (!output) {
        set_error(error, error_size, "cannot create keyframe snapshot");
        goto cleanup;
    }
    for (;;) {
        size_t count = fread(buffer, 1, sizeof(buffer), input);
        if (count && fwrite(buffer, 1, count, output) != count) {
            set_error(error, error_size, "cannot write complete keyframe snapshot");
            goto cleanup;
        }
        if (count < sizeof(buffer)) {
            if (ferror(input)) {
                set_error(error, error_size, "cannot read complete keyframe snapshot");
                goto cleanup;
            }
            break;
        }
    }
    if (fclose(input) != 0) {
        input = NULL;
        set_error(error, error_size, "cannot close keyframe snapshot source");
        goto cleanup;
    }
    input = NULL;
    if (fflush(output) != 0) {
        set_error(error, error_size, "cannot commit keyframe snapshot");
        goto cleanup;
    }
    if (fclose(output) != 0) {
        output = NULL;
        set_error(error, error_size, "cannot close keyframe snapshot");
        goto cleanup;
    }
    output = NULL;
    if (!h3cspeed_sha256_file(snapshot->path, snapshot->sha256,
                               error, error_size)) goto cleanup;
    ok = 1;

cleanup:
    if (input) (void)fclose(input);
    if (output) (void)fclose(output);
    if (!ok) h3cspeed_keyframe_snapshot_discard(snapshot);
    return ok;
}

static void set_error(char *error, size_t error_size, const char *message) {
    if (error && error_size) {
        (void)snprintf(error, error_size, "%s", message);
    }
}

static void set_errorf(char *error, size_t error_size, const char *format,
                       unsigned long long value) {
    if (error && error_size) (void)snprintf(error, error_size, format, value);
}

static int add_u64(uint64_t left, uint64_t right, uint64_t *result) {
    if (left > UINT64_MAX - right) return 0;
    *result = left + right;
    return 1;
}

static int mul_u64(uint64_t left, uint64_t right, uint64_t *result) {
    if (left && right > UINT64_MAX / left) return 0;
    *result = left * right;
    return 1;
}

static uint32_t read_u32(const unsigned char *bytes) {
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
           ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static uint64_t read_u64(const unsigned char *bytes) {
    uint64_t value = 0;
    for (unsigned index = 0; index < 8; index++)
        value |= (uint64_t)bytes[index] << (index * 8u);
    return value;
}

/* Reject malformed UTF-8 in recipe and prompt metadata without normalizing it.
 * Expected prompt matching below remains byte-exact. */
static int valid_utf8(const unsigned char *bytes, size_t length) {
    size_t index = 0;
    while (index < length) {
        unsigned char first = bytes[index++];
        if (first <= 0x7fu) continue;
        unsigned count;
        uint32_t codepoint;
        if (first >= 0xc2u && first <= 0xdfu) {
            count = 1;
            codepoint = first & 0x1fu;
        } else if (first >= 0xe0u && first <= 0xefu) {
            count = 2;
            codepoint = first & 0x0fu;
        } else if (first >= 0xf0u && first <= 0xf4u) {
            count = 3;
            codepoint = first & 0x07u;
        } else {
            return 0;
        }
        if (length - index < count) return 0;
        for (unsigned part = 0; part < count; part++) {
            unsigned char next = bytes[index++];
            if ((next & 0xc0u) != 0x80u) return 0;
            codepoint = (codepoint << 6) | (next & 0x3fu);
        }
        if ((count == 2 && codepoint < 0x800u) ||
            (count == 3 && codepoint < 0x10000u) || codepoint > 0x10ffffu ||
            (codepoint >= 0xd800u && codepoint <= 0xdfffu)) return 0;
    }
    return 1;
}

static int read_exact(FILE *file, unsigned char *buffer, size_t size) {
    return size == 0 || fread(buffer, 1, size, file) == size;
}

int h3cspeed_text_embedding_load_file_ex(
    const char *path, const char *expected_prompt,
    const uint32_t *expected_token_ids, size_t expected_token_count,
    const uint8_t expected_model_sha256[32],
    const h3_text_embedding_file_expectation *expectation,
    h3_text_embedding *output, char *error, size_t error_size) {
    FILE *file = NULL;
    unsigned char *bytes = NULL;
    uint16_t *values = NULL;
    uint8_t *tags = NULL;
    uint64_t file_size_u64;
    size_t file_size;
    uint64_t prompt_bytes, token_count, recipe_bytes, embedding_bytes;
    uint64_t tags_bytes, token_ids_bytes, expected_size;
    uint64_t header_embedding_bytes, header_tags_bytes, header_token_ids_bytes;
    uint64_t offset;
    uint64_t calculated_embedding_bytes, calculated_token_ids_bytes;
    size_t prompt_size, recipe_size, token_ids_size, embedding_size, tags_size;
    uint32_t header_size, version, width, flags;
    uint32_t mode = H3CSPEED_TEXT_EMBEDDING_MODE_T2V;
    uint32_t keyframe_role = 0;
    uint32_t keyframe_count = 0;
    uint32_t keyframe_order = 0;
    uint32_t first_resize_policy = H3CSPEED_TEXT_EMBEDDING_RESIZE_STRETCH;
    uint32_t last_resize_policy = H3CSPEED_TEXT_EMBEDDING_RESIZE_COVER;
    uint32_t render_width = 0;
    uint32_t render_height = 0;
    uint32_t metadata_bytes = 0;
    size_t metadata_size = 0;
    const unsigned char *metadata = NULL;
    uint64_t elements;
    int ok = 0;

    if (error && error_size) error[0] = '\0';
    if (output) memset(output, 0, sizeof(*output));
    if (!path || !*path || !expected_prompt || !expected_model_sha256 || !output) {
        set_error(error, error_size, "sidecar path, expected prompt, model hash, and output are required");
        return 0;
    }
    if (expected_token_count && !expected_token_ids) {
        set_error(error, error_size, "expected token IDs are required");
        return 0;
    }
    file = fopen(path, "rb");
    if (!file) {
        if (error && error_size) (void)snprintf(error, error_size,
            "cannot open sidecar %s: %s", path, strerror(errno));
        return 0;
    }
    if (
#if defined(_WIN32)
        _fseeki64(file, 0, SEEK_END) != 0
#else
        fseek(file, 0, SEEK_END) != 0
#endif
    ) {
        set_error(error, error_size, "cannot seek sidecar");
        goto cleanup;
    }
    {
#if defined(_WIN32)
        __int64 end = _ftelli64(file);
#else
        long end = ftell(file);
#endif
        if (end < 0) {
            set_error(error, error_size, "cannot determine sidecar size");
            goto cleanup;
        }
        file_size_u64 = (uint64_t)end;
    }
    if (file_size_u64 < H3CSPEED_TEXT_EMBEDDING_FILE_HEADER_SIZE ||
        file_size_u64 > H3CSPEED_TEXT_MAX_FILE ||
        file_size_u64 > (uint64_t)SIZE_MAX) {
        set_error(error, error_size, "sidecar size is invalid or too large");
        goto cleanup;
    }
    file_size = (size_t)file_size_u64;
    bytes = (unsigned char *)malloc(file_size);
    if (!bytes) {
        set_error(error, error_size, "out of memory reading sidecar");
        goto cleanup;
    }
    if (
#if defined(_WIN32)
        _fseeki64(file, 0, SEEK_SET) != 0
#else
        fseek(file, 0, SEEK_SET) != 0
#endif
        || !read_exact(file, bytes, file_size)) {
        set_error(error, error_size, "cannot read complete sidecar");
        goto cleanup;
    }
    if (memcmp(bytes, "H3CSEV01", 8) != 0) {
        set_error(error, error_size, "sidecar magic is invalid");
        goto cleanup;
    }
    version = read_u32(bytes + 8);
    header_size = read_u32(bytes + 12);
    if ((version != H3CSPEED_TEXT_VERSION_V1 && version != H3CSPEED_TEXT_VERSION_V2) ||
        header_size != H3CSPEED_TEXT_EMBEDDING_FILE_HEADER_SIZE) {
        set_error(error, error_size, "sidecar version or header size is invalid");
        goto cleanup;
    }
    if (expectation &&
        expectation->mode == H3CSPEED_TEXT_EMBEDDING_MODE_FL2VA_I2V &&
        version != H3CSPEED_TEXT_VERSION_V2) {
        set_error(error, error_size,
                  "FL2VA I2V requires a version 2 conditioning sidecar");
        goto cleanup;
    }
    prompt_bytes = read_u64(bytes + 16);
    token_count = read_u64(bytes + 24);
    recipe_bytes = read_u64(bytes + 32);
    width = read_u32(bytes + 40);
    flags = read_u32(bytes + 44);
    header_embedding_bytes = read_u64(bytes + 48);
    header_tags_bytes = read_u64(bytes + 56);
    header_token_ids_bytes = read_u64(bytes + 64);
    /* bytes 104..127 are reserved in v1. Version 2 uses a fixed, audited
     * metadata layout and still rejects every unassigned byte. */
    if (version == H3CSPEED_TEXT_VERSION_V2) {
        mode = bytes[104];
        keyframe_role = bytes[105];
        keyframe_count = bytes[106];
        keyframe_order = bytes[107];
        first_resize_policy = bytes[108];
        last_resize_policy = bytes[109];
        for (size_t index = 110; index < 112; index++) {
            if (bytes[index] != 0) {
                set_error(error, error_size, "sidecar v2 reserved bytes are non-zero");
                goto cleanup;
            }
        }
        render_width = read_u32(bytes + 112);
        render_height = read_u32(bytes + 116);
        metadata_bytes = read_u32(bytes + 120);
        for (size_t index = 124; index < H3CSPEED_TEXT_EMBEDDING_FILE_HEADER_SIZE; index++) {
            if (bytes[index] != 0) {
                set_error(error, error_size, "sidecar v2 reserved bytes are non-zero");
                goto cleanup;
            }
        }
        if (mode != H3CSPEED_TEXT_EMBEDDING_MODE_T2V &&
            mode != H3CSPEED_TEXT_EMBEDDING_MODE_FL2VA_I2V) {
            set_error(error, error_size, "sidecar v2 mode is unsupported");
            goto cleanup;
        }
        if (keyframe_role & ~(H3CSPEED_TEXT_EMBEDDING_ROLE_FIRST |
                              H3CSPEED_TEXT_EMBEDDING_ROLE_LAST)) {
            set_error(error, error_size, "sidecar v2 keyframe role is invalid");
            goto cleanup;
        }
        if (keyframe_count != (uint8_t)(
                ((keyframe_role & H3CSPEED_TEXT_EMBEDDING_ROLE_FIRST) != 0) +
                ((keyframe_role & H3CSPEED_TEXT_EMBEDDING_ROLE_LAST) != 0)) ||
            keyframe_order != keyframe_role ||
            (mode == H3CSPEED_TEXT_EMBEDDING_MODE_T2V && keyframe_role != 0) ||
            (mode == H3CSPEED_TEXT_EMBEDDING_MODE_FL2VA_I2V && keyframe_count == 0) ||
            first_resize_policy != H3CSPEED_TEXT_EMBEDDING_RESIZE_STRETCH ||
            last_resize_policy != H3CSPEED_TEXT_EMBEDDING_RESIZE_COVER ||
            (mode == H3CSPEED_TEXT_EMBEDDING_MODE_T2V &&
             (render_width != 0 || render_height != 0 || metadata_bytes != 0)) ||
            (mode == H3CSPEED_TEXT_EMBEDDING_MODE_FL2VA_I2V &&
             (render_width < 64 || render_height < 64 ||
              render_width % 32u != 0 || render_height % 32u != 0 ||
              metadata_bytes != keyframe_count * 32u))) {
            set_error(error, error_size, "sidecar v2 metadata is inconsistent");
            goto cleanup;
        }
        metadata_size = (size_t)metadata_bytes;
    } else {
        for (size_t index = 104; index < H3CSPEED_TEXT_EMBEDDING_FILE_HEADER_SIZE; index++) {
            if (bytes[index] != 0) {
                set_error(error, error_size, "sidecar reserved header bytes are non-zero");
                goto cleanup;
            }
        }
    }
    if (memcmp(bytes + 72, expected_model_sha256, 32) != 0) {
        set_error(error, error_size, "sidecar model fingerprint does not match expected SHA-256");
        goto cleanup;
    }
    if (width != H3CSPEED_TEXT_EMBEDDING_FILE_WIDTH ||
        flags != H3CSPEED_TEXT_FLAG_TAGS || token_count == 0 ||
        recipe_bytes == 0 || recipe_bytes > UINT32_C(65536)) {
        set_error(error, error_size, "sidecar dimensions or flags are invalid");
        goto cleanup;
    }
    if (!mul_u64(token_count, UINT64_C(4), &calculated_token_ids_bytes) ||
        !mul_u64(token_count, (uint64_t)width, &elements) ||
        !mul_u64(elements, UINT64_C(2), &calculated_embedding_bytes) ||
        token_count != (uint64_t)expected_token_count ||
        calculated_token_ids_bytes != header_token_ids_bytes ||
        calculated_embedding_bytes != header_embedding_bytes ||
        header_tags_bytes != token_count) {
        set_error(error, error_size, "sidecar tensor byte lengths are invalid");
        goto cleanup;
    }
    offset = H3CSPEED_TEXT_EMBEDDING_FILE_HEADER_SIZE;
    if (!add_u64(offset, prompt_bytes, &offset) ||
        !add_u64(offset, recipe_bytes, &offset) ||
        !add_u64(offset, metadata_size, &offset) ||
        !add_u64(offset, header_token_ids_bytes, &offset) ||
        !add_u64(offset, header_embedding_bytes, &offset) ||
        !add_u64(offset, header_tags_bytes, &expected_size) ||
        expected_size != file_size_u64 || prompt_bytes > (uint64_t)SIZE_MAX ||
        recipe_bytes > (uint64_t)SIZE_MAX || header_token_ids_bytes > (uint64_t)SIZE_MAX ||
        header_embedding_bytes > (uint64_t)SIZE_MAX || header_tags_bytes > (uint64_t)SIZE_MAX) {
        set_error(error, error_size, "sidecar has trailing bytes or truncated payload");
        goto cleanup;
    }
    token_ids_bytes = header_token_ids_bytes;
    embedding_bytes = header_embedding_bytes;
    tags_bytes = header_tags_bytes;
    prompt_size = (size_t)prompt_bytes;
    recipe_size = (size_t)recipe_bytes;
    token_ids_size = (size_t)token_ids_bytes;
    embedding_size = (size_t)embedding_bytes;
    tags_size = (size_t)tags_bytes;
    if (prompt_bytes != (uint64_t)strlen(expected_prompt) ||
        memcmp(bytes + H3CSPEED_TEXT_EMBEDDING_FILE_HEADER_SIZE, expected_prompt,
               prompt_size) != 0 ||
        !valid_utf8(bytes + H3CSPEED_TEXT_EMBEDDING_FILE_HEADER_SIZE,
                    prompt_size)) {
        set_error(error, error_size, "sidecar prompt does not match expected UTF-8 bytes");
        goto cleanup;
    }
    {
        const unsigned char *recipe = bytes + H3CSPEED_TEXT_EMBEDDING_FILE_HEADER_SIZE + prompt_size;
        const char *expected_recipe = version == H3CSPEED_TEXT_VERSION_V2 ?
            H3CSPEED_TEXT_EMBEDDING_FILE_RECIPE_V2 : H3CSPEED_TEXT_EMBEDDING_FILE_RECIPE;
        size_t expected_recipe_size = strlen(expected_recipe);
        if (recipe_size != expected_recipe_size ||
            memcmp(recipe, expected_recipe, expected_recipe_size) != 0 ||
            !valid_utf8(recipe, recipe_size)) {
            set_error(error, error_size, "sidecar recipe is invalid or unsupported");
            goto cleanup;
        }
    }
    {
        metadata = bytes + H3CSPEED_TEXT_EMBEDDING_FILE_HEADER_SIZE + prompt_size + recipe_size;
        const unsigned char *ids = metadata + metadata_size;
        if (version == H3CSPEED_TEXT_VERSION_V2) {
            if (expectation) {
                if (expectation->mode != mode ||
                    expectation->keyframe_role != keyframe_role ||
                    expectation->keyframe_count != keyframe_count ||
                    expectation->keyframe_order != keyframe_order ||
                    expectation->first_resize_policy != first_resize_policy ||
                    expectation->last_resize_policy != last_resize_policy ||
                    expectation->render_width != render_width ||
                    expectation->render_height != render_height) {
                    set_error(error, error_size, "sidecar v2 request metadata does not match");
                    goto cleanup;
                }
            } else if (mode != H3CSPEED_TEXT_EMBEDDING_MODE_T2V) {
                set_error(error, error_size, "I2V sidecar metadata requires an explicit expectation");
                goto cleanup;
            }
            if (keyframe_role & H3CSPEED_TEXT_EMBEDDING_ROLE_FIRST) {
                if (!expectation || !expectation->first_image_sha256 ||
                    memcmp(metadata, expectation->first_image_sha256, 32) != 0) {
                    set_error(error, error_size, "sidecar first-frame SHA-256 does not match");
                    goto cleanup;
                }
                metadata += 32;
            }
            if (keyframe_role & H3CSPEED_TEXT_EMBEDDING_ROLE_LAST) {
                if (!expectation || !expectation->last_image_sha256 ||
                    memcmp(metadata, expectation->last_image_sha256, 32) != 0) {
                    set_error(error, error_size, "sidecar last-frame SHA-256 does not match");
                    goto cleanup;
                }
            }
        }
        for (size_t index = 0; index < expected_token_count; index++) {
            uint32_t actual = read_u32(ids + index * 4u);
            if (actual != expected_token_ids[index]) {
                set_errorf(error, error_size, "sidecar token ID mismatch at index %llu",
                           (unsigned long long)index);
                goto cleanup;
            }
        }
    }
    if (elements > (uint64_t)(SIZE_MAX / sizeof(*values))) {
        set_error(error, error_size, "sidecar embedding allocation overflows size_t");
        goto cleanup;
    }
    values = (uint16_t *)malloc(embedding_size);
    tags = (uint8_t *)malloc(tags_size);
    if (!values || !tags) {
        set_error(error, error_size, "out of memory allocating sidecar embedding");
        goto cleanup;
    }
    {
        const unsigned char *payload = bytes + H3CSPEED_TEXT_EMBEDDING_FILE_HEADER_SIZE +
            prompt_size + recipe_size + metadata_size + token_ids_size;
        memcpy(values, payload, embedding_size);
        payload += embedding_size;
        for (size_t index = 0; index < tags_size; index++) {
            if (payload[index] > 1u) {
                set_error(error, error_size, "sidecar tag is not 0 or 1");
                goto cleanup;
            }
        }
        memcpy(tags, payload, tags_size);
    }
    output->tokens = (size_t)token_count;
    output->width = (size_t)width;
    output->values = values;
    output->tags = tags;
    values = NULL;
    tags = NULL;
    ok = 1;
cleanup:
    free(values);
    free(tags);
    free(bytes);
    if (file) (void)fclose(file);
    if (!ok && output) h3_text_embedding_free(output);
    return ok;
}

int h3cspeed_text_embedding_load_file(
    const char *path, const char *expected_prompt,
    const uint32_t *expected_token_ids, size_t expected_token_count,
    const uint8_t expected_model_sha256[32],
    h3_text_embedding *output, char *error, size_t error_size) {
    return h3cspeed_text_embedding_load_file_ex(
        path, expected_prompt, expected_token_ids, expected_token_count,
        expected_model_sha256, NULL, output, error, error_size);
}
