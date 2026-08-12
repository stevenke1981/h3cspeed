#include "h3_tokenizer.h"

#include <unicode/uchar.h>
#include <unicode/unorm2.h>
#include <unicode/ustring.h>
#include <unicode/utf8.h>
#include <yyjson.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *key;
    uint32_t value;
    uint64_t hash;
    unsigned used;
} h3_map_entry;

typedef struct {
    h3_map_entry *entries;
    size_t capacity;
    size_t count;
} h3_map;

typedef struct {
    uint32_t value;
    size_t byte_start;
    size_t byte_length;
} h3_codepoint;

typedef struct {
    char **items;
    size_t count;
    size_t capacity;
} h3_strings;

typedef struct {
    uint32_t *items;
    size_t count;
    size_t capacity;
} h3_ids;

struct h3_tokenizer {
    h3_map vocab;
    h3_map merge_ranks;
    h3_map added_tokens;
    char **inverse_vocab;
    char **inverse_added;
    size_t inverse_count;
    char **added_alternatives;
    size_t added_count;
    char *byte_encoder[256];
    int16_t byte_decoder[324];
};

static void set_error(char *error, size_t size, const char *message) {
    if (error && size) snprintf(error, size, "%s", message ? message : "tokenizer error");
}

static char *copy_string(const char *value) {
    if (!value) return NULL;
    size_t length = strlen(value);
    char *copy = malloc(length + 1);
    if (copy) memcpy(copy, value, length + 1);
    return copy;
}

static uint64_t hash_string(const char *value) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (const unsigned char *cursor = (const unsigned char *)value; *cursor; cursor++) {
        hash ^= *cursor;
        hash *= UINT64_C(1099511628211);
    }
    return hash ? hash : 1;
}

static int map_rehash(h3_map *map, size_t capacity) {
    h3_map_entry *entries = calloc(capacity, sizeof(*entries));
    if (!entries) return 0;
    for (size_t index = 0; index < map->capacity; index++) {
        h3_map_entry entry = map->entries[index];
        if (!entry.used) continue;
        size_t slot = (size_t)entry.hash & (capacity - 1);
        while (entries[slot].used) slot = (slot + 1) & (capacity - 1);
        entries[slot] = entry;
    }
    free(map->entries);
    map->entries = entries;
    map->capacity = capacity;
    return 1;
}

static int map_reserve(h3_map *map, size_t wanted) {
    if (map->capacity && wanted * 10 < map->capacity * 7) return 1;
    size_t capacity = map->capacity ? map->capacity * 2 : 1024;
    while (wanted * 10 >= capacity * 7) {
        if (capacity > SIZE_MAX / 2) return 0;
        capacity *= 2;
    }
    return map_rehash(map, capacity);
}

static int map_put(h3_map *map, const char *key, uint32_t value) {
    if (!map_reserve(map, map->count + 1)) return 0;
    uint64_t hash = hash_string(key);
    size_t slot = (size_t)hash & (map->capacity - 1);
    while (map->entries[slot].used) {
        if (map->entries[slot].hash == hash &&
            strcmp(map->entries[slot].key, key) == 0) {
            map->entries[slot].value = value;
            return 1;
        }
        slot = (slot + 1) & (map->capacity - 1);
    }
    char *owned = copy_string(key);
    if (!owned) return 0;
    map->entries[slot] = (h3_map_entry){owned, value, hash, 1};
    map->count++;
    return 1;
}

static int map_get(const h3_map *map, const char *key, uint32_t *value) {
    if (!map || !map->capacity) return 0;
    uint64_t hash = hash_string(key);
    size_t slot = (size_t)hash & (map->capacity - 1);
    size_t start = slot;
    while (map->entries[slot].used) {
        if (map->entries[slot].hash == hash &&
            strcmp(map->entries[slot].key, key) == 0) {
            if (value) *value = map->entries[slot].value;
            return 1;
        }
        slot = (slot + 1) & (map->capacity - 1);
        if (slot == start) break;
    }
    return 0;
}

static void map_free(h3_map *map) {
    if (!map) return;
    for (size_t index = 0; index < map->capacity; index++) {
        if (map->entries[index].used) free(map->entries[index].key);
    }
    free(map->entries);
    memset(map, 0, sizeof(*map));
}

static int strings_push_owned(h3_strings *strings, char *value) {
    if (strings->count == strings->capacity) {
        size_t capacity = strings->capacity ? strings->capacity * 2 : 16;
        char **items = realloc(strings->items, capacity * sizeof(*items));
        if (!items) return 0;
        strings->items = items;
        strings->capacity = capacity;
    }
    strings->items[strings->count++] = value;
    return 1;
}

static int strings_push_copy(h3_strings *strings, const char *value) {
    char *copy = copy_string(value);
    if (!copy) return 0;
    if (!strings_push_owned(strings, copy)) {
        free(copy);
        return 0;
    }
    return 1;
}

static void strings_free(h3_strings *strings) {
    if (!strings) return;
    for (size_t index = 0; index < strings->count; index++) free(strings->items[index]);
    free(strings->items);
    memset(strings, 0, sizeof(*strings));
}

static int ids_push(h3_ids *ids, uint32_t value) {
    if (ids->count == ids->capacity) {
        size_t capacity = ids->capacity ? ids->capacity * 2 : 64;
        uint32_t *items = realloc(ids->items, capacity * sizeof(*items));
        if (!items) return 0;
        ids->items = items;
        ids->capacity = capacity;
    }
    ids->items[ids->count++] = value;
    return 1;
}

static char *utf8_from_codepoint(uint32_t value) {
    char bytes[5] = {0};
    int32_t offset = 0;
    UBool error = 0;
    U8_APPEND(bytes, offset, 4, (UChar32)value, error);
    if (error) return NULL;
    bytes[offset] = '\0';
    return copy_string(bytes);
}

static char *normalize_nfc(const char *utf8) {
    UErrorCode status = U_ZERO_ERROR;
    int32_t utf16_length = 0;
    u_strFromUTF8(NULL, 0, &utf16_length, utf8, -1, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) return NULL;
    status = U_ZERO_ERROR;
    UChar *source = malloc(((size_t)utf16_length + 1) * sizeof(*source));
    if (!source) return NULL;
    u_strFromUTF8(source, utf16_length + 1, NULL, utf8, -1, &status);
    if (U_FAILURE(status)) {
        free(source);
        return NULL;
    }
    const UNormalizer2 *normalizer = unorm2_getNFCInstance(&status);
    if (U_FAILURE(status)) {
        free(source);
        return NULL;
    }
    int32_t normalized_length = unorm2_normalize(normalizer, source, utf16_length,
                                                  NULL, 0, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
        free(source);
        return NULL;
    }
    status = U_ZERO_ERROR;
    UChar *normalized = malloc(((size_t)normalized_length + 1) * sizeof(*normalized));
    if (!normalized) {
        free(source);
        return NULL;
    }
    unorm2_normalize(normalizer, source, utf16_length, normalized,
                     normalized_length + 1, &status);
    free(source);
    if (U_FAILURE(status)) {
        free(normalized);
        return NULL;
    }
    int32_t output_length = 0;
    status = U_ZERO_ERROR;
    u_strToUTF8(NULL, 0, &output_length, normalized, normalized_length, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
        free(normalized);
        return NULL;
    }
    status = U_ZERO_ERROR;
    char *output = malloc((size_t)output_length + 1);
    if (!output) {
        free(normalized);
        return NULL;
    }
    u_strToUTF8(output, output_length + 1, NULL, normalized, normalized_length, &status);
    free(normalized);
    if (U_FAILURE(status)) {
        free(output);
        return NULL;
    }
    output[output_length] = '\0';
    return output;
}

static h3_codepoint *decode_codepoints(const char *text, size_t *count) {
    size_t length = strlen(text);
    h3_codepoint *points = calloc(length ? length : 1, sizeof(*points));
    if (!points) return NULL;
    size_t used = 0;
    int32_t offset = 0;
    while ((size_t)offset < length) {
        int32_t start = offset;
        UChar32 value = 0;
        U8_NEXT(text, offset, (int32_t)length, value);
        if (value < 0) {
            free(points);
            return NULL;
        }
        points[used++] = (h3_codepoint){(uint32_t)value, (size_t)start,
                                        (size_t)(offset - start)};
    }
    *count = used;
    return points;
}

static int is_letter(uint32_t value) {
    int8_t category = u_charType((UChar32)value);
    return category == U_UPPERCASE_LETTER || category == U_LOWERCASE_LETTER ||
           category == U_TITLECASE_LETTER || category == U_MODIFIER_LETTER ||
           category == U_OTHER_LETTER;
}

static int is_number(uint32_t value) {
    int8_t category = u_charType((UChar32)value);
    return category == U_DECIMAL_DIGIT_NUMBER || category == U_LETTER_NUMBER ||
           category == U_OTHER_NUMBER;
}

static int is_space(uint32_t value) {
    return u_isUWhiteSpace((UChar32)value) || (value >= 0x1c && value <= 0x1f);
}

static char *slice_codepoints(const char *text, const h3_codepoint *points,
                              size_t start, size_t stop) {
    if (start >= stop) return copy_string("");
    size_t begin = points[start].byte_start;
    size_t end = points[stop - 1].byte_start + points[stop - 1].byte_length;
    char *slice = malloc(end - begin + 1);
    if (!slice) return NULL;
    memcpy(slice, text + begin, end - begin);
    slice[end - begin] = '\0';
    return slice;
}

static size_t contraction(const h3_codepoint *points, size_t count, size_t index) {
    static const char *values[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    if (points[index].value != '\'') return 0;
    for (size_t item = 0; item < sizeof(values) / sizeof(values[0]); item++) {
        size_t length = strlen(values[item]);
        if (index + length > count) continue;
        int matches = 1;
        for (size_t offset = 1; offset < length; offset++) {
            uint32_t got = points[index + offset].value;
            if (got >= 'A' && got <= 'Z') got += 'a' - 'A';
            if (got != (unsigned char)values[item][offset]) matches = 0;
        }
        if (matches) return length;
    }
    return 0;
}

static int pretokenize(const char *input, h3_strings *pieces) {
    char *text = normalize_nfc(input);
    if (!text) return 0;
    size_t count = 0;
    h3_codepoint *points = decode_codepoints(text, &count);
    if (!points) {
        free(text);
        return 0;
    }
    size_t index = 0;
    while (index < count) {
        size_t matched = contraction(points, count, index);
        if (matched) {
            char *piece = slice_codepoints(text, points, index, index + matched);
            if (!piece || !strings_push_owned(pieces, piece)) goto fail;
            index += matched;
            continue;
        }
        uint32_t value = points[index].value;
        ptrdiff_t letter_start = (ptrdiff_t)index;
        if (is_letter(value)) {
            /* Already at the first letter. */
        } else if (value != '\r' && value != '\n' && !is_number(value) &&
                   index + 1 < count && is_letter(points[index + 1].value)) {
            letter_start++;
        } else {
            letter_start = -1;
        }
        if (letter_start >= 0) {
            size_t stop = (size_t)letter_start;
            while (stop < count && is_letter(points[stop].value)) stop++;
            char *piece = slice_codepoints(text, points, index, stop);
            if (!piece || !strings_push_owned(pieces, piece)) goto fail;
            index = stop;
            continue;
        }
        if (is_number(value)) {
            char *piece = slice_codepoints(text, points, index, index + 1);
            if (!piece || !strings_push_owned(pieces, piece)) goto fail;
            index++;
            continue;
        }
        size_t punctuation_start = index +
            (value == ' ' && index + 1 < count &&
             !is_space(points[index + 1].value) &&
             !is_letter(points[index + 1].value) &&
             !is_number(points[index + 1].value));
        size_t stop = punctuation_start;
        while (stop < count && !is_space(points[stop].value) &&
               !is_letter(points[stop].value) && !is_number(points[stop].value)) stop++;
        if (stop > punctuation_start) {
            while (stop < count &&
                   (points[stop].value == '\r' || points[stop].value == '\n')) stop++;
            char *piece = slice_codepoints(text, points, index, stop);
            if (!piece || !strings_push_owned(pieces, piece)) goto fail;
            index = stop;
            continue;
        }
        if (is_space(value)) {
            size_t whitespace_end = index + 1;
            while (whitespace_end < count && is_space(points[whitespace_end].value)) {
                whitespace_end++;
            }
            ptrdiff_t newline_end = -1;
            for (size_t cursor = index; cursor < whitespace_end; cursor++) {
                if (points[cursor].value == '\r' || points[cursor].value == '\n') {
                    newline_end = (ptrdiff_t)cursor + 1;
                }
            }
            size_t piece_end;
            if (newline_end >= 0) piece_end = (size_t)newline_end;
            else if (whitespace_end == count) piece_end = whitespace_end;
            else if (whitespace_end - index > 1) piece_end = whitespace_end - 1;
            else piece_end = index + 1;
            char *piece = slice_codepoints(text, points, index, piece_end);
            if (!piece || !strings_push_owned(pieces, piece)) goto fail;
            index = piece_end;
            continue;
        }
        goto fail;
    }
    free(points);
    free(text);
    return 1;
fail:
    free(points);
    free(text);
    return 0;
}

static char *pair_key(const char *left, const char *right) {
    static const char separator[] = "\xef\xbf\xbf"; /* U+FFFF, as upstream. */
    size_t left_length = strlen(left);
    size_t right_length = strlen(right);
    size_t separator_length = sizeof(separator) - 1;
    if (left_length > SIZE_MAX - right_length - separator_length - 1) return NULL;
    char *key = malloc(left_length + separator_length + right_length + 1);
    if (!key) return NULL;
    memcpy(key, left, left_length);
    memcpy(key + left_length, separator, separator_length);
    memcpy(key + left_length + separator_length, right, right_length + 1);
    return key;
}

static int bpe_piece(const h3_tokenizer *tokenizer, const char *piece, h3_ids *output,
                     char *error, size_t error_size) {
    h3_strings symbols = {0};
    const unsigned char *bytes = (const unsigned char *)piece;
    for (size_t index = 0; bytes[index]; index++) {
        if (!strings_push_copy(&symbols, tokenizer->byte_encoder[bytes[index]])) goto oom;
    }
    while (symbols.count > 1) {
        uint32_t best_rank = UINT32_MAX;
        size_t best = 0;
        int found = 0;
        for (size_t index = 0; index + 1 < symbols.count; index++) {
            char *key = pair_key(symbols.items[index], symbols.items[index + 1]);
            if (!key) goto oom;
            uint32_t rank = 0;
            int present = map_get(&tokenizer->merge_ranks, key, &rank);
            free(key);
            if (present && (!found || rank < best_rank)) {
                found = 1;
                best_rank = rank;
                best = index;
            }
        }
        if (!found) break;
        const char *left = symbols.items[best];
        const char *right = symbols.items[best + 1];
        h3_strings merged = {0};
        for (size_t index = 0; index < symbols.count;) {
            if (index + 1 < symbols.count && strcmp(symbols.items[index], left) == 0 &&
                strcmp(symbols.items[index + 1], right) == 0) {
                size_t left_length = strlen(symbols.items[index]);
                size_t right_length = strlen(symbols.items[index + 1]);
                char *combined = malloc(left_length + right_length + 1);
                if (!combined) {
                    strings_free(&merged);
                    goto oom;
                }
                memcpy(combined, symbols.items[index], left_length);
                memcpy(combined + left_length, symbols.items[index + 1], right_length + 1);
                if (!strings_push_owned(&merged, combined)) {
                    free(combined);
                    strings_free(&merged);
                    goto oom;
                }
                index += 2;
            } else {
                if (!strings_push_copy(&merged, symbols.items[index])) {
                    strings_free(&merged);
                    goto oom;
                }
                index++;
            }
        }
        strings_free(&symbols);
        symbols = merged;
    }
    for (size_t index = 0; index < symbols.count; index++) {
        uint32_t identifier = 0;
        if (!map_get(&tokenizer->vocab, symbols.items[index], &identifier)) {
            set_error(error, error_size, "BPE symbol is absent from vocabulary");
            strings_free(&symbols);
            return 0;
        }
        if (!ids_push(output, identifier)) goto oom;
    }
    strings_free(&symbols);
    return 1;
oom:
    strings_free(&symbols);
    set_error(error, error_size, "out of memory during BPE");
    return 0;
}

static int encode_plain(const h3_tokenizer *tokenizer, const char *text, h3_ids *output,
                        char *error, size_t error_size) {
    h3_strings pieces = {0};
    if (!pretokenize(text, &pieces)) {
        set_error(error, error_size, "unable to pre-tokenize input");
        return 0;
    }
    for (size_t index = 0; index < pieces.count; index++) {
        if (!bpe_piece(tokenizer, pieces.items[index], output, error, error_size)) {
            strings_free(&pieces);
            return 0;
        }
    }
    strings_free(&pieces);
    return 1;
}

static int compare_added(const void *left_pointer, const void *right_pointer) {
    const char *left = *(const char *const *)left_pointer;
    const char *right = *(const char *const *)right_pointer;
    size_t left_length = strlen(left);
    size_t right_length = strlen(right);
    if (left_length > right_length) return -1;
    if (left_length < right_length) return 1;
    return strcmp(left, right);
}

static int append_inverse(char ***array, size_t *capacity, uint32_t identifier,
                          const char *value) {
    if ((size_t)identifier >= *capacity) {
        size_t next = *capacity ? *capacity : 1024;
        while (next <= (size_t)identifier) {
            if (next > SIZE_MAX / 2) return 0;
            next *= 2;
        }
        char **grown = realloc(*array, next * sizeof(**array));
        if (!grown) return 0;
        memset(grown + *capacity, 0, (next - *capacity) * sizeof(*grown));
        *array = grown;
        *capacity = next;
    }
    free((*array)[identifier]);
    (*array)[identifier] = copy_string(value);
    return (*array)[identifier] != NULL;
}

static int parse_tokenizer(h3_tokenizer *tokenizer, yyjson_val *root,
                           char *error, size_t error_size) {
    yyjson_val *model = yyjson_obj_get(root, "model");
    yyjson_val *normalizer = yyjson_obj_get(root, "normalizer");
    yyjson_val *type = model ? yyjson_obj_get(model, "type") : NULL;
    yyjson_val *unk = model ? yyjson_obj_get(model, "unk_token") : NULL;
    yyjson_val *normalizer_type = normalizer ? yyjson_obj_get(normalizer, "type") : NULL;
    yyjson_val *vocab = model ? yyjson_obj_get(model, "vocab") : NULL;
    yyjson_val *merges = model ? yyjson_obj_get(model, "merges") : NULL;
    if (!yyjson_is_obj(model) || !yyjson_is_str(type) ||
        strcmp(yyjson_get_str(type), "BPE") != 0 ||
        (unk && !yyjson_is_null(unk)) || !yyjson_is_str(normalizer_type) ||
        strcmp(yyjson_get_str(normalizer_type), "NFC") != 0 ||
        !yyjson_is_obj(vocab) || !yyjson_is_arr(merges)) {
        set_error(error, error_size, "unexpected tokenizer specification");
        return 0;
    }

    char **inverse_vocab = NULL;
    char **inverse_added = NULL;
    size_t inverse_vocab_capacity = 0;
    size_t inverse_added_capacity = 0;
    uint32_t maximum_id = 0;

    yyjson_obj_iter vocab_iter = yyjson_obj_iter_with(vocab);
    yyjson_val *key = NULL;
    while ((key = yyjson_obj_iter_next(&vocab_iter))) {
        yyjson_val *value = yyjson_obj_iter_get_val(key);
        if (!yyjson_is_str(key) || !yyjson_is_uint(value) || yyjson_get_uint(value) > UINT32_MAX) {
            set_error(error, error_size, "invalid tokenizer vocabulary");
            goto fail;
        }
        const char *symbol = yyjson_get_str(key);
        uint32_t identifier = (uint32_t)yyjson_get_uint(value);
        if (!map_put(&tokenizer->vocab, symbol, identifier) ||
            !append_inverse(&inverse_vocab, &inverse_vocab_capacity, identifier, symbol)) {
            set_error(error, error_size, "out of memory loading vocabulary");
            goto fail;
        }
        if (identifier > maximum_id) maximum_id = identifier;
    }

    uint32_t rank = 0;
    yyjson_arr_iter merge_iter = yyjson_arr_iter_with(merges);
    yyjson_val *entry = NULL;
    while ((entry = yyjson_arr_iter_next(&merge_iter))) {
        const char *left = NULL;
        const char *right = NULL;
        char *owned_left = NULL;
        if (yyjson_is_str(entry)) {
            const char *text = yyjson_get_str(entry);
            const char *separator = strchr(text, ' ');
            if (!separator) {
                set_error(error, error_size, "invalid tokenizer merge");
                goto fail;
            }
            size_t length = (size_t)(separator - text);
            owned_left = malloc(length + 1);
            if (!owned_left) goto oom;
            memcpy(owned_left, text, length);
            owned_left[length] = '\0';
            left = owned_left;
            right = separator + 1;
        } else if (yyjson_is_arr(entry) && yyjson_arr_size(entry) == 2) {
            yyjson_val *first = yyjson_arr_get_first(entry);
            yyjson_val *second = yyjson_arr_get(entry, 1);
            if (!yyjson_is_str(first) || !yyjson_is_str(second)) {
                set_error(error, error_size, "invalid tokenizer merge pair");
                goto fail;
            }
            left = yyjson_get_str(first);
            right = yyjson_get_str(second);
        } else {
            set_error(error, error_size, "invalid tokenizer merge");
            goto fail;
        }
        char *joined = pair_key(left, right);
        free(owned_left);
        if (!joined || !map_put(&tokenizer->merge_ranks, joined, rank++)) {
            free(joined);
            goto oom;
        }
        free(joined);
    }

    yyjson_val *added = yyjson_obj_get(root, "added_tokens");
    if (added && !yyjson_is_arr(added)) {
        set_error(error, error_size, "invalid added_tokens array");
        goto fail;
    }
    size_t added_count = added ? yyjson_arr_size(added) : 0;
    tokenizer->added_alternatives = calloc(added_count ? added_count : 1,
                                            sizeof(*tokenizer->added_alternatives));
    if (!tokenizer->added_alternatives) goto oom;
    yyjson_arr_iter added_iter;
    memset(&added_iter, 0, sizeof(added_iter));
    if (added) added_iter = yyjson_arr_iter_with(added);
    while (added && (entry = yyjson_arr_iter_next(&added_iter))) {
        yyjson_val *content = yyjson_obj_get(entry, "content");
        yyjson_val *identifier_value = yyjson_obj_get(entry, "id");
        yyjson_val *single_word = yyjson_obj_get(entry, "single_word");
        yyjson_val *lstrip = yyjson_obj_get(entry, "lstrip");
        yyjson_val *rstrip = yyjson_obj_get(entry, "rstrip");
        yyjson_val *normalized = yyjson_obj_get(entry, "normalized");
        if (!yyjson_is_str(content) || !yyjson_is_uint(identifier_value) ||
            yyjson_get_uint(identifier_value) > UINT32_MAX ||
            (single_word && yyjson_get_bool(single_word)) ||
            (lstrip && yyjson_get_bool(lstrip)) ||
            (rstrip && yyjson_get_bool(rstrip)) ||
            (normalized && yyjson_get_bool(normalized))) {
            set_error(error, error_size, "unsupported added-token policy");
            goto fail;
        }
        const char *text = yyjson_get_str(content);
        uint32_t identifier = (uint32_t)yyjson_get_uint(identifier_value);
        if (!map_put(&tokenizer->added_tokens, text, identifier) ||
            !append_inverse(&inverse_added, &inverse_added_capacity, identifier, text)) {
            goto oom;
        }
        tokenizer->added_alternatives[tokenizer->added_count] = copy_string(text);
        if (!tokenizer->added_alternatives[tokenizer->added_count]) goto oom;
        tokenizer->added_count++;
        if (identifier > maximum_id) maximum_id = identifier;
    }
    qsort(tokenizer->added_alternatives, tokenizer->added_count,
          sizeof(*tokenizer->added_alternatives), compare_added);

    tokenizer->inverse_count = (size_t)maximum_id + 1;
    tokenizer->inverse_vocab = calloc(tokenizer->inverse_count, sizeof(*tokenizer->inverse_vocab));
    tokenizer->inverse_added = calloc(tokenizer->inverse_count, sizeof(*tokenizer->inverse_added));
    if (!tokenizer->inverse_vocab || !tokenizer->inverse_added) goto oom;
    for (size_t index = 0; index < inverse_vocab_capacity && index < tokenizer->inverse_count; index++) {
        tokenizer->inverse_vocab[index] = inverse_vocab[index];
        inverse_vocab[index] = NULL;
    }
    for (size_t index = 0; index < inverse_added_capacity && index < tokenizer->inverse_count; index++) {
        tokenizer->inverse_added[index] = inverse_added[index];
        inverse_added[index] = NULL;
    }
    for (size_t index = 0; index < inverse_vocab_capacity; index++) free(inverse_vocab[index]);
    for (size_t index = 0; index < inverse_added_capacity; index++) free(inverse_added[index]);
    free(inverse_vocab);
    free(inverse_added);
    return 1;
oom:
    set_error(error, error_size, "out of memory loading tokenizer");
fail:
    if (inverse_vocab) {
        for (size_t index = 0; index < inverse_vocab_capacity; index++) free(inverse_vocab[index]);
    }
    if (inverse_added) {
        for (size_t index = 0; index < inverse_added_capacity; index++) free(inverse_added[index]);
    }
    free(inverse_vocab);
    free(inverse_added);
    return 0;
}

h3_tokenizer *h3_tokenizer_load(const char *path, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path) {
        set_error(error, error_size, "tokenizer path is required");
        return NULL;
    }
    yyjson_read_err json_error;
    yyjson_doc *document = yyjson_read_file(path, 0, NULL, &json_error);
    if (!document) {
        char message[256];
        snprintf(message, sizeof(message), "cannot read tokenizer: %s", json_error.msg);
        set_error(error, error_size, message);
        return NULL;
    }
    yyjson_val *root = yyjson_doc_get_root(document);
    h3_tokenizer *tokenizer = calloc(1, sizeof(*tokenizer));
    if (!tokenizer) {
        yyjson_doc_free(document);
        set_error(error, error_size, "out of memory loading tokenizer");
        return NULL;
    }
    for (size_t index = 0; index < 324; index++) tokenizer->byte_decoder[index] = -1;
    unsigned extra = 0;
    for (unsigned byte = 0; byte < 256; byte++) {
        int visible = (byte >= '!' && byte <= '~') ||
                      (byte >= 0xa1 && byte <= 0xac) ||
                      (byte >= 0xae && byte <= 0xff);
        uint32_t codepoint = visible ? byte : 256 + extra++;
        tokenizer->byte_encoder[byte] = utf8_from_codepoint(codepoint);
        if (!tokenizer->byte_encoder[byte]) {
            set_error(error, error_size, "out of memory building byte tokenizer");
            yyjson_doc_free(document);
            h3_tokenizer_free(tokenizer);
            return NULL;
        }
        tokenizer->byte_decoder[codepoint] = (int16_t)byte;
    }
    if (!parse_tokenizer(tokenizer, root, error, error_size)) {
        yyjson_doc_free(document);
        h3_tokenizer_free(tokenizer);
        return NULL;
    }
    yyjson_doc_free(document);
    return tokenizer;
}

void h3_tokenizer_free(h3_tokenizer *tokenizer) {
    if (!tokenizer) return;
    map_free(&tokenizer->vocab);
    map_free(&tokenizer->merge_ranks);
    map_free(&tokenizer->added_tokens);
    for (size_t index = 0; index < tokenizer->inverse_count; index++) {
        free(tokenizer->inverse_vocab ? tokenizer->inverse_vocab[index] : NULL);
        free(tokenizer->inverse_added ? tokenizer->inverse_added[index] : NULL);
    }
    free(tokenizer->inverse_vocab);
    free(tokenizer->inverse_added);
    for (size_t index = 0; index < tokenizer->added_count; index++) {
        free(tokenizer->added_alternatives[index]);
    }
    free(tokenizer->added_alternatives);
    for (size_t index = 0; index < 256; index++) free(tokenizer->byte_encoder[index]);
    free(tokenizer);
}

static int added_match(const h3_tokenizer *tokenizer, const char *text,
                       size_t start, size_t *location, const char **token) {
    int found = 0;
    for (size_t index = 0; index < tokenizer->added_count; index++) {
        const char *candidate = tokenizer->added_alternatives[index];
        const char *match = strstr(text + start, candidate);
        if (!match) continue;
        size_t candidate_location = (size_t)(match - text);
        if (!found || candidate_location < *location ||
            (candidate_location == *location && strlen(candidate) > strlen(*token))) {
            *location = candidate_location;
            *token = candidate;
            found = 1;
        }
    }
    return found;
}

int h3_tokenizer_encode(const h3_tokenizer *tokenizer, const char *utf8,
                        int pad_empty, uint32_t **ids, size_t *count,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!tokenizer || !utf8 || !ids || !count) return 0;
    *ids = NULL;
    *count = 0;
    /* ICU rejects malformed UTF-8 during NFC normalization. */
    char *validation = normalize_nfc(utf8);
    if (!validation) {
        set_error(error, error_size, "prompt is not valid UTF-8");
        return 0;
    }
    free(validation);

    h3_ids output = {0};
    size_t length = strlen(utf8);
    size_t start = 0;
    while (start < length) {
        size_t location = SIZE_MAX;
        const char *added = NULL;
        if (!added_match(tokenizer, utf8, start, &location, &added)) break;
        if (location > start) {
            char *plain = malloc(location - start + 1);
            if (!plain) goto oom;
            memcpy(plain, utf8 + start, location - start);
            plain[location - start] = '\0';
            int ok = encode_plain(tokenizer, plain, &output, error, error_size);
            free(plain);
            if (!ok) goto fail;
        }
        uint32_t identifier = 0;
        if (!map_get(&tokenizer->added_tokens, added, &identifier) ||
            !ids_push(&output, identifier)) goto oom;
        start = location + strlen(added);
    }
    if (start < length) {
        if (!encode_plain(tokenizer, utf8 + start, &output, error, error_size)) goto fail;
    }
    if (output.count == 0 && pad_empty && !ids_push(&output, H3_PAD_TOKEN_ID)) goto oom;
    *ids = output.items;
    *count = output.count;
    return 1;
oom:
    set_error(error, error_size, "out of memory encoding prompt");
fail:
    free(output.items);
    return 0;
}

void h3_tokenizer_ids_free(uint32_t *ids) {
    free(ids);
}

typedef struct {
    unsigned char *values;
    size_t count;
    size_t capacity;
} byte_buffer;

static int bytes_append(byte_buffer *buffer, const void *values, size_t count) {
    if (count > SIZE_MAX - buffer->count) return 0;
    size_t wanted = buffer->count + count;
    if (wanted > buffer->capacity) {
        size_t capacity = buffer->capacity ? buffer->capacity * 2 : 256;
        while (capacity < wanted) {
            if (capacity > SIZE_MAX / 2) return 0;
            capacity *= 2;
        }
        unsigned char *grown = realloc(buffer->values, capacity);
        if (!grown) return 0;
        buffer->values = grown;
        buffer->capacity = capacity;
    }
    memcpy(buffer->values + buffer->count, values, count);
    buffer->count += count;
    return 1;
}

char *h3_tokenizer_decode(const h3_tokenizer *tokenizer,
                          const uint32_t *ids, size_t count,
                          char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!tokenizer || (!ids && count)) return NULL;
    byte_buffer result = {0};
    byte_buffer pending = {0};
    for (size_t index = 0; index < count; index++) {
        uint32_t identifier = ids[index];
        if ((size_t)identifier >= tokenizer->inverse_count) {
            set_error(error, error_size, "token ID is out of range");
            goto fail;
        }
        const char *added = tokenizer->inverse_added[identifier];
        if (added) {
            if (pending.count && !bytes_append(&result, pending.values, pending.count)) goto oom;
            pending.count = 0;
            if (!bytes_append(&result, added, strlen(added))) goto oom;
            continue;
        }
        const char *symbol = tokenizer->inverse_vocab[identifier];
        if (!symbol) {
            set_error(error, error_size, "unknown token ID");
            goto fail;
        }
        int32_t offset = 0;
        int32_t length = (int32_t)strlen(symbol);
        while (offset < length) {
            UChar32 codepoint = 0;
            U8_NEXT(symbol, offset, length, codepoint);
            if (codepoint < 0 || codepoint >= 324 ||
                tokenizer->byte_decoder[codepoint] < 0) {
                set_error(error, error_size, "invalid byte-level token");
                goto fail;
            }
            unsigned char byte = (unsigned char)tokenizer->byte_decoder[codepoint];
            if (!bytes_append(&pending, &byte, 1)) goto oom;
        }
    }
    if (pending.count && !bytes_append(&result, pending.values, pending.count)) goto oom;
    if (!bytes_append(&result, "", 1)) goto oom;
    free(pending.values);
    return (char *)result.values;
oom:
    set_error(error, error_size, "out of memory decoding tokens");
fail:
    free(result.values);
    free(pending.values);
    return NULL;
}
