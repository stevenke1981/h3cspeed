#ifndef H3CSPEED_DIRENT_H
#define H3CSPEED_DIRENT_H
#if defined(_WIN32)
#include <windows.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

struct dirent { char d_name[1024]; };
typedef struct {
    HANDLE handle;
    WIN32_FIND_DATAW data;
    struct dirent entry;
    int first;
    wchar_t *pattern;
} DIR;

static inline wchar_t *h3cspeed_find_pattern(const char *path) {
    const char *source = path ? path : ".";
    int wide_length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                           source, -1, NULL, 0);
    if (wide_length <= 0) return NULL;
    wchar_t *wide = (wchar_t *)malloc((size_t)wide_length * sizeof(*wide));
    if (!wide || MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                     source, -1, wide, wide_length) <= 0) {
        free(wide); return NULL;
    }
    DWORD full_length = GetFullPathNameW(wide, 0, NULL, NULL);
    if (!full_length) { free(wide); return NULL; }
    wchar_t *full = (wchar_t *)malloc((size_t)full_length * sizeof(*full));
    if (!full || !GetFullPathNameW(wide, full_length, full, NULL)) {
        free(wide); free(full); return NULL;
    }
    free(wide);
    int unc = full[0] == L'\\' && full[1] == L'\\';
    const wchar_t *prefix = unc ? L"\\\\?\\UNC\\" : L"\\\\?\\";
    const wchar_t *body = unc ? full + 2 : full;
    size_t needed = wcslen(prefix) + wcslen(body) + 3;
    wchar_t *pattern = (wchar_t *)malloc(needed * sizeof(*pattern));
    if (pattern)
        _snwprintf_s(pattern, needed, _TRUNCATE, L"%ls%ls\\*", prefix, body);
    free(full);
    return pattern;
}
static inline DIR *opendir(const char *path) {
    DIR *directory = (DIR *)calloc(1, sizeof(*directory));
    if (!directory) return NULL;
    directory->pattern = h3cspeed_find_pattern(path);
    if (!directory->pattern) { free(directory); errno = EINVAL; return NULL; }
    directory->handle = FindFirstFileW(directory->pattern, &directory->data);
    directory->first = directory->handle != INVALID_HANDLE_VALUE;
    if (!directory->first) {
        free(directory->pattern); free(directory); errno = ENOENT; return NULL;
    }
    return directory;
}
static inline struct dirent *readdir(DIR *directory) {
    if (!directory || directory->handle == INVALID_HANDLE_VALUE) return NULL;
    if (!directory->first && !FindNextFileW(directory->handle, &directory->data)) return NULL;
    directory->first = 0;
    if (!WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                             directory->data.cFileName, -1,
                             directory->entry.d_name,
                             (int)sizeof(directory->entry.d_name),
                             NULL, NULL)) return NULL;
    return &directory->entry;
}
static inline int closedir(DIR *directory) {
    if (!directory) return -1;
    FindClose(directory->handle); free(directory->pattern); free(directory); return 0;
}
#else
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC system_header
#endif
#include_next <dirent.h>
#endif
#endif
