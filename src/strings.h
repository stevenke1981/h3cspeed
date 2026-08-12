#ifndef H3CSPEED_STRINGS_H
#define H3CSPEED_STRINGS_H
#if defined(_WIN32)
#include <string.h>
#include <ctype.h>
static inline int h3cspeed_strcasecmp(const char *left, const char *right) {
    while (*left && *right) {
        int a = tolower((unsigned char)*left++);
        int b = tolower((unsigned char)*right++);
        if (a != b) return a - b;
    }
    return (unsigned char)*left - (unsigned char)*right;
}
static inline int h3cspeed_strncasecmp(const char *left, const char *right, size_t count) {
    while (count > 0 && *left && *right) {
        int a = tolower((unsigned char)*left++);
        int b = tolower((unsigned char)*right++);
        if (a != b) return a - b;
        count--;
    }
    if (count == 0) return 0;
    return (unsigned char)*left - (unsigned char)*right;
}
#define strcasecmp h3cspeed_strcasecmp
#define strncasecmp h3cspeed_strncasecmp
#else
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC system_header
#endif
#include_next <strings.h>
#endif
#endif
