#ifndef H3CSPEED_MSVC_COMPAT_H
#define H3CSPEED_MSVC_COMPAT_H

/* Small compile-time compatibility surface for the pinned POSIX-oriented H3
 * host sources.  Runtime media process/terminal paths remain opt-in; CUDA
 * inference and the public C API are unchanged. */
#if defined(_WIN32)
#ifndef _CRT_RAND_S
#define _CRT_RAND_S
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <signal.h>
#include <sys/stat.h>
#include <io.h>
#include <direct.h>
#include <malloc.h>
#include <errno.h>
#include <fcntl.h>

#if defined(_MSC_VER) && !defined(__clang__) && !defined(__CUDACC__)
#define __attribute__(...)
#endif
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS 1
#endif
#ifndef S_ISREG
#define S_ISREG(mode) (((mode) & _S_IFMT) == _S_IFREG)
#endif
#ifndef S_ISDIR
#define S_ISDIR(mode) (((mode) & _S_IFMT) == _S_IFDIR)
#endif
#ifndef strdup
#define strdup _strdup
#endif
#ifndef fileno
#define fileno _fileno
#endif
#ifndef open
/* The pinned safetensors readers use POSIX open() for binary payloads.  The
 * MSVC CRT defaults descriptors to text mode, where 0x1a is treated as EOF
 * and large weight reads can truncate silently. */
#define open(path, flags) _open(path, (flags) | _O_BINARY)
#endif
#ifndef mkdir
#define mkdir(path, mode) _mkdir(path)
#endif
#ifndef fchmod
#define fchmod(fd, mode) _chmod(fd, mode)
#endif
#ifndef O_CLOEXEC
#define O_CLOEXEC _O_NOINHERIT
#endif
static inline int h3cspeed_posix_memalign(void **memory, size_t alignment,
                                          size_t size) {
    if (!memory) return EINVAL;
    if (alignment < sizeof(void *) || (alignment & (alignment - 1)) != 0)
        return EINVAL;
    *memory = _aligned_malloc(size, alignment);
    return *memory ? 0 : ENOMEM;
}
#define posix_memalign h3cspeed_posix_memalign
/* The implementation lives in h3_windows_posix.c so all host objects share
 * one external symbol rather than each object receiving a private cursor shim. */
#ifdef __cplusplus
extern "C" {
#endif
int64_t pread(int fd, void *buffer, size_t bytes, int64_t offset);
#ifdef __cplusplus
}
#endif
/* MSVC's plain struct stat stores st_size in 32 bits.  H3 safetensors are
 * routinely tens of GiB, so force both the type and calls onto the 64-bit CRT
 * ABI before the pinned sources are parsed. */
#define stat _stat64
#define fstat _fstat64
#ifndef SIGPIPE
#define SIGPIPE SIGABRT
#endif
#ifndef H3CSPEED_SIGACTION_DEFINED
#define H3CSPEED_SIGACTION_DEFINED 1
typedef void (*h3cspeed_sighandler_t)(int);
struct h3cspeed_sigaction {
    h3cspeed_sighandler_t sa_handler;
    unsigned long sa_mask;
    int sa_flags;
};
static inline int h3cspeed_sigemptyset(unsigned long *mask) {
    if (mask) *mask = 0; return 0;
}
static inline int h3cspeed_sigaction(int signal_number,
                                     const struct h3cspeed_sigaction *action,
                                     struct h3cspeed_sigaction *old_action) {
    (void)signal_number; (void)action; (void)old_action; return 0;
}
#define sigaction h3cspeed_sigaction
#define sigemptyset h3cspeed_sigemptyset
#endif

#define CLOCK_MONOTONIC 1
static inline int h3cspeed_clock_gettime(int ignored, struct timespec *value) {
    (void)ignored;
    if (!value) return -1;
    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    if (!QueryPerformanceFrequency(&frequency) ||
        !QueryPerformanceCounter(&counter) || frequency.QuadPart <= 0) {
        return -1;
    }
    value->tv_sec = (time_t)(counter.QuadPart / frequency.QuadPart);
    value->tv_nsec = (long)(((counter.QuadPart % frequency.QuadPart) *
                             INT64_C(1000000000)) / frequency.QuadPart);
    return 0;
}
#define clock_gettime h3cspeed_clock_gettime
#endif

#endif
