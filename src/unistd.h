#ifndef H3CSPEED_UNISTD_H
#define H3CSPEED_UNISTD_H
#if defined(_WIN32)
#include <windows.h>
#include <io.h>
#include <direct.h>
#include <process.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <limits.h>

#ifdef __cplusplus
extern "C" {
#endif
int setenv(const char *name, const char *value, int overwrite);
int unsetenv(const char *name);
void arc4random_buf(void *buffer, size_t length);
char *mkdtemp(char *pattern);
int mkstemp(char *pattern);
int mkstemps(char *pattern, int suffix_length);
#ifdef __cplusplus
}
#endif

typedef int64_t ssize_t;
#define STDIN_FILENO 0
#define STDOUT_FILENO 1
#define STDERR_FILENO 2
#define F_OK 0
#define X_OK 1
#define R_OK 4
#define W_OK 2
#define _SC_AVPHYS_PAGES 1
#define _SC_PHYS_PAGES 2
#define _SC_PAGE_SIZE 3

static inline long sysconf(int name) {
    MEMORYSTATUSEX status;
    memset(&status, 0, sizeof(status));
    status.dwLength = sizeof(status);
    if (name == _SC_PAGE_SIZE) return 4096;
    if (!GlobalMemoryStatusEx(&status)) return -1;
    if (name == _SC_PHYS_PAGES) return (long)(status.ullTotalPhys / 4096);
    if (name == _SC_AVPHYS_PAGES) return (long)(status.ullAvailPhys / 4096);
    return -1;
}
static inline int usleep(unsigned usec) { Sleep((DWORD)((usec + 999) / 1000)); return 0; }
#define sleep(seconds) ((unsigned)(Sleep((DWORD)(seconds) * 1000), 0))
#define access _access
#define chdir _chdir
#define getcwd _getcwd
#define lseek _lseeki64
#define ftruncate _chsize_s
#define unlink _unlink
#define isatty _isatty
#define close _close
static inline ssize_t h3cspeed_read(int descriptor, void *buffer,
                                    size_t bytes) {
    unsigned int request = (unsigned int)(bytes > (size_t)INT_MAX ?
                                          INT_MAX : bytes);
    return (ssize_t)_read(descriptor, buffer, request);
}
static inline ssize_t h3cspeed_write(int descriptor, const void *buffer,
                                     size_t bytes) {
    unsigned int request = (unsigned int)(bytes > (size_t)INT_MAX ?
                                          INT_MAX : bytes);
    return (ssize_t)_write(descriptor, buffer, request);
}
#define read h3cspeed_read
#define write h3cspeed_write
#ifndef SSIZE_MAX
#define SSIZE_MAX INT_MAX
#endif
static inline int h3cspeed_pipe(int stream[2]) {
    return _pipe(stream, 65536, _O_BINARY | _O_NOINHERIT);
}
#define pipe h3cspeed_pipe
#else
#include_next <unistd.h>
#endif
#endif
