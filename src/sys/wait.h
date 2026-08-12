#ifndef H3CSPEED_SYS_WAIT_H
#define H3CSPEED_SYS_WAIT_H
#if defined(_WIN32)
#include <windows.h>
#include <errno.h>
static inline intptr_t h3cspeed_waitpid(intptr_t pid, int *status, int options) {
    (void)options;
    HANDLE process = (HANDLE)pid;
    DWORD value = 0;
    if (!process || WaitForSingleObject(process, INFINITE) != WAIT_OBJECT_0 ||
        !GetExitCodeProcess(process, &value)) {
        errno = ECHILD;
        return -1;
    }
    CloseHandle(process);
    if (status) *status = ((int)value & 0xff) << 8;
    return pid;
}
#define waitpid h3cspeed_waitpid
#define WIFEXITED(status) 1
#define WEXITSTATUS(status) (((status) >> 8) & 0xff)
#else
#include_next <sys/wait.h>
#endif
#endif
