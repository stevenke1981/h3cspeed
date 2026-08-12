#ifndef H3CSPEED_SYS_IOCTL_H
#define H3CSPEED_SYS_IOCTL_H
#if defined(_WIN32)
static inline int ioctl(int fd, unsigned long request, ...) { (void)fd; (void)request; return -1; }
#else
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC system_header
#endif
#include_next <sys/ioctl.h>
#endif
#endif
