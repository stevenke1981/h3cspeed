#ifndef H3CSPEED_PTHREAD_H
#define H3CSPEED_PTHREAD_H
#if defined(_WIN32)
#include <windows.h>
#include <process.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>

typedef CRITICAL_SECTION pthread_mutex_t;
typedef HANDLE pthread_t;
typedef struct { int unused; } pthread_attr_t;
static inline int pthread_mutex_init(pthread_mutex_t *mutex, const void *attr) {
    (void)attr; InitializeCriticalSection(mutex); return 0;
}
static inline int pthread_mutex_destroy(pthread_mutex_t *mutex) {
    DeleteCriticalSection(mutex); return 0;
}
static inline int pthread_mutex_lock(pthread_mutex_t *mutex) {
    EnterCriticalSection(mutex); return 0;
}
static inline int pthread_mutex_unlock(pthread_mutex_t *mutex) {
    LeaveCriticalSection(mutex); return 0;
}
typedef unsigned (__stdcall *h3cspeed_thread_fn)(void *);
typedef struct {
    void *(*start)(void *);
    void *arg;
} h3cspeed_thread_context;
static unsigned __stdcall h3cspeed_thread_entry(void *raw) {
    h3cspeed_thread_context *context = (h3cspeed_thread_context *)raw;
    void *(*start)(void *) = context->start;
    void *arg = context->arg;
    free(context);
    return (unsigned)(uintptr_t)start(arg);
}
static inline int pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                                 void *(*start)(void *), void *arg) {
    (void)attr;
    h3cspeed_thread_context *context =
        (h3cspeed_thread_context *)malloc(sizeof(*context));
    if (!context) return ENOMEM;
    context->start = start;
    context->arg = arg;
    uintptr_t handle = _beginthreadex(NULL, 0, h3cspeed_thread_entry,
                                      context, 0, NULL);
    if (!handle) free(context);
    if (!handle) return errno ? errno : EAGAIN;
    *thread = (HANDLE)handle;
    return 0;
}
static inline int pthread_join(pthread_t thread, void **result) {
    (void)result;
    if (WaitForSingleObject(thread, INFINITE) != WAIT_OBJECT_0) return EINVAL;
    CloseHandle(thread); return 0;
}
#else
#include_next <pthread.h>
#endif
#endif
