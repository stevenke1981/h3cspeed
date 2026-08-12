#ifndef H3CSPEED_SPAWN_H
#define H3CSPEED_SPAWN_H
#if defined(_WIN32)
#include <process.h>
#include <errno.h>
#include <stddef.h>
#include <stdlib.h>
typedef intptr_t pid_t;

enum h3cspeed_spawn_action_kind {
    H3CSPEED_SPAWN_DUP2 = 1,
    H3CSPEED_SPAWN_CLOSE = 2,
};
typedef struct {
    int kind;
    int descriptor;
    int target;
} h3cspeed_spawn_action;
#define H3CSPEED_SPAWN_MAX_ACTIONS 32
typedef struct {
    size_t count;
    h3cspeed_spawn_action actions[H3CSPEED_SPAWN_MAX_ACTIONS];
} posix_spawn_file_actions_t;
static inline int posix_spawn_file_actions_init(posix_spawn_file_actions_t *actions) {
    if (!actions) return EINVAL;
    actions->count = 0;
    return 0;
}
static inline int posix_spawn_file_actions_destroy(posix_spawn_file_actions_t *actions) {
    (void)actions; return 0;
}
static inline int posix_spawn_file_actions_adddup2(posix_spawn_file_actions_t *actions,
                                                    int oldfd, int newfd) {
    if (!actions || actions->count >= H3CSPEED_SPAWN_MAX_ACTIONS)
        return EINVAL;
    h3cspeed_spawn_action *action = &actions->actions[actions->count++];
    action->kind = H3CSPEED_SPAWN_DUP2;
    action->descriptor = oldfd;
    action->target = newfd;
    return 0;
}
static inline int posix_spawn_file_actions_addclose(posix_spawn_file_actions_t *actions,
                                                     int fd) {
    if (!actions || actions->count >= H3CSPEED_SPAWN_MAX_ACTIONS)
        return EINVAL;
    h3cspeed_spawn_action *action = &actions->actions[actions->count++];
    action->kind = H3CSPEED_SPAWN_CLOSE;
    action->descriptor = fd;
    action->target = -1;
    return 0;
}
int h3cspeed_posix_spawnp(pid_t *pid, const char *file,
                          const posix_spawn_file_actions_t *actions,
                          const void *attr, char *const argv[],
                          char *const envp[]);
#define posix_spawnp h3cspeed_posix_spawnp
#else
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC system_header
#endif
#include_next <spawn.h>
#endif
#endif
