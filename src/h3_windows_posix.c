#if defined(_WIN32)
#include <errno.h>
#include <fcntl.h>
#include <io.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <windows.h>

#include "spawn.h"

int setenv(const char *name, const char *value, int overwrite) {
    if (!name || !*name || (!overwrite && getenv(name))) return 0;
    return _putenv_s(name, value ? value : "");
}

int unsetenv(const char *name) { return _putenv_s(name, ""); }

void arc4random_buf(void *buffer, size_t length) {
    unsigned char *bytes = (unsigned char *)buffer;
    for (size_t index = 0; index < length; index++) {
        unsigned int value = 0;
        if (rand_s(&value) != 0) value = (unsigned int)rand();
        bytes[index] = (unsigned char)value;
    }
}

static int h3cspeed_make_temp(char *pattern, size_t suffix_length,
                              int directory) {
    if (strncmp(pattern, "/tmp/", 5) == 0 ||
        strncmp(pattern, "\\tmp\\", 5) == 0) {
        /* The pinned POSIX templates use /tmp and some are exact-size arrays.
         * A shorter relative prefix preserves their storage contract while
         * avoiding writes to the Windows drive root. */
        memmove(pattern + 2, pattern + 5, strlen(pattern + 5) + 1);
        pattern[0] = '.';
        pattern[1] = '\\';
    }
    size_t length = strlen(pattern);
    if (length < 6 + suffix_length) { errno = EINVAL; return -1; }
    char *prefix = pattern + length - suffix_length - 6;
    for (int attempt = 0; attempt < 100; attempt++) {
        unsigned int value = 0;
        if (rand_s(&value) != 0) value = (unsigned int)rand();
        for (size_t index = 0; index < 6; index++) {
            prefix[index] = (char)('a' + (value % 26)); value /= 26;
        }
        if (!directory) {
            int fd = _open(pattern, _O_CREAT | _O_EXCL | _O_RDWR | _O_BINARY,
                           _S_IREAD | _S_IWRITE);
            if (fd >= 0) return fd;
        } else {
            if (_mkdir(pattern) == 0) return 0;
        }
    }
    return -1;
}

char *mkdtemp(char *pattern) {
    if (h3cspeed_make_temp(pattern, 0, 1) < 0) return NULL;
    return pattern;
}
int mkstemp(char *pattern) { return h3cspeed_make_temp(pattern, 0, 0); }
int mkstemps(char *pattern, int suffix_length) {
    if (suffix_length < 0) { errno = EINVAL; return -1; }
    return h3cspeed_make_temp(pattern, (size_t)suffix_length, 0);
}

static SRWLOCK h3cspeed_pread_lock = SRWLOCK_INIT;

int64_t pread(int fd, void *buffer, size_t bytes, int64_t offset) {
    if (fd < 0 || (!buffer && bytes) || offset < 0) {
        errno = EINVAL;
        return -1;
    }
    unsigned int request = (unsigned int)(bytes > UINT_MAX ? UINT_MAX : bytes);
    HANDLE original = (HANDLE)_get_osfhandle(fd);
    if (original == INVALID_HANDLE_VALUE) {
        errno = EBADF;
        return -1;
    }
    /* ReOpenFile gives this read an independent cursor, preserving POSIX
     * pread semantics even when offload workers read the same tensor file.
     * dwFlags may contain only FILE_FLAG_* bits; FILE_ATTRIBUTE_NORMAL is
     * invalid here and forced the serialized seek/_read fallback. Keep
     * SEQUENTIAL_SCAN so the OS page cache does not double-buffer weights
     * that the explicit host RAM cache already retains. */
    HANDLE reopened = ReOpenFile(original, GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        FILE_FLAG_SEQUENTIAL_SCAN);
    if (reopened != INVALID_HANDLE_VALUE) {
        LARGE_INTEGER position;
        position.QuadPart = offset;
        DWORD received = 0;
        BOOL ok = SetFilePointerEx(reopened, position, NULL, FILE_BEGIN) &&
                  ReadFile(reopened, buffer, request, &received, NULL);
        DWORD error = ok ? ERROR_SUCCESS : GetLastError();
        CloseHandle(reopened);
        if (!ok) {
            (void)error;
            errno = EIO;
            return -1;
        }
        return (int64_t)received;
    }

    /* Non-disk handles cannot be reopened.  Serialize the cursor fallback so
     * callers still observe atomic position-preserving reads. */
    AcquireSRWLockExclusive(&h3cspeed_pread_lock);
    int64_t current = _lseeki64(fd, 0, SEEK_CUR);
    int result = -1;
    if (current >= 0 && _lseeki64(fd, offset, SEEK_SET) >= 0) {
        result = _read(fd, buffer, request);
        (void)_lseeki64(fd, current, SEEK_SET);
    }
    ReleaseSRWLockExclusive(&h3cspeed_pread_lock);
    return result;
}

static void __cdecl h3cspeed_ignore_invalid_parameter(
        const wchar_t *expression, const wchar_t *function,
        const wchar_t *file, unsigned int line, uintptr_t reserved) {
    (void)expression; (void)function; (void)file; (void)line; (void)reserved;
}

static intptr_t h3cspeed_os_handle(int descriptor) {
    _invalid_parameter_handler previous =
        _set_thread_local_invalid_parameter_handler(
            h3cspeed_ignore_invalid_parameter);
    intptr_t raw = _get_osfhandle(descriptor);
    (void)_set_thread_local_invalid_parameter_handler(previous);
    return raw;
}

static int h3cspeed_append_wchar(wchar_t **text, size_t *length,
                                 size_t *capacity, wchar_t value) {
    if (*length + 1 >= *capacity) {
        size_t next = *capacity ? *capacity * 2 : 256;
        wchar_t *grown = (wchar_t *)realloc(*text, next * sizeof(**text));
        if (!grown) return 0;
        *text = grown;
        *capacity = next;
    }
    (*text)[(*length)++] = value;
    (*text)[*length] = L'\0';
    return 1;
}

static wchar_t *h3cspeed_utf8_to_wide(const char *text) {
    int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text, -1,
                                    NULL, 0);
    if (!count) return NULL;
    wchar_t *wide = (wchar_t *)malloc((size_t)count * sizeof(*wide));
    if (!wide || !MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text, -1,
                                      wide, count)) {
        free(wide);
        return NULL;
    }
    return wide;
}

static int h3cspeed_append_argument(wchar_t **command, size_t *length,
                                    size_t *capacity, const wchar_t *argument) {
    int quoted = !*argument || wcspbrk(argument, L" \t\"") != NULL;
    if (quoted && !h3cspeed_append_wchar(command, length, capacity, L'\"'))
        return 0;
    size_t slashes = 0;
    for (const wchar_t *cursor = argument; ; cursor++) {
        if (*cursor == L'\\') { slashes++; continue; }
        if (*cursor == L'\"') {
            for (size_t index = 0; index < slashes * 2 + 1; index++)
                if (!h3cspeed_append_wchar(command, length, capacity, L'\\'))
                    return 0;
            if (!h3cspeed_append_wchar(command, length, capacity, L'\"'))
                return 0;
        } else {
            size_t count = !*cursor && quoted ? slashes * 2 : slashes;
            for (size_t index = 0; index < count; index++)
                if (!h3cspeed_append_wchar(command, length, capacity, L'\\'))
                    return 0;
            if (!*cursor) break;
            if (!h3cspeed_append_wchar(command, length, capacity, *cursor))
                return 0;
        }
        slashes = 0;
    }
    return !quoted || h3cspeed_append_wchar(command, length, capacity, L'\"');
}

static wchar_t *h3cspeed_command_line(char *const argv[]) {
    wchar_t *command = NULL;
    size_t length = 0, capacity = 0;
    for (size_t index = 0; argv[index]; index++) {
        wchar_t *argument = h3cspeed_utf8_to_wide(argv[index]);
        if (!argument ||
            (index && !h3cspeed_append_wchar(&command, &length, &capacity, L' ')) ||
            !h3cspeed_append_argument(&command, &length, &capacity, argument)) {
            free(argument);
            free(command);
            return NULL;
        }
        free(argument);
    }
    return command;
}

static int h3cspeed_spawn_errno(DWORD error) {
    if (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND ||
        error == ERROR_BAD_EXE_FORMAT) return ENOENT;
    if (error == ERROR_ACCESS_DENIED) return EACCES;
    if (error == ERROR_NOT_ENOUGH_MEMORY || error == ERROR_OUTOFMEMORY)
        return ENOMEM;
    return EINVAL;
}

static wchar_t *h3cspeed_resolve_program(const char *file) {
    wchar_t *wide = h3cspeed_utf8_to_wide(file);
    if (!wide) return NULL;
    wchar_t resolved[32768];
    DWORD length = SearchPathW(NULL, wide, L".exe",
                               (DWORD)(sizeof(resolved) / sizeof(*resolved)),
                               resolved, NULL);
    if (length && length < sizeof(resolved) / sizeof(*resolved)) {
        free(wide);
        size_t bytes = ((size_t)length + 1) * sizeof(*resolved);
        wchar_t *copy = (wchar_t *)malloc(bytes);
        if (copy) memcpy(copy, resolved, bytes);
        return copy;
    }
    return wide;
}

int h3cspeed_posix_spawnp(pid_t *pid, const char *file,
                          const posix_spawn_file_actions_t *actions,
                          const void *attr, char *const argv[],
                          char *const envp[]) {
    (void)attr;
    (void)envp;
    if (!file || !argv || !argv[0]) return EINVAL;

    int descriptor_count = 3;
    for (size_t index = 0; actions && index < actions->count; index++) {
        const h3cspeed_spawn_action *action = &actions->actions[index];
        int largest = action->descriptor;
        if (action->kind == H3CSPEED_SPAWN_DUP2 && action->target > largest)
            largest = action->target;
        if (action->kind != H3CSPEED_SPAWN_DUP2 &&
            action->kind != H3CSPEED_SPAWN_CLOSE) return EINVAL;
        if (largest < 0) return EINVAL;
        size_t limit = (USHRT_MAX - sizeof(int)) /
            (sizeof(unsigned char) + sizeof(intptr_t));
        if ((size_t)largest + 1 > limit) return EINVAL;
        if (largest + 1 > descriptor_count) descriptor_count = largest + 1;
    }

    HANDLE *mapped = (HANDLE *)malloc((size_t)descriptor_count * sizeof(*mapped));
    HANDLE *created = (HANDLE *)malloc((size_t)descriptor_count * sizeof(*created));
    HANDLE *inherited = (HANDLE *)malloc((size_t)descriptor_count * sizeof(*inherited));
    unsigned char *included = (unsigned char *)calloc((size_t)descriptor_count, 1);
    unsigned char *needed = (unsigned char *)calloc((size_t)descriptor_count, 1);
    if (!mapped || !created || !inherited || !included || !needed) {
        free(mapped); free(created); free(inherited); free(included); free(needed);
        return ENOMEM;
    }
    size_t created_count = 0;
    for (int descriptor = 0; descriptor < descriptor_count; descriptor++)
        mapped[descriptor] = INVALID_HANDLE_VALUE;
    for (int descriptor = 0; descriptor < 3; descriptor++) needed[descriptor] = 1;
    for (size_t index = 0; actions && index < actions->count; index++)
        if (actions->actions[index].kind == H3CSPEED_SPAWN_DUP2)
            needed[actions->actions[index].descriptor] = 1;
    for (int descriptor = 0; descriptor < descriptor_count; descriptor++) {
        if (!needed[descriptor]) continue;
        intptr_t raw = h3cspeed_os_handle(descriptor);
        if (raw == -1) continue;
        HANDLE duplicate = INVALID_HANDLE_VALUE;
        if (!DuplicateHandle(GetCurrentProcess(), (HANDLE)raw,
                             GetCurrentProcess(), &duplicate, 0, TRUE,
                             DUPLICATE_SAME_ACCESS)) continue;
        mapped[descriptor] = duplicate;
        created[created_count++] = duplicate;
        DWORD flags = 0;
        included[descriptor] = (unsigned char)(
            descriptor < 3 ||
            (GetHandleInformation((HANDLE)raw, &flags) &&
             (flags & HANDLE_FLAG_INHERIT)));
    }
    int code = 0;
    for (size_t index = 0; actions && index < actions->count; index++) {
        const h3cspeed_spawn_action *action = &actions->actions[index];
        if (action->kind == H3CSPEED_SPAWN_DUP2) {
            if (mapped[action->descriptor] == INVALID_HANDLE_VALUE) {
                code = EBADF;
                break;
            }
            mapped[action->target] = mapped[action->descriptor];
            included[action->target] = 1;
        } else {
            mapped[action->descriptor] = INVALID_HANDLE_VALUE;
            included[action->descriptor] = 0;
        }
    }

    size_t inherited_count = 0;
    for (int descriptor = 0; descriptor < descriptor_count; descriptor++) {
        HANDLE handle = mapped[descriptor];
        if (!included[descriptor] || handle == INVALID_HANDLE_VALUE) continue;
        int seen = 0;
        for (size_t index = 0; index < inherited_count; index++)
            if (inherited[index] == handle) seen = 1;
        if (!seen) inherited[inherited_count++] = handle;
    }

    size_t reserved_size = sizeof(int) + (size_t)descriptor_count *
        (sizeof(unsigned char) + sizeof(intptr_t));
    unsigned char *reserved = (unsigned char *)calloc(1, reserved_size);
    wchar_t *command = h3cspeed_command_line(argv);
    wchar_t *program = h3cspeed_resolve_program(file);
    SIZE_T attributes_size = 0;
    STARTUPINFOEXW startup;
    PROCESS_INFORMATION process;
    memset(&startup, 0, sizeof(startup));
    memset(&process, 0, sizeof(process));
    startup.StartupInfo.cb = sizeof(startup);
    if (!reserved || !command || !program) code = ENOMEM;
    if (!code) {
        memcpy(reserved, &descriptor_count, sizeof(descriptor_count));
        unsigned char *flags = reserved + sizeof(int);
        unsigned char *handles = flags + descriptor_count;
        for (int descriptor = 0; descriptor < descriptor_count; descriptor++) {
            intptr_t value = -1;
            if (included[descriptor] &&
                mapped[descriptor] != INVALID_HANDLE_VALUE) {
                flags[descriptor] = 0x01; /* CRT FOPEN, binary by default. */
                value = (intptr_t)mapped[descriptor];
            }
            memcpy(handles + (size_t)descriptor * sizeof(value),
                   &value, sizeof(value));
        }
        startup.StartupInfo.cbReserved2 = (WORD)reserved_size;
        startup.StartupInfo.lpReserved2 = reserved;
        startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
        startup.StartupInfo.hStdInput = included[0] ? mapped[0] : INVALID_HANDLE_VALUE;
        startup.StartupInfo.hStdOutput = included[1] ? mapped[1] : INVALID_HANDLE_VALUE;
        startup.StartupInfo.hStdError = included[2] ? mapped[2] : INVALID_HANDLE_VALUE;

        if (inherited_count) {
            (void)InitializeProcThreadAttributeList(NULL, 1, 0, &attributes_size);
            startup.lpAttributeList = (LPPROC_THREAD_ATTRIBUTE_LIST)
                malloc(attributes_size);
        }
        if (inherited_count && (!startup.lpAttributeList ||
            !InitializeProcThreadAttributeList(startup.lpAttributeList, 1, 0,
                                               &attributes_size) ||
            !UpdateProcThreadAttribute(
                startup.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                inherited, inherited_count * sizeof(*inherited), NULL, NULL))) {
            code = ENOMEM;
        }
    }
    if (!code) {
        BOOL ok = CreateProcessW(program, command, NULL, NULL,
            inherited_count != 0,
            EXTENDED_STARTUPINFO_PRESENT, NULL, NULL,
            &startup.StartupInfo, &process);
        if (!ok) code = h3cspeed_spawn_errno(GetLastError());
    }
    if (process.hThread) CloseHandle(process.hThread);
    if (startup.lpAttributeList) {
        DeleteProcThreadAttributeList(startup.lpAttributeList);
        free(startup.lpAttributeList);
    }
    free(command);
    free(program);
    free(reserved);
    for (size_t index = 0; index < created_count; index++)
        CloseHandle(created[index]);
    free(mapped);
    free(created);
    free(inherited);
    free(included);
    free(needed);
    if (code) {
        if (process.hProcess) CloseHandle(process.hProcess);
        return code;
    }
    if (pid) *pid = (intptr_t)process.hProcess;
    else CloseHandle(process.hProcess);
    return 0;
}
#endif
