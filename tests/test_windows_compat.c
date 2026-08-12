#include "spawn.h"
#include "strings.h"
#include "unistd.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>

static int fail(const char *message) {
    fprintf(stderr, "windows compat test failed: %s\n", message);
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 2) return fail("missing child executable");
    if (strncasecmp("AbC-tail", "aBc-other", 3) != 0)
        return fail("strncasecmp count boundary");

    char file_pattern[] = "/tmp/h3cspeed-file-XXXXXX";
    int file = mkstemp(file_pattern);
    if (file < 0) return fail("mkstemp");
    DWORD ignored = 0;
    HANDLE file_handle = (HANDLE)_get_osfhandle(file);
    if (!DeviceIoControl(file_handle, FSCTL_SET_SPARSE, NULL, 0,
                         NULL, 0, &ignored, NULL)) {
        DWORD sparse_error = GetLastError();
        close(file); unlink(file_pattern);
        if (sparse_error != ERROR_INVALID_FUNCTION &&
            sparse_error != ERROR_NOT_SUPPORTED)
            return fail("mark sparse fixture");
    } else {
        const int64_t offset = INT64_C(5) * 1024 * 1024 * 1024 + 7;
        const char payload[4] = {'A', 0x1a, 'B', 'C'};
        LARGE_INTEGER large_offset;
        large_offset.QuadPart = offset;
        if (!SetFilePointerEx(file_handle, large_offset, NULL, FILE_BEGIN) ||
            !SetEndOfFile(file_handle) ||
            !SetFilePointerEx(file_handle, large_offset, NULL, FILE_BEGIN) ||
            _write(file, payload, (unsigned)sizeof(payload)) !=
                (int)sizeof(payload)) {
            close(file); unlink(file_pattern);
            return fail("large sparse fixture");
        }
        close(file);
        file = open(file_pattern, O_RDONLY);
        if (file < 0) {
            unlink(file_pattern); return fail("reopen sparse fixture");
        }
        char sequential[4] = {0};
        if (_lseeki64(file, offset, SEEK_SET) != offset ||
            read(file, sequential, sizeof(sequential)) !=
                (ssize_t)sizeof(sequential) ||
            memcmp(sequential, payload, sizeof(payload)) != 0) {
            close(file); unlink(file_pattern);
            return fail("64-bit binary read");
        }
        char readback[4] = {0};
        if (pread(file, readback, sizeof(readback), offset) !=
                (int64_t)sizeof(readback) ||
            memcmp(readback, payload, sizeof(payload)) != 0) {
            close(file); unlink(file_pattern);
            return fail("64-bit binary pread");
        }
        struct stat status;
        if (fstat(file, &status) != 0 || status.st_size != offset + 4) {
            close(file); unlink(file_pattern); return fail("64-bit fstat");
        }
        close(file);
        unlink(file_pattern);
    }

    char directory_pattern[] = "/tmp/h3cspeed-dir-XXXXXX";
    if (!mkdtemp(directory_pattern)) return fail("mkdtemp");
    if (_rmdir(directory_pattern) != 0) return fail("remove temp directory");

    int input_pipe[2] = {-1, -1};
    int output_pipe[2] = {-1, -1};
    if (pipe(input_pipe) != 0 || pipe(output_pipe) != 0)
        return fail("pipe creation");
    posix_spawn_file_actions_t actions;
    if (posix_spawn_file_actions_init(&actions) != 0 ||
        posix_spawn_file_actions_adddup2(&actions, input_pipe[0], 0) != 0 ||
        posix_spawn_file_actions_adddup2(&actions, output_pipe[1], 1) != 0)
        return fail("spawn dup actions");
    int descriptors[] = {input_pipe[0], input_pipe[1],
                         output_pipe[0], output_pipe[1]};
    for (size_t index = 0; index < 4; index++)
        if (posix_spawn_file_actions_addclose(&actions, descriptors[index]) != 0)
            return fail("spawn close actions");
    char *arguments[] = {argv[1], NULL};
    pid_t child = -1;
    int spawn_code = posix_spawnp(&child, argv[1], &actions, NULL,
                                  arguments, NULL);
    posix_spawn_file_actions_destroy(&actions);
    close(input_pipe[0]);
    close(output_pipe[1]);
    if (spawn_code != 0) return fail("spawn child");
    if (write(input_pipe[1], "redirect-ok\n", 12) != 12)
        return fail("write child stdin");
    close(input_pipe[1]);
    char child_output[32] = {0};
    ssize_t received = read(output_pipe[0], child_output,
                            (unsigned)(sizeof(child_output) - 1));
    close(output_pipe[0]);
    int child_status = 0;
    if (waitpid(child, &child_status, 0) < 0 ||
        !WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0)
        return fail("child exit status");
    int output_ok =
        (received == 9 && memcmp(child_output, "child-ok\n", 9) == 0) ||
        (received == 10 && memcmp(child_output, "child-ok\r\n", 10) == 0);
    if (!output_ok) {
        fprintf(stderr, "received=%lld output='%.*s'\n",
                (long long)received,
                (int)(received > 0 ? received : 0), child_output);
        return fail("child stdout redirect");
    }

    puts("windows compatibility: passed");
    return 0;
}
