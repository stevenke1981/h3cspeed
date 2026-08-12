#include <stdio.h>
#include <string.h>

int main(void) {
    char input[64] = {0};
    if (!fgets(input, sizeof(input), stdin)) return 2;
    if (strcmp(input, "redirect-ok\n") != 0) return 3;
    if (fputs("child-ok\n", stdout) < 0 || fflush(stdout) != 0) return 4;
    return 0;
}
