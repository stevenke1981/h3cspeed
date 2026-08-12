#include "h3_resize_portable.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int test_identity(void) {
    const uint8_t input[] = {
        255, 0, 0,  0, 255, 0,
        0, 0, 255,  255, 255, 255
    };
    uint8_t *output = NULL;
    if (!h3cspeed_resize_rgb24_lanczos(input, 1, 2, 2, 2, 2, &output)) return 0;
    int ok = output && memcmp(input, output, sizeof(input)) == 0;
    free(output);
    return ok;
}

static int test_constant(void) {
    uint8_t input[3 * 3 * 3];
    for (size_t index = 0; index < sizeof(input); index += 3) {
        input[index] = 17;
        input[index + 1] = 91;
        input[index + 2] = 203;
    }
    uint8_t *output = NULL;
    if (!h3cspeed_resize_rgb24_lanczos(input, 1, 3, 3, 11, 7, &output)) return 0;
    int ok = output != NULL;
    for (size_t index = 0; ok && index < (size_t)11 * 7 * 3; index += 3) {
        if (output[index] != 17 || output[index + 1] != 91 ||
            output[index + 2] != 203) ok = 0;
    }
    free(output);
    return ok;
}

int main(void) {
    if (!test_identity()) {
        fprintf(stderr, "identity resize failed\n");
        return 1;
    }
    if (!test_constant()) {
        fprintf(stderr, "constant resize failed\n");
        return 1;
    }
    puts("resize tests passed");
    return 0;
}
