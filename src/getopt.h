#ifndef H3CSPEED_GETOPT_H
#define H3CSPEED_GETOPT_H
#if defined(_WIN32)
#include <string.h>

enum { no_argument = 0, required_argument = 1, optional_argument = 2 };
struct option {
    const char *name;
    int has_arg;
    int *flag;
    int val;
};
static char *optarg;
static int optind = 1;
static int opterr = 1;
static int optopt;
static inline int h3cspeed_getopt_long(int argc, char *const argv[],
                                       const char *short_options,
                                       const struct option *long_options,
                                       int *long_index) {
    (void)opterr;
    if (optind >= argc || !argv[optind] || argv[optind][0] != '-' ||
        argv[optind][1] == '\0') return -1;
    char *argument = argv[optind++];
    if (argument[1] == '-') {
        char *name = argument + 2;
        char *value = strchr(name, '=');
        if (value) *value++ = '\0';
        for (int index = 0; long_options && long_options[index].name; index++) {
            if (strcmp(name, long_options[index].name) != 0) continue;
            if (long_index) *long_index = index;
            if (long_options[index].has_arg == required_argument) {
                if (!value && optind < argc) value = argv[optind++];
                if (!value) return '?';
                optarg = value;
            } else optarg = value;
            if (long_options[index].flag) {
                *long_options[index].flag = long_options[index].val; return 0;
            }
            return long_options[index].val;
        }
        return '?';
    }
    char short_name = argument[1];
    const char *entry = strchr(short_options ? short_options : "", short_name);
    if (!entry) { optopt = short_name; return '?'; }
    if (entry[1] == ':') {
        if (argument[2] != '\0') optarg = argument + 2;
        else if (optind < argc) optarg = argv[optind++];
        else return '?';
    } else optarg = NULL;
    return short_name;
}
#define getopt_long h3cspeed_getopt_long
#else
#include_next <getopt.h>
#endif
#endif
