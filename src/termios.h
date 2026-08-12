#ifndef H3CSPEED_TERMIOS_H
#define H3CSPEED_TERMIOS_H
#if defined(_WIN32)
struct termios { int c_lflag; };
#define ICANON 0x0002
#define ECHO 0x0008
#define TCSANOW 0
static inline int tcgetattr(int fd, struct termios *termios) { (void)fd; (void)termios; return -1; }
static inline int tcsetattr(int fd, int action, const struct termios *termios) { (void)fd; (void)action; (void)termios; return -1; }
#else
#include_next <termios.h>
#endif
#endif
