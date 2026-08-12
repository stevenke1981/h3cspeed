#include "linenoise.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *linenoiseEditMore = NULL;
static linenoiseCompletionCallback *completion_callback;
static linenoiseHintsCallback *hints_callback;
static linenoiseFreeHintsCallback *free_hints_callback;

int linenoiseEditStart(struct linenoiseState *state, int stdin_fd, int stdout_fd,
                       char *buffer, size_t length, const char *prompt) {
    (void)state; (void)stdin_fd; (void)stdout_fd; (void)buffer;
    (void)length; (void)prompt; return -1;
}
char *linenoiseEditFeed(struct linenoiseState *state) { (void)state; return NULL; }
void linenoiseEditStop(struct linenoiseState *state) { (void)state; }
void linenoiseHide(struct linenoiseState *state) { (void)state; }
void linenoiseShow(struct linenoiseState *state) { (void)state; }

char *linenoise(const char *prompt) {
    if (prompt) fputs(prompt, stdout);
    fflush(stdout);
    char line[4096];
    if (!fgets(line, sizeof(line), stdin)) return NULL;
    line[strcspn(line, "\r\n")] = '\0';
    return _strdup(line);
}
void linenoiseFree(void *ptr) { free(ptr); }
void linenoiseSetCompletionCallback(linenoiseCompletionCallback *callback) { completion_callback = callback; }
void linenoiseSetHintsCallback(linenoiseHintsCallback *callback) { hints_callback = callback; }
void linenoiseSetFreeHintsCallback(linenoiseFreeHintsCallback *callback) { free_hints_callback = callback; }
void linenoiseAddCompletion(linenoiseCompletions *completions, const char *string) {
    (void)completions; (void)string;
}
int linenoiseHistoryAdd(const char *line) { (void)line; return 1; }
int linenoiseHistorySetMaxLen(int length) { (void)length; return 1; }
int linenoiseHistorySave(const char *filename) { (void)filename; return 1; }
int linenoiseHistoryLoad(const char *filename) { (void)filename; return 1; }
void linenoiseClearScreen(void) { }
void linenoiseSetMultiLine(int multiline) { (void)multiline; }
void linenoisePrintKeyCodes(void) { }
void linenoiseMaskModeEnable(void) { }
void linenoiseMaskModeDisable(void) { }
