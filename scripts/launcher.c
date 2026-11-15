/* Keep AppKit and the microphone request in the Skrybek bundle process. */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(void) {
    if (chdir(PROJECT_DIR) != 0 || setenv("PATH", PROJECT_PATH, 1) != 0) {
        perror("Skrybek launcher");
        return 1;
    }
    void *library = dlopen(PYTHON_LIBRARY, RTLD_NOW | RTLD_GLOBAL);
    if (library == NULL) {
        fprintf(stderr, "%s\n", dlerror());
        return 1;
    }
    int (*python_main)(int, char **) = dlsym(library, "Py_BytesMain");
    if (python_main == NULL) {
        fprintf(stderr, "%s\n", dlerror());
        return 1;
    }
    char *args[] = {PYTHON_EXECUTABLE, PROJECT_DIR "/desktop.py", NULL};
    return python_main(2, args);
}
