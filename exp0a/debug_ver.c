#include <stdio.h>
#include <string.h>
#include <bpf/libbpf.h>
static int pr(enum libbpf_print_level l, const char *f, va_list a) {
    if (l == LIBBPF_DEBUG) return 0;
    return vfprintf(stderr, f, a);
}
int main(int argc, char **argv) {
    libbpf_set_print(pr);
    static char vlog[1048576];
    LIBBPF_OPTS(bpf_object_open_opts, opts, .kernel_log_buf = vlog, .kernel_log_size = sizeof(vlog), .kernel_log_level = 1);
    struct bpf_object *obj = bpf_object__open_file("kssm_exp0a.bpf.o", &opts);
    struct bpf_program *p;
    bpf_object__for_each_program(p, obj)
        bpf_program__set_autoload(p, strcmp(bpf_program__name(p), argv[1]) == 0);
    int err = bpf_object__load(obj);
    printf("prog=%s err=%d\n", argv[1], err);
    size_t len = strlen(vlog);
    if (len > 2500) printf("...%s\n", vlog + len - 2500);
    else printf("%s\n", vlog);
    bpf_object__close(obj);
    return 0;
}
