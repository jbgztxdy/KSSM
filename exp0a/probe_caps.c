#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <bpf/libbpf.h>

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
    if (level == LIBBPF_DEBUG)
        return 0;
    return vfprintf(stderr, format, args);
}

static void try_prog(const char *obj_path, const char *prog_name)
{
    printf("\n=== Testing %s ===\n", prog_name);
    struct bpf_object *obj = bpf_object__open_file(obj_path, NULL);
    if (!obj) { printf("open failed\n"); return; }

    struct bpf_program *prog;
    bpf_object__for_each_program(prog, obj) {
        if (strcmp(bpf_program__name(prog), prog_name) == 0)
            bpf_program__set_autoload(prog, true);
        else
            bpf_program__set_autoload(prog, false);
    }

    int err = bpf_object__load(obj);
    if (err) {
        printf("LOAD FAILED: %d (%s)\n", err, strerror(-err));
        bpf_object__close(obj);
        return;
    }
    printf("LOAD OK (verifier passed!)\n");

    prog = bpf_object__find_program_by_name(obj, prog_name);
    struct bpf_link *link = bpf_program__attach(prog);
    if (libbpf_get_error(link)) {
        err = libbpf_get_error(link);
        printf("ATTACH FAILED: %d (%s)\n", err, strerror(-err));
    } else {
        printf("ATTACH OK!\n");
        bpf_link__destroy(link);
    }
    bpf_object__close(obj);
}

int main(void)
{
    libbpf_set_print(libbpf_print_fn);
    const char *obj = "/home/yjw/workspace/research/kssm_probe/exp0a/probe_caps.bpf.o";
    try_prog(obj, "test_fentry_task");
    try_prog(obj, "test_fentry_both");
    try_prog(obj, "test_lsm_both");
    try_prog(obj, "test_fentry_stack");
    return 0;
}
