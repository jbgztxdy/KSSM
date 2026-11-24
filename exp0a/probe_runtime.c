#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <bpf/libbpf.h>
#include "probe_runtime.skel.h"

int main(void)
{
    struct probe_runtime_bpf *skel = probe_runtime_bpf__open_and_load();
    if (!skel) { fprintf(stderr, "open_and_load failed\n"); return 1; }
    skel->bss->target_pid = getpid();
    if (probe_runtime_bpf__attach(skel)) { fprintf(stderr, "attach failed\n"); return 1; }

    for (int i = 0; i < 10; i++) {
        int fd = open("/etc/hostname", O_RDONLY);
        char buf[64];
        if (fd >= 0) {
            ssize_t r = read(fd, buf, sizeof(buf));
            (void)r;
            close(fd);
        }
    }

    printf("lsm_hits=%llu, fentry_hits=%llu, last_kstack=%lld, last_ustack=%lld\n",
           (unsigned long long)skel->bss->lsm_hits,
           (unsigned long long)skel->bss->fentry_hits,
           (long long)skel->data->last_kstack,
           (long long)skel->data->last_ustack);
    probe_runtime_bpf__destroy(skel);
    return 0;
}
