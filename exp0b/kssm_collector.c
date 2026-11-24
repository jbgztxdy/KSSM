#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "trace_event.h"

static volatile sig_atomic_t g_stop = 0;
static FILE *g_out = NULL;
static uint64_t g_written = 0;

static void on_sig(int s) { (void)s; g_stop = 1; }

static int handle_event(void *ctx, void *data, size_t sz)
{
    (void)ctx;
    if (sz == sizeof(struct kssm_event) && g_out) {
        fwrite(data, sizeof(struct kssm_event), 1, g_out);
        g_written++;
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <mount_dir_for_sdev> <out.bin> [pid_filter]\n", argv[0]);
        return 1;
    }
    struct stat st;
    if (stat(argv[1], &st) != 0) { perror("stat"); return 1; }
    uint32_t kdev = (major(st.st_dev) << 20) | minor(st.st_dev);
    uint32_t pid_filter = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 0;

    g_out = fopen(argv[2], "wb");
    if (!g_out) { perror("fopen"); return 1; }
    setvbuf(g_out, NULL, _IOFBF, 1024 * 1024);

    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);

    struct bpf_object *obj = bpf_object__open_file("/home/yjw/workspace/research/KSSM/exp0b/kssm_collector.bpf.o", NULL);
    if (!obj) { fprintf(stderr, "open bpf failed\n"); return 1; }

    struct bpf_map *m;
    bpf_object__for_each_map(m, obj) {
        if (strstr(bpf_map__name(m), ".rodata")) {
            size_t sz = 0;
            uint32_t *vals = (uint32_t *)bpf_map__initial_value(m, &sz);
            if (vals && sz >= 8) {
                vals[0] = kdev;
                vals[1] = pid_filter;
            }
            break;
        }
    }

    if (bpf_object__load(obj) != 0) { fprintf(stderr, "load bpf failed\n"); return 1; }

    struct bpf_link *links[16] = {0};
    int nl = 0;
    struct bpf_program *p;
    bpf_object__for_each_program(p, obj) {
        struct bpf_link *l = bpf_program__attach(p);
        if (libbpf_get_error(l)) {
            fprintf(stderr, "attach failed: %s\n", bpf_program__name(p));
            return 1;
        }
        links[nl++] = l;
    }

    int rb_fd = bpf_object__find_map_fd_by_name(obj, "rb");
    struct ring_buffer *rb = ring_buffer__new(rb_fd, handle_event, NULL, NULL);
    if (!rb) { fprintf(stderr, "ringbuf new failed\n"); return 1; }

    fprintf(stderr, "READY kdev=0x%x (maj=%u min=%u) hooks=%d\n",
            kdev, major(st.st_dev), minor(st.st_dev), nl);
    fflush(stderr);

    while (!g_stop)
        ring_buffer__poll(rb, 50);
    ring_buffer__consume(rb);

    int cfd = bpf_object__find_map_fd_by_name(obj, "cnt_map");
    int ncpus = libbpf_num_possible_cpus();
    uint64_t *vals = calloc(ncpus, sizeof(uint64_t));
    uint64_t emitted = 0, dropped = 0;
    uint32_t k0 = 0, k1 = 1;
    if (bpf_map_lookup_elem(cfd, &k0, vals) == 0)
        for (int i = 0; i < ncpus; i++) emitted += vals[i];
    if (bpf_map_lookup_elem(cfd, &k1, vals) == 0)
        for (int i = 0; i < ncpus; i++) dropped += vals[i];
    free(vals);

    fflush(g_out);
    fclose(g_out);
    fprintf(stderr, "DONE written=%llu emitted=%llu dropped=%llu\n",
            (unsigned long long)g_written, (unsigned long long)emitted, (unsigned long long)dropped);

    ring_buffer__free(rb);
    for (int i = 0; i < nl; i++) bpf_link__destroy(links[i]);
    bpf_object__close(obj);
    return 0;
}
