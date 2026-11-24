#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "kssm_exp4_shared.h"

#define PACK4(v) (((uint64_t)(v)) | ((uint64_t)(v)<<16) | ((uint64_t)(v)<<32) | ((uint64_t)(v)<<48))
#define FD_CACHE_SZ 256

struct fd_ent { uint64_t ino; int fd; uint64_t dropped; char path[256]; };
static struct fd_ent g_c[FD_CACHE_SZ];
static int g_cn = 0;
static volatile sig_atomic_t g_stop = 0;
static void on_sig(int s) { (void)s; g_stop = 1; }

static struct fd_ent *get_fd(uint32_t pid, uint64_t ino) {
    for (int i = 0; i < g_cn; i++) if (g_c[i].ino == ino && g_c[i].fd >= 0) return &g_c[i];
    char dp[64]; snprintf(dp, sizeof(dp), "/proc/%u/fd", pid);
    DIR *d = opendir(dp); if (!d) return NULL;
    struct dirent *e; int fd = -1; char tgt[256] = {0};
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') continue;
        char fp[300]; snprintf(fp, sizeof(fp), "/proc/%u/fd/%s", pid, e->d_name);
        struct stat st;
        if (stat(fp, &st) == 0 && (uint64_t)st.st_ino == ino) {
            ssize_t rl = readlink(fp, tgt, sizeof(tgt) - 1);
            if (rl > 0) tgt[rl] = '\0';
            fd = open(fp, O_RDONLY);
            break;
        }
    }
    closedir(d);
    if (fd < 0) return NULL;
    int slot = (g_cn < FD_CACHE_SZ) ? g_cn++ : 0;
    if (slot == 0 && g_cn == FD_CACHE_SZ && g_c[0].fd >= 0) close(g_c[0].fd);
    g_c[slot].ino = ino; g_c[slot].fd = fd; g_c[slot].dropped = 0;
    snprintf(g_c[slot].path, sizeof(g_c[slot].path), "%s", tgt[0] ? tgt : "(unknown)");
    return &g_c[slot];
}

static int on_hint(void *ctx, void *data, size_t sz) {
    (void)ctx;
    if (sz != sizeof(struct exp4_drop_hint)) return 0;
    const struct exp4_drop_hint *h = (const struct exp4_drop_hint *)data;
    struct fd_ent *fe = get_fd(h->pid, h->ino);
    if (!fe) return 0;
    posix_fadvise(fe->fd, (off_t)h->drop_start, (off_t)h->drop_len, POSIX_FADV_DONTNEED);
    fe->dropped += h->drop_len;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "Usage: %s <mnt> <kssm|size|seqdrop> <weights.bin>\n", argv[0]); return 1; }
    struct stat st; if (stat(argv[1], &st) != 0) { perror("stat"); return 1; }
    uint32_t kdev = (major(st.st_dev) << 20) | minor(st.st_dev);
    uint32_t mode = (strcmp(argv[2], "size") == 0) ? EXP4_MODE_SIZE :
                    ((strcmp(argv[2], "seqdrop") == 0) ? EXP4_MODE_SEQDROP : EXP4_MODE_KSSM);

    signal(SIGINT, on_sig); signal(SIGTERM, on_sig);
    struct bpf_object *obj = bpf_object__open_file("/home/yjw/workspace/research/KSSM/exp4_read_hardened/kssm_exp4.bpf.o", NULL);
    if (!obj) return 1;
    struct bpf_map *m;
    bpf_object__for_each_map(m, obj) {
        if (strstr(bpf_map__name(m), ".rodata")) {
            size_t sz = 0; uint32_t *v = bpf_map__initial_value(m, &sz);
            if (v && sz >= 8) { v[0] = kdev; v[1] = mode; }
            break;
        }
    }
    if (bpf_object__load(obj) != 0) { fprintf(stderr, "load failed\n"); return 1; }

    uint32_t k0 = 0;
    struct kssm_live_weights *w = calloc(1, sizeof(*w));
    FILE *wf = fopen(argv[3], "rb");
    if (!wf || fread(w, sizeof(*w), 1, wf) != 1) { perror("read weights"); return 1; }
    fclose(wf);
    bpf_map_update_elem(bpf_object__find_map_fd_by_name(obj, "weights_map"), &k0, w, BPF_ANY);
    free(w);

    uint8_t zbuf[256] = {0}; uint64_t *zh = (uint64_t *)zbuf;
    for (int g = 0; g < 16; g++) zh[g] = PACK4(16384ULL);
    bpf_map_update_elem(bpf_object__find_map_fd_by_name(obj, "zero_ino_map"), &k0, zbuf, BPF_ANY);

    struct bpf_program *p = bpf_object__find_program_by_name(obj, "on_ext4_read");
    struct bpf_link *lnk = bpf_program__attach(p);
    if (libbpf_get_error(lnk)) { fprintf(stderr, "attach failed\n"); return 1; }

    struct ring_buffer *rb = ring_buffer__new(bpf_object__find_map_fd_by_name(obj, "drop_rb"), on_hint, NULL, NULL);
    fprintf(stderr, "READY mode=%s\n", argv[2]); fflush(stderr);

    while (!g_stop) ring_buffer__poll(rb, 10);
    ring_buffer__consume(rb);

    struct bpf_prog_info info = {}; uint32_t ilen = sizeof(info);
    bpf_obj_get_info_by_fd(bpf_program__fd(p), &info, &ilen);
    double avg = info.run_cnt ? (double)info.run_time_ns / info.run_cnt : 0.0;

    uint64_t b_st = 0, b_emb = 0, b_sst = 0; int n_st = 0, n_emb = 0, n_sst = 0;
    for (int i = 0; i < g_cn; i++) {
        if (strstr(g_c[i].path, ".safetensors")) { b_st += g_c[i].dropped; n_st++; }
        else if (strstr(g_c[i].path, "hot_seq_embed")) { b_emb += g_c[i].dropped; n_emb++; }
        else if (strstr(g_c[i].path, ".sst")) { b_sst += g_c[i].dropped; n_sst++; }
        if (g_c[i].fd >= 0) close(g_c[i].fd);
    }
    printf("AUDIT mode=%s | avg_ns=%.2f | .safetensors=%d(%.2fG) | HotSeqEmbed_FP=%d(%.2fG) | RocksDB_SST_FP=%d(%.2fG)\n",
           argv[2], avg, n_st, b_st/(1024.0*1024.0*1024.0), n_emb, b_emb/(1024.0*1024.0*1024.0), n_sst, b_sst/(1024.0*1024.0*1024.0));
    fflush(stdout);

    ring_buffer__free(rb);
    bpf_link__destroy(lnk);
    bpf_object__close(obj);
    return 0;
}
