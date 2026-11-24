#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "kssm_write_shared.h"

#ifndef F_SET_RW_HINT
#define F_SET_RW_HINT 1036
#define F_GET_RW_HINT 1035
#endif
#ifndef RWH_WRITE_LIFE_SHORT
#define RWH_WRITE_LIFE_SHORT  2
#define RWH_WRITE_LIFE_MEDIUM 3
#define RWH_WRITE_LIFE_LONG   4
#endif

#define PACK4(v) (((uint64_t)(v)) | ((uint64_t)(v)<<16) | ((uint64_t)(v)<<32) | ((uint64_t)(v)<<48))

static volatile sig_atomic_t g_stop = 0;
static int g_head_fd = -1;
static double g_W[3][128];
static double g_b[3];
static uint64_t g_births = 0, g_hints_applied = 0, g_hints_verified = 0;
static uint64_t g_deaths = 0, g_slowloop_syncs = 0, g_correct_preds = 0;
static int g_dirty_head = 0;

static void on_sig(int s) { (void)s; g_stop = 1; }

static void sync_head_to_kernel(void)
{
    if (g_head_fd < 0) return;
    struct kssm_life_head h = {0};
    for (int c = 0; c < 3; c++) {
        for (int d = 0; d < 64; d++) {
            double vi = g_W[c][d] * 256.0;
            double vt = g_W[c][64 + d] * 256.0;
            if (vi > 32767) vi = 32767; if (vi < -32768) vi = -32768;
            if (vt > 32767) vt = 32767; if (vt < -32768) vt = -32768;
            h.w_i[c][d] = (int16_t)llround(vi);
            h.w_t[c][d] = (int16_t)llround(vt);
        }
        h.b[c] = (int32_t)llround(g_b[c] * 65536.0);
    }
    uint32_t k0 = 0;
    if (bpf_map_update_elem(g_head_fd, &k0, &h, BPF_ANY) == 0)
        g_slowloop_syncs++;
}

static void apply_rw_hint(uint32_t pid, uint64_t ino, uint8_t pred_class)
{
    char dirp[64];
    snprintf(dirp, sizeof(dirp), "/proc/%u/fd", pid);
    DIR *d = opendir(dirp);
    if (!d) return;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        char fdp[300];
        snprintf(fdp, sizeof(fdp), "/proc/%u/fd/%s", pid, ent->d_name);
        struct stat st;
        if (stat(fdp, &st) == 0 && (uint64_t)st.st_ino == ino) {
            int fd = open(fdp, O_RDONLY);
            if (fd >= 0) {
                uint64_t hint = (pred_class == 0) ? RWH_WRITE_LIFE_SHORT :
                                ((pred_class == 1) ? RWH_WRITE_LIFE_MEDIUM : RWH_WRITE_LIFE_LONG);
                if (fcntl(fd, F_SET_RW_HINT, &hint) == 0) {
                    g_hints_applied++;
                    uint64_t verify = 0;
                    if (fcntl(fd, F_GET_RW_HINT, &verify) == 0 && verify == hint)
                        g_hints_verified++;
                }
                close(fd);
            }
            break;
        }
    }
    closedir(d);
}

static int handle_msg(void *ctx, void *data, size_t sz)
{
    (void)ctx;
    if (sz != sizeof(struct kssm_write_msg)) return 0;
    const struct kssm_write_msg *m = (const struct kssm_write_msg *)data;

    if (m->msg_type == MSG_BIRTH_HINT) {
        g_births++;
        apply_rw_hint(m->pid, m->ino, m->pred_class);
    } else if (m->msg_type == MSG_DEATH_LABEL) {
        g_deaths++;
        if (m->pred_class == m->true_class) g_correct_preds++;
        /* Zero-BPTT online logistic regression update on < 1 KB linear readout head! */
        double x[128];
        for (int c = 0; c < 64; c++) {
            x[c]      = m->snap_hi[c] / 256.0;
            x[64 + c] = m->snap_ht[c] / 256.0;
        }
        double lg[3], mx = -1e9;
        for (int k = 0; k < 3; k++) {
            double s = g_b[k];
            for (int j = 0; j < 128; j++) s += g_W[k][j] * x[j];
            lg[k] = s;
            if (s > mx) mx = s;
        }
        double denom = 0.0, pr[3];
        for (int k = 0; k < 3; k++) { pr[k] = exp(lg[k] - mx); denom += pr[k]; }
        for (int k = 0; k < 3; k++) pr[k] /= denom;

        double lr = (m->true_class > 0) ? 0.08 : 0.02;
        for (int k = 0; k < 3; k++) {
            double grad = pr[k] - (k == m->true_class ? 1.0 : 0.0);
            for (int j = 0; j < 128; j++) g_W[k][j] -= lr * grad * x[j];
            g_b[k] -= lr * grad;
        }
        g_dirty_head = 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "Usage: %s <mount_dir>\n", argv[0]); return 1; }
    struct stat st;
    if (stat(argv[1], &st) != 0) { perror("stat"); return 1; }
    uint32_t kdev = (major(st.st_dev) << 20) | minor(st.st_dev);

    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);

    struct bpf_object *obj = bpf_object__open_file("/home/yjw/workspace/research/KSSM/exp1_write_fdp/kssm_write_live.bpf.o", NULL);
    if (!obj) return 1;
    struct bpf_map *mp;
    bpf_object__for_each_map(mp, obj) {
        if (strstr(bpf_map__name(mp), ".rodata")) {
            size_t sz = 0; uint32_t *v = bpf_map__initial_value(mp, &sz);
            if (v && sz >= 4) v[0] = kdev;
            break;
        }
    }
    if (bpf_object__load(obj) != 0) { fprintf(stderr, "BPF load failed!\n"); return 1; }

    /* Load 24KB frozen Guard-SWAR base & 784B initial head */
    uint32_t k0 = 0;
    struct kssm_life_base *base = calloc(1, sizeof(*base));
    FILE *fb = fopen("/home/yjw/workspace/research/KSSM/exp1_write_fdp/kssm_life_base.bin", "rb");
    if (!fb || fread(base, sizeof(*base), 1, fb) != 1) { perror("read base"); return 1; }
    fclose(fb);
    bpf_map_update_elem(bpf_object__find_map_fd_by_name(obj, "base_map"), &k0, base, BPF_ANY);
    free(base);

    g_head_fd = bpf_object__find_map_fd_by_name(obj, "head_map");
    struct kssm_life_head init_h = {0};
    FILE *fh = fopen("/home/yjw/workspace/research/KSSM/exp1_write_fdp/kssm_life_head.bin", "rb");
    if (!fh || fread(&init_h, sizeof(init_h), 1, fh) != 1) { perror("read head"); return 1; }
    fclose(fh);
    for (int c = 0; c < 3; c++) {
        for (int d = 0; d < 64; d++) {
            g_W[c][d]      = init_h.w_i[c][d] / 256.0;
            g_W[c][64 + d] = init_h.w_t[c][d] / 256.0;
        }
        g_b[c] = init_h.b[c] / 65536.0;
    }
    bpf_map_update_elem(g_head_fd, &k0, &init_h, BPF_ANY);

    uint8_t zbuf[512] = {0};
    uint64_t *zh = (uint64_t *)zbuf;
    for (int g = 0; g < 16; g++) zh[g] = PACK4(16384ULL);
    bpf_map_update_elem(bpf_object__find_map_fd_by_name(obj, "zero_ino_map"), &k0, zbuf, BPF_ANY);

    struct bpf_link *links[8] = {0}; int nl = 0;
    struct bpf_program *p;
    bpf_object__for_each_program(p, obj) {
        struct bpf_link *l = bpf_program__attach(p);
        if (libbpf_get_error(l)) { fprintf(stderr, "attach failed %s\n", bpf_program__name(p)); return 1; }
        links[nl++] = l;
    }

    struct ring_buffer *rb = ring_buffer__new(bpf_object__find_map_fd_by_name(obj, "msg_rb"), handle_msg, NULL, NULL);
    fprintf(stderr, "READY kdev=0x%x hooks=%d head_bytes=%zu\n", kdev, nl, sizeof(struct kssm_life_head));
    fflush(stderr);

    int ticks = 0;
    while (!g_stop) {
        ring_buffer__poll(rb, 50);
        if (++ticks >= 40) { /* every ~2s sync calibrated <1KB readout head atomically back into kernel! */
            ticks = 0;
            if (g_dirty_head) {
                sync_head_to_kernel();
                g_dirty_head = 0;
            }
        }
    }
    ring_buffer__consume(rb);
    if (g_dirty_head) sync_head_to_kernel();

    printf("=== LIVE WRITE-PATH + ZERO-BPTT SLOW-LOOP VERIFICATION ===\n");
    bpf_object__for_each_program(p, obj) {
        struct bpf_prog_info info = {}; uint32_t ilen = sizeof(info);
        if (bpf_obj_get_info_by_fd(bpf_program__fd(p), &info, &ilen) == 0) {
            double avg = info.run_cnt ? (double)info.run_time_ns / info.run_cnt : 0.0;
            printf("  prog %-15s | run_cnt=%8llu | run_time_ms=%7.2f | avg_ns=%.2f\n",
                   bpf_program__name(p), (unsigned long long)info.run_cnt, info.run_time_ns / 1e6, avg);
        }
    }
    printf("  births=%llu | F_SET_RW_HINT applied=%llu | F_GET_RW_HINT verified=%llu\n",
           (unsigned long long)g_births, (unsigned long long)g_hints_applied, (unsigned long long)g_hints_verified);
    printf("  deaths_feedback=%llu | slow_loop_kernel_map_syncs=%llu (head=784 B) | live_acc=%.1f%%\n",
           (unsigned long long)g_deaths, (unsigned long long)g_slowloop_syncs,
           g_deaths ? (100.0 * g_correct_preds / g_deaths) : 0.0);

    ring_buffer__free(rb);
    for (int i = 0; i < nl; i++) bpf_link__destroy(links[i]);
    bpf_object__close(obj);
    return 0;
}
