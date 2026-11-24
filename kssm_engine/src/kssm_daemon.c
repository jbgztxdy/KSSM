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
#include <sched.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "../include/kssm_abi.h"

#ifndef F_SET_RW_HINT
#define F_SET_RW_HINT 1036
#define RWH_WRITE_LIFE_SHORT  2
#define RWH_WRITE_LIFE_MEDIUM 3
#define RWH_WRITE_LIFE_LONG   4
#endif
#define PACK4(v) (((uint64_t)(v)) | ((uint64_t)(v)<<16) | ((uint64_t)(v)<<32) | ((uint64_t)(v)<<48))

static volatile sig_atomic_t g_stop = 0;
static int g_head_fd = -1;
static double g_Wl[3][128], g_bl[3];
static int16_t g_ws_i[64], g_ws_t[64];
static int32_t g_bs = 0;
static int g_dirty = 0;
static uint64_t g_drop_calls = 0, g_drop_bytes = 0, g_birth_hints = 0, g_deaths = 0, g_syncs = 0;

static void on_sig(int s) { (void)s; g_stop = 1; }
static int libbpf_quiet(enum libbpf_print_level l, const char *fmt, va_list a) {
    return (l == LIBBPF_DEBUG) ? 0 : vfprintf(stderr, fmt, a);
}
static int cmp_d(const void *a, const void *b) { double x=*(const double*)a, y=*(const double*)b; return (x>y)-(x<y); }

static int open_proc_ino(uint32_t pid, uint64_t ino)
{
    char dirp[64]; snprintf(dirp, sizeof(dirp), "/proc/%u/fd", pid);
    DIR *d = opendir(dirp); if (!d) return -1;
    struct dirent *e; int fd = -1;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') continue;
        char fdp[300]; snprintf(fdp, sizeof(fdp), "/proc/%u/fd/%s", pid, e->d_name);
        struct stat st;
        if (stat(fdp, &st) == 0 && (uint64_t)st.st_ino == ino) {
            fd = open(fdp, O_RDONLY);
            break;
        }
    }
    closedir(d);
    return fd;
}

static void sync_dual_head(void)
{
    if (g_head_fd < 0) return;
    struct kssm_dual_head h = {0};
    memcpy(h.w_scan_i, g_ws_i, sizeof(g_ws_i));
    memcpy(h.w_scan_t, g_ws_t, sizeof(g_ws_t));
    h.b_scan = g_bs;
    for (int c = 0; c < 3; c++) {
        for (int d = 0; d < 64; d++) {
            double vi = g_Wl[c][d] * 256.0, vt = g_Wl[c][64+d] * 256.0;
            if (vi > 32767) vi = 32767; if (vi < -32768) vi = -32768;
            if (vt > 32767) vt = 32767; if (vt < -32768) vt = -32768;
            h.w_life_i[c][d] = (int16_t)llround(vi);
            h.w_life_t[c][d] = (int16_t)llround(vt);
        }
        h.b_life[c] = (int32_t)llround(g_bl[c] * 65536.0);
    }
    uint32_t k0 = 0;
    if (bpf_map_update_elem(g_head_fd, &k0, &h, BPF_ANY) == 0) g_syncs++;
}

static int on_ctrl_msg(void *ctx, void *data, size_t sz)
{
    (void)ctx;
    if (sz != sizeof(struct kssm_ctrl_msg)) return 0;
    const struct kssm_ctrl_msg *m = (const struct kssm_ctrl_msg *)data;
    if (m->msg_type == KSSM_MSG_DROP_HINT) {
        int fd = open_proc_ino(m->pid, m->ino);
        if (fd >= 0) {
            posix_fadvise(fd, (off_t)m->arg0, (off_t)m->arg1, POSIX_FADV_DONTNEED);
            close(fd);
            g_drop_calls++; g_drop_bytes += m->arg1;
        }
    } else if (m->msg_type == KSSM_MSG_BIRTH_HINT) {
        int fd = open_proc_ino(m->pid, m->ino);
        if (fd >= 0) {
            uint64_t hint = (m->pred_class == 0) ? RWH_WRITE_LIFE_SHORT :
                            ((m->pred_class == 1) ? RWH_WRITE_LIFE_MEDIUM : RWH_WRITE_LIFE_LONG);
            if (fcntl(fd, F_SET_RW_HINT, &hint) == 0) g_birth_hints++;
            close(fd);
        }
    } else if (m->msg_type == KSSM_MSG_DEATH_FEEDBACK) {
        g_deaths++;
        double x[128];
        for (int c = 0; c < 64; c++) { x[c] = m->snap_hi[c]/256.0; x[64+c] = m->snap_ht[c]/256.0; }
        double lg[3], mx = -1e9, den = 0, pr[3];
        for (int k = 0; k < 3; k++) {
            double s = g_bl[k]; for (int j = 0; j < 128; j++) s += g_Wl[k][j] * x[j];
            lg[k] = s; if (s > mx) mx = s;
        }
        for (int k = 0; k < 3; k++) { pr[k] = exp(lg[k] - mx); den += pr[k]; }
        for (int k = 0; k < 3; k++) pr[k] /= den;
        double lr = (m->true_class > 0) ? 0.06 : 0.02;
        for (int k = 0; k < 3; k++) {
            double gr = pr[k] - (k == m->true_class ? 1.0 : 0.0);
            for (int j = 0; j < 128; j++) g_Wl[k][j] -= lr * gr * x[j];
            g_bl[k] -= lr * gr;
        }
        g_dirty = 1;
    }
    return 0;
}

static void load_models_into_maps(struct bpf_object *obj, const char *base_path, const char *head_path)
{
    uint32_t k0 = 0;
    int bfd = bpf_object__find_map_fd_by_name(obj, "base_map");
    if (bfd >= 0) {
        struct kssm_base_model *bm = calloc(1, sizeof(*bm));
        FILE *f = fopen(base_path, "rb");
        if (f && fread(bm, sizeof(*bm), 1, f) == 1) {
            bpf_map_update_elem(bfd, &k0, bm, BPF_ANY);
        }
        if (f) fclose(f);
        free(bm);
    }
    g_head_fd = bpf_object__find_map_fd_by_name(obj, "head_map");
    if (g_head_fd >= 0) {
        struct kssm_dual_head dh = {0};
        FILE *f = fopen(head_path, "rb");
        if (f && fread(&dh, sizeof(dh), 1, f) == 1) {
            memcpy(g_ws_i, dh.w_scan_i, sizeof(g_ws_i));
            memcpy(g_ws_t, dh.w_scan_t, sizeof(g_ws_t));
            g_bs = dh.b_scan;
            for (int c = 0; c < 3; c++) {
                for (int d = 0; d < 64; d++) {
                    g_Wl[c][d] = dh.w_life_i[c][d]/256.0;
                    g_Wl[c][64+d] = dh.w_life_t[c][d]/256.0;
                }
                g_bl[c] = dh.b_life[c]/65536.0;
            }
            bpf_map_update_elem(g_head_fd, &k0, &dh, BPF_ANY);
        }
        if (f) fclose(f);
    }
    int zfd = bpf_object__find_map_fd_by_name(obj, "zero_ino_map");
    if (zfd >= 0) {
        uint8_t zbuf[512] = {0}; uint64_t *zh = (uint64_t *)zbuf;
        for (int g = 0; g < 16; g++) zh[g] = PACK4(16384ULL);
        bpf_map_update_elem(zfd, &k0, zbuf, BPF_ANY);
    }
}

static int run_verify_and_microbench(const char *obj_path, const char *base_path, const char *head_path)
{
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(2, &set); sched_setaffinity(0, sizeof(set), &set);
    const char *progs[] = {
        "lsm_kssm_unified_native", "kssm_ext4_read", "kssm_ext4_write",
        "kssm_vfs_open", "kssm_vfs_unlink", "kssm_vfs_fsync"
    };
    printf("=== UNIFIED KSSM ENGINE: LINUX 5.15 VERIFIER & ATTACH CHECK ===\n");
    for (int i = 0; i < 6; i++) {
        static char vlog[262144]; memset(vlog, 0, sizeof(vlog));
        LIBBPF_OPTS(bpf_object_open_opts, oo, .kernel_log_buf=vlog, .kernel_log_size=sizeof(vlog), .kernel_log_level=4);
        struct bpf_object *obj = bpf_object__open_file(obj_path, &oo);
        struct bpf_program *p;
        bpf_object__for_each_program(p, obj) bpf_program__set_autoload(p, strcmp(bpf_program__name(p), progs[i])==0);
        if (bpf_object__load(obj) != 0) {
            printf("  [FAIL] %s\n%s\n", progs[i], vlog);
            bpf_object__close(obj); return 1;
        }
        p = bpf_object__find_program_by_name(obj, progs[i]);
        struct bpf_link *l = bpf_program__attach(p);
        int ok = !libbpf_get_error(l);
        struct bpf_prog_info info = {}; uint32_t ilen = sizeof(info);
        bpf_obj_get_info_by_fd(bpf_program__fd(p), &info, &ilen);
        const char *vl = strstr(vlog, "processed "); char vbuf[140]="N/A";
        if (vl) { const char *nl=strchr(vl,10); size_t n=nl?(size_t)(nl-vl):strlen(vl); if(n>139)n=139; memcpy(vbuf,vl,n); vbuf[n]=0; }
        printf("  [PASS%s] %-24s | xlated=%4u | jited=%5u B | %s\n",
               ok ? "+ATTACH" : "", progs[i], info.xlated_prog_len/8, info.jited_prog_len, vbuf);
        if (ok) bpf_link__destroy(l);
        bpf_object__close(obj);
    }

    printf("\n=== PRUNED g_s1[32][4] 4-BAND SELECTIVE GUARD-SWAR 64D MICROBENCH (CPU 2, 2.10 GHz) ===\n");
    double times[2] = {0, 0};
    const char *bnames[2] = {"bench_noop", "bench_unified_sel4"};
    for (int bi = 0; bi < 2; bi++) {
        struct bpf_object *obj = bpf_object__open_file(obj_path, NULL);
        struct bpf_program *p;
        bpf_object__for_each_program(p, obj) bpf_program__set_autoload(p, strcmp(bpf_program__name(p), bnames[bi])==0);
        if (bpf_object__load(obj) != 0) return 1;
        load_models_into_maps(obj, base_path, head_path);
        int ufd = bpf_object__find_map_fd_by_name(obj, "bench_u_map");
        if (ufd >= 0) {
            int nc = libbpf_num_possible_cpus();
            struct { uint64_t vi[16], vt[16]; uint32_t s[6]; } *v = calloc(nc, sizeof(*v));
            for (int c=0; c<nc; c++) for (int g=0; g<16; g++) { v[c].vi[g]=PACK4(16384ULL); v[c].vt[g]=PACK4(16384ULL); }
            uint32_t k0 = 0; bpf_map_update_elem(ufd, &k0, v, BPF_ANY); free(v);
        }
        p = bpf_object__find_program_by_name(obj, bnames[bi]);
        int fd = bpf_program__fd(p);
        unsigned char pkt[64]={0};
        LIBBPF_OPTS(bpf_test_run_opts, to, .data_in=pkt, .data_size_in=64, .repeat=200000);
        bpf_prog_test_run_opts(fd, &to);
        double res[7]; to.repeat = 500000;
        for (int t = 0; t < 7; t++) {
            struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
            bpf_prog_test_run_opts(fd, &to);
            clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
            res[t] = ((t1.tv_sec-t0.tv_sec)*1e9 + (t1.tv_nsec-t0.tv_nsec))/500000.0;
        }
        qsort(res, 7, sizeof(double), cmp_d);
        times[bi] = res[3];
        printf("  %-22s: median=%.2f ns | min=%.2f ns\n", bnames[bi], res[3], res[0]);
        bpf_object__close(obj);
    }
    printf("  -> Net Unified 4-Band Selective (g_s1[32][4], 10-app trained weights): %.2f ns\n", times[1] - times[0]);
    return 0;
}

int main(int argc, char **argv)
{
    libbpf_set_print(libbpf_quiet);
    const char *obj_path  = "/home/yjw/workspace/research/KSSM/kssm_engine/bpf/kssm_engine.bpf.o";
    const char *base_path = "/home/yjw/workspace/research/KSSM/kssm_engine/models/kssm_base.bin";
    const char *head_path = "/home/yjw/workspace/research/KSSM/kssm_engine/models/kssm_head.bin";

    if (argc >= 2 && strcmp(argv[1], "--verify") == 0)
        return run_verify_and_microbench(obj_path, base_path, head_path);

    if (argc < 4) {
        fprintf(stderr, "Usage:\n  %s --verify\n  %s <mount_dir> <tl_percpu|shared_inode|common_lru> <actions:0|1>\n", argv[0], argv[0]);
        return 1;
    }
    struct stat st; if (stat(argv[1], &st) != 0) { perror("stat"); return 1; }
    uint32_t kdev = (major(st.st_dev) << 20) | minor(st.st_dev);
    uint32_t gmode = (strcmp(argv[2], "shared_inode") == 0) ? KSSM_GATING_SHARED_INODE :
                     ((strcmp(argv[2], "common_lru") == 0) ? KSSM_GATING_COMMON_LRU : KSSM_GATING_TL_PERCPU);
    uint32_t en_act = (uint32_t)atoi(argv[3]);

    signal(SIGINT, on_sig); signal(SIGTERM, on_sig);

    struct bpf_object *obj = bpf_object__open_file(obj_path, NULL);
    if (!obj) return 1;
    struct bpf_map *m;
    bpf_object__for_each_map(m, obj) {
        if (strstr(bpf_map__name(m), ".rodata")) {
            size_t sz = 0; uint32_t *v = bpf_map__initial_value(m, &sz);
            if (v && sz >= 16) { v[0] = kdev; v[1] = gmode; v[2] = en_act; v[3] = 0; }
            break;
        }
    }
    struct bpf_program *p;
    bpf_object__for_each_program(p, obj) {
        const char *n = bpf_program__name(p);
        bool load = (strncmp(n, "kssm_", 5) == 0);
        bpf_program__set_autoload(p, load);
    }
    if (bpf_object__load(obj) != 0) { fprintf(stderr, "daemon load failed\n"); return 1; }
    load_models_into_maps(obj, base_path, head_path);

    struct bpf_link *links[8] = {0}; int nl = 0;
    bpf_object__for_each_program(p, obj) {
        if (!bpf_program__autoload(p)) continue;
        struct bpf_link *l = bpf_program__attach(p);
        if (libbpf_get_error(l)) { fprintf(stderr, "attach failed %s\n", bpf_program__name(p)); return 1; }
        links[nl++] = l;
    }
    struct ring_buffer *rb = ring_buffer__new(bpf_object__find_map_fd_by_name(obj, "ctrl_rb"), on_ctrl_msg, NULL, NULL);
    fprintf(stderr, "READY mode=%s actions=%u kdev=0x%x hooks=%d\n", argv[2], en_act, kdev, nl);
    fflush(stderr);

    int ticks = 0;
    while (!g_stop) {
        ring_buffer__poll(rb, 50);
        if (++ticks >= 40) { ticks = 0; if (g_dirty) { sync_dual_head(); g_dirty = 0; } }
    }
    ring_buffer__consume(rb);
    if (g_dirty) sync_dual_head();

    printf("=== KSSM UNIFIED ENGINE STATS (mode=%s) ===\n", argv[2]);
    bpf_object__for_each_program(p, obj) {
        if (!bpf_program__autoload(p)) continue;
        struct bpf_prog_info info = {}; uint32_t ilen = sizeof(info);
        if (bpf_obj_get_info_by_fd(bpf_program__fd(p), &info, &ilen) == 0) {
            double avg = info.run_cnt ? (double)info.run_time_ns / info.run_cnt : 0.0;
            printf("  prog %-16s | run_cnt=%10llu | run_time_ms=%8.2f | avg_ns=%.2f\n",
                   bpf_program__name(p), (unsigned long long)info.run_cnt, info.run_time_ns/1e6, avg);
        }
    }
    int mfd = bpf_object__find_map_fd_by_name(obj, "metrics_map");
    int nc = libbpf_num_possible_cpus();
    struct kssm_cpu_metrics *mv = calloc(nc, sizeof(*mv));
    uint32_t k0 = 0;
    if (mv && bpf_map_lookup_elem(mfd, &k0, mv) == 0) {
        struct kssm_cpu_metrics tot = {0};
        for (int i = 0; i < nc; i++) {
            tot.total_hooks += mv[i].total_hooks; tot.fast_skips += mv[i].fast_skips;
            tot.ssm_updates += mv[i].ssm_updates; tot.scan_votes += mv[i].scan_votes;
            tot.hot_votes   += mv[i].hot_votes;   tot.births_pred += mv[i].births_pred;
            tot.deaths_obs  += mv[i].deaths_obs;  tot.msgs_sent += mv[i].msgs_sent;
            tot.rb_drops    += mv[i].rb_drops;
        }
        double trig = tot.total_hooks ? 100.0 * tot.ssm_updates / tot.total_hooks : 0.0;
        printf("  hooks=%llu | skips=%llu | ssm_upd=%llu (%.2f%%) | drop_calls=%llu (%.2fG) | birth_hints=%llu | slow_syncs=%llu\n",
               (unsigned long long)tot.total_hooks, (unsigned long long)tot.fast_skips,
               (unsigned long long)tot.ssm_updates, trig, (unsigned long long)g_drop_calls,
               g_drop_bytes/(1024.0*1024.0*1024.0), (unsigned long long)g_birth_hints, (unsigned long long)g_syncs);
    }
    free(mv);
    ring_buffer__free(rb);
    for (int i = 0; i < nl; i++) bpf_link__destroy(links[i]);
    bpf_object__close(obj);
    return 0;
}
