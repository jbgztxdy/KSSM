#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <sched.h>
#include <time.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "kssm_common.h"

struct kssm_swar_tables {
    uint64_t emb_i[KSSM_TOTAL_SLOTS][16];
    uint64_t emb_t[KSSM_TOTAL_SLOTS][16];
};

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int sig) { (void)sig; g_stop = 1; }

static int libbpf_quiet(enum libbpf_print_level level, const char *fmt, va_list args)
{
    if (level == LIBBPF_DEBUG) return 0;
    return vfprintf(stderr, fmt, args);
}

static void init_all_maps(struct bpf_object *obj)
{
    uint32_t seed = 0x12345678;
    #define NEXT_R() (seed = seed * 1664525u + 1013904223u)
    uint32_t k0 = 0;

    int swar_fd = bpf_object__find_map_fd_by_name(obj, "swar_tbl_map");
    if (swar_fd >= 0) {
        struct kssm_swar_tables *st = calloc(1, sizeof(*st));
        /* Strictly enforce contractive invariant set per timescale band:
         * d=1,c=2 -> |e|<=250; d=2,c=3 -> |e|<=120; d=3,c=4 -> |e|<=60; d=4,c=5 -> |e|<=30 */
        const int max_e[4] = {240, 120, 60, 30};
        for (int s = 0; s < KSSM_TOTAL_SLOTS; s++) {
            for (int g = 0; g < 16; g++) {
                int band = g >> 2;
                int lim = max_e[band];
                uint64_t wi = 0, wt = 0;
                for (int lane = 0; lane < 4; lane++) {
                    uint16_t vi = (uint16_t)(2048 + ((int)(NEXT_R() % (2 * lim + 1)) - lim));
                    uint16_t vt = (uint16_t)(2048 + ((int)(NEXT_R() % (2 * lim + 1)) - lim));
                    wi |= ((uint64_t)vi << (lane * 16));
                    wt |= ((uint64_t)vt << (lane * 16));
                }
                st->emb_i[s][g] = wi;
                st->emb_t[s][g] = wt;
            }
        }
        bpf_map_update_elem(swar_fd, &k0, st, BPF_ANY);
        free(st);
    }

    int ro_fd = bpf_object__find_map_fd_by_name(obj, "readout_map");
    if (ro_fd >= 0) {
        struct kssm_readout *ro = calloc(1, sizeof(*ro));
        for (int c = 0; c < KSSM_DIM; c++) {
            ro->w_scan_i[c] = (int16_t)((int)(NEXT_R() % 257) - 128);
            ro->w_scan_t[c] = (int16_t)((int)(NEXT_R() % 257) - 128);
            ro->w_life_i[c] = (int16_t)((int)(NEXT_R() % 257) - 128);
            ro->w_life_t[c] = (int16_t)((int)(NEXT_R() % 257) - 128);
        }
        ro->b_scan = 64;
        ro->b_life = -32;
        bpf_map_update_elem(ro_fd, &k0, ro, BPF_ANY);
        free(ro);
    }

    int z_fd = bpf_object__find_map_fd_by_name(obj, "zero_inode_map");
    if (z_fd >= 0) {
        struct kssm_inode_state z = {0};
        for (int c = 0; c < KSSM_DIM; c++) z.hidden.h[c] = 16384;
        bpf_map_update_elem(z_fd, &k0, &z, BPF_ANY);
    }

    int b_fd = bpf_object__find_map_fd_by_name(obj, "bench_state_map");
    if (b_fd >= 0) {
        int ncpus = libbpf_num_possible_cpus();
        struct {
            struct kssm_hidden h_i, h_t;
            uint32_t s1, s2, s3, s4, s5;
            int32_t sink;
        } *vals = calloc(ncpus, sizeof(*vals));
        for (int cpu = 0; cpu < ncpus; cpu++)
            for (int c = 0; c < KSSM_DIM; c++) {
                vals[cpu].h_i.h[c] = 16384;
                vals[cpu].h_t.h[c] = 16384;
            }
        bpf_map_update_elem(b_fd, &k0, vals, BPF_ANY);
        free(vals);
    }
}

static void print_prog_stats(int prog_fd, const char *name, const char *vlog, int attached)
{
    struct bpf_prog_info info = {};
    uint32_t len = sizeof(info);
    if (bpf_obj_get_info_by_fd(prog_fd, &info, &len) == 0) {
        const char *ver_line = strstr(vlog, "processed ");
        char vbuf[160] = "N/A";
        if (ver_line) {
            const char *nl = strchr(ver_line, 10);
            size_t n = nl ? (size_t)(nl - ver_line) : strlen(ver_line);
            if (n >= sizeof(vbuf)) n = sizeof(vbuf) - 1;
            memcpy(vbuf, ver_line, n);
            vbuf[n] = 0;
        }
        printf("  [PASS%s] %-26s | xlated=%4u | jited=%5u B | %s\n",
               attached ? "+ATTACH" : "",
               name, info.xlated_prog_len / 8, info.jited_prog_len, vbuf);
    }
}

static void verify_and_attach(const char *obj_path, const char *prog_name)
{
    static char vlog[262144];
    memset(vlog, 0, sizeof(vlog));
    LIBBPF_OPTS(bpf_object_open_opts, open_opts,
        .kernel_log_buf = vlog, .kernel_log_size = sizeof(vlog), .kernel_log_level = 4);
    struct bpf_object *obj = bpf_object__open_file(obj_path, &open_opts);
    struct bpf_program *p;
    bpf_object__for_each_program(p, obj)
        bpf_program__set_autoload(p, strcmp(bpf_program__name(p), prog_name) == 0);
    if (bpf_object__load(obj) != 0) {
        printf("  [FAIL] %-26s\n", prog_name);
        size_t l = strlen(vlog);
        printf("%s\n", l > 800 ? vlog + l - 800 : vlog);
        bpf_object__close(obj);
        return;
    }
    p = bpf_object__find_program_by_name(obj, prog_name);
    struct bpf_link *lnk = bpf_program__attach(p);
    int ok = !libbpf_get_error(lnk);
    print_prog_stats(bpf_program__fd(p), prog_name, vlog, ok);
    if (ok) bpf_link__destroy(lnk);
    bpf_object__close(obj);
}

static int cmp_double(const void *a, const void *b)
{
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static double run_xdp_bench(const char *obj_path, const char *prog_name, int repeat, int trials)
{
    static char vlog[262144];
    memset(vlog, 0, sizeof(vlog));
    LIBBPF_OPTS(bpf_object_open_opts, open_opts,
        .kernel_log_buf = vlog, .kernel_log_size = sizeof(vlog), .kernel_log_level = 4);
    struct bpf_object *obj = bpf_object__open_file(obj_path, &open_opts);
    struct bpf_program *p;
    bpf_object__for_each_program(p, obj)
        bpf_program__set_autoload(p, strcmp(bpf_program__name(p), prog_name) == 0);
    if (bpf_object__load(obj) != 0) { printf("  [FAIL] %s\n", prog_name); return -1; }
    init_all_maps(obj);
    p = bpf_object__find_program_by_name(obj, prog_name);
    int fd = bpf_program__fd(p);
    print_prog_stats(fd, prog_name, vlog, 0);

    unsigned char pkt[64] = {0};
    LIBBPF_OPTS(bpf_test_run_opts, topts,
        .data_in = pkt, .data_size_in = sizeof(pkt), .repeat = 200000);
    bpf_prog_test_run_opts(fd, &topts);

    double res[16];
    topts.repeat = repeat;
    for (int t = 0; t < trials; t++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
        bpf_prog_test_run_opts(fd, &topts);
        clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
        res[t] = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / repeat;
    }
    qsort(res, trials, sizeof(double), cmp_double);
    printf("         -> median=%.2f ns | min=%.2f ns | max=%.2f ns | kern_dur=%u ns\n",
           res[trials / 2], res[0], res[trials - 1], topts.duration);
    bpf_object__close(obj);
    return res[trials / 2];
}

static int run_vfs_daemon(const char *obj_path, const char *variant)
{
    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);
    struct bpf_object *obj = bpf_object__open_file(obj_path, NULL);
    if (!obj) return 1;

    struct bpf_map *rodata = NULL, *m;
    bpf_object__for_each_map(m, obj)
        if (strstr(bpf_map__name(m), ".rodata")) { rodata = m; break; }

    int is_empty = (strcmp(variant, "empty") == 0);
    struct bpf_program *p;
    bpf_object__for_each_program(p, obj) {
        const char *n = bpf_program__name(p);
        bool load = is_empty ?
            (strcmp(n, "ext4_empty_read") == 0 || strcmp(n, "ext4_empty_write") == 0) :
            (strcmp(n, "ext4_kssm_read") == 0 || strcmp(n, "ext4_kssm_write") == 0 ||
             strcmp(n, "vfs_kssm_open") == 0 || strcmp(n, "vfs_kssm_unlink") == 0);
        bpf_program__set_autoload(p, load);
    }

    if (!is_empty && rodata) {
        size_t sz = 0;
        uint32_t *vals = (uint32_t *)bpf_map__initial_value(rodata, &sz);
        if (vals && sz >= 12) {
            uint32_t use_stk = 1, use_gate = 1, mode = 2;
            if (strcmp(variant, "gated_subspace") == 0)           { use_stk = 1; use_gate = 1; mode = 2; }
            else if (strcmp(variant, "gated_subspace_nostk") == 0){ use_stk = 0; use_gate = 1; mode = 2; }
            else if (strcmp(variant, "ungated_subspace") == 0)    { use_stk = 1; use_gate = 0; mode = 2; }
            else if (strcmp(variant, "ungated_subspace_nostk") == 0){ use_stk = 0; use_gate = 0; mode = 2; }
            else if (strcmp(variant, "gated_shared") == 0)        { use_stk = 1; use_gate = 1; mode = 3; }
            else if (strcmp(variant, "gated_shared_nostk") == 0)  { use_stk = 0; use_gate = 1; mode = 3; }
            else if (strcmp(variant, "ungated_shared_nostk") == 0){ use_stk = 0; use_gate = 0; mode = 3; }
            vals[0] = use_stk; vals[1] = use_gate; vals[2] = mode;
        }
    }

    if (bpf_object__load(obj) != 0) { fprintf(stderr, "daemon load failed\n"); return 1; }
    if (!is_empty) init_all_maps(obj);

    struct bpf_link *links[8] = {0};
    int nl = 0;
    bpf_object__for_each_program(p, obj) {
        if (!bpf_program__autoload(p)) continue;
        struct bpf_link *l = bpf_program__attach(p);
        if (libbpf_get_error(l)) { fprintf(stderr, "attach failed %s\n", bpf_program__name(p)); return 1; }
        links[nl++] = l;
    }

    fprintf(stderr, "READY\n");
    fflush(stderr);
    while (!g_stop) usleep(50000);

    printf("=== BPF DAEMON STATS (%s) ===\n", variant);
    bpf_object__for_each_program(p, obj) {
        if (!bpf_program__autoload(p)) continue;
        struct bpf_prog_info info = {};
        uint32_t ilen = sizeof(info);
        if (bpf_obj_get_info_by_fd(bpf_program__fd(p), &info, &ilen) == 0) {
            double avg_ns = info.run_cnt ? (double)info.run_time_ns / info.run_cnt : 0.0;
            printf("  prog %-18s | run_cnt=%10llu | run_time_ms=%8.2f | avg_ns=%.2f\n",
                   bpf_program__name(p), (unsigned long long)info.run_cnt,
                   (double)info.run_time_ns / 1e6, avg_ns);
        }
    }

    if (!is_empty) {
        int sfd = bpf_object__find_map_fd_by_name(obj, "stats_map");
        int ncpus = libbpf_num_possible_cpus();
        struct kssm_cpu_stats *vals = calloc(ncpus, sizeof(*vals));
        uint32_t k0 = 0;
        if (vals && bpf_map_lookup_elem(sfd, &k0, vals) == 0) {
            struct kssm_cpu_stats tot = {0};
            for (int i = 0; i < ncpus; i++) {
                tot.total_hooks += vals[i].total_hooks;
                tot.ssm_updates += vals[i].ssm_updates;
                tot.fast_skips += vals[i].fast_skips;
            }
            double trig = tot.total_hooks ? 100.0 * tot.ssm_updates / tot.total_hooks : 0.0;
            printf("  hooks=%llu | ssm_updates=%llu (%.2f%%) | fast_skips=%llu\n",
                   (unsigned long long)tot.total_hooks, (unsigned long long)tot.ssm_updates,
                   trig, (unsigned long long)tot.fast_skips);
        }
        free(vals);
    }
    fflush(stdout);
    for (int i = 0; i < nl; i++) bpf_link__destroy(links[i]);
    bpf_object__close(obj);
    return 0;
}

int main(int argc, char **argv)
{
    libbpf_set_print(libbpf_quiet);
    const char *obj = "/home/yjw/workspace/research/kssm_probe/exp0a/kssm_exp0a_v4.bpf.o";
    if (argc >= 3 && strcmp(argv[1], "--daemon") == 0)
        return run_vfs_daemon(obj, argv[2]);

    cpu_set_t set; CPU_ZERO(&set); CPU_SET(2, &set);
    sched_setaffinity(0, sizeof(set), &set);

    printf("===========================================================================\n");
    printf(" PART 1: Linux 5.15.0-177 Verifier & Attach Verification (Zero Patch)\n");
    printf("===========================================================================\n");
    verify_and_attach(obj, "lsm_kssm_native_swar_64d");
    verify_and_attach(obj, "ext4_kssm_read");
    verify_and_attach(obj, "ext4_kssm_write");
    verify_and_attach(obj, "vfs_kssm_open");
    verify_and_attach(obj, "vfs_kssm_unlink");

    printf("\n===========================================================================\n");
    printf(" PART 2: Guard-Bit SWAR 64D + Fast8 Readout Microbenchmarks (2.10 GHz)\n");
    printf("===========================================================================\n");
    int repeat = 500000, trials = 7;
    double noop = run_xdp_bench(obj, "bench_noop", repeat, trials);
    double sub8 = run_xdp_bench(obj, "bench_subspace_ro1head", repeat, trials);
    double shr8 = run_xdp_bench(obj, "bench_shared_ro1head", repeat, trials);
    printf("\n=== SUMMARY (Net SSM + Fast8 Readout Latency) ===\n");
    printf("  Guard-SWAR 4-Lane 64D (Subspace + Fast8 Readout): %.2f ns\n", sub8 - noop);
    printf("  Guard-SWAR 4-Lane 64D (Shared   + Fast8 Readout): %.2f ns\n", shr8 - noop);
    return 0;
}
