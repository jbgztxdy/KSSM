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

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int sig) { (void)sig; g_stop = 1; }

static int libbpf_quiet(enum libbpf_print_level level, const char *fmt, va_list args)
{
    if (level == LIBBPF_DEBUG) return 0;
    return vfprintf(stderr, fmt, args);
}

static void init_tables(int tbl_fd, int ro_fd)
{
    struct kssm_tables *tbl = calloc(1, sizeof(*tbl));
    struct kssm_readout *ro = calloc(1, sizeof(*ro));
    if (!tbl || !ro) { perror("calloc"); exit(1); }

    uint32_t seed = 0x12345678;
    #define NEXT_RAND() (seed = seed * 1664525u + 1013904223u)

    for (int s = 0; s < KSSM_TOTAL_SLOTS; s++) {
        for (int c = 0; c < KSSM_DIM; c++) {
            tbl->emb_inode[s][c] = (int16_t)((int)(NEXT_RAND() >> 23) - 256);
            tbl->emb_task[s][c]  = (int16_t)((int)(NEXT_RAND() >> 23) - 256);
        }
    }
    for (int s = 0; s < KSSM_S1_SIZE; s++) {
        for (int c = 0; c < KSSM_DIM; c++) {
            /* Q7.8: alpha in [0.5, 0.96] -> [128, 246], gamma/eta in [-0.25, 0.25] -> [-64, 64] */
            tbl->alpha_inode[s][c] = (int16_t)(128 + (NEXT_RAND() % 118));
            tbl->alpha_task[s][c]  = (int16_t)(128 + (NEXT_RAND() % 118));
            tbl->gamma_t2i[s][c]   = (int16_t)((int)(NEXT_RAND() % 129) - 64);
            tbl->eta_i2t[s][c]     = (int16_t)((int)(NEXT_RAND() % 129) - 64);

            tbl->sh_decay_i[s][c] = (uint8_t)(1 + (c % 4));
            tbl->sh_coup_i[s][c]  = (uint8_t)(2 + (c % 4));
            tbl->sh_decay_t[s][c] = (uint8_t)(1 + (c % 4));
            tbl->sh_coup_t[s][c]  = (uint8_t)(2 + (c % 4));
        }
    }
    for (int c = 0; c < KSSM_DIM; c++) {
        ro->w_scan_i[c] = (int16_t)((int)(NEXT_RAND() % 257) - 128);
        ro->w_scan_t[c] = (int16_t)((int)(NEXT_RAND() % 257) - 128);
        ro->w_life_i[c] = (int16_t)((int)(NEXT_RAND() % 257) - 128);
        ro->w_life_t[c] = (int16_t)((int)(NEXT_RAND() % 257) - 128);
    }
    ro->b_scan = 128;
    ro->b_life = -64;

    uint32_t k0 = 0;
    if (tbl_fd >= 0 && bpf_map_update_elem(tbl_fd, &k0, tbl, BPF_ANY) != 0)
        perror("update tbl_map");
    if (ro_fd >= 0 && bpf_map_update_elem(ro_fd, &k0, ro, BPF_ANY) != 0)
        perror("update readout_map");

    free(tbl);
    free(ro);
}

static void print_prog_stats(int prog_fd, const char *name, const char *vlog)
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
        printf("  [OK] %-32s | xlated=%4u insns | jited=%5u B | %s\n",
               name, info.xlated_prog_len / 8, info.jited_prog_len, vbuf);
    }
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
        .kernel_log_buf = vlog,
        .kernel_log_size = sizeof(vlog),
        .kernel_log_level = 4, /* BPF_LOG_STATS */
    );

    struct bpf_object *obj = bpf_object__open_file(obj_path, &open_opts);
    if (!obj) { fprintf(stderr, "open failed\n"); return -1; }

    struct bpf_program *p;
    bpf_object__for_each_program(p, obj) {
        bpf_program__set_autoload(p, strcmp(bpf_program__name(p), prog_name) == 0);
    }

    if (bpf_object__load(obj) != 0) {
        fprintf(stderr, "LOAD FAILED for %s\n%s\n", prog_name, vlog);
        bpf_object__close(obj);
        return -1;
    }

    int tbl_fd = bpf_object__find_map_fd_by_name(obj, "tbl_map");
    int ro_fd  = bpf_object__find_map_fd_by_name(obj, "readout_map");
    init_tables(tbl_fd, ro_fd);

    p = bpf_object__find_program_by_name(obj, prog_name);
    int fd = bpf_program__fd(p);
    print_prog_stats(fd, prog_name, vlog);

    unsigned char pkt[64] = {0};
    LIBBPF_OPTS(bpf_test_run_opts, topts,
        .data_in = pkt,
        .data_size_in = sizeof(pkt),
        .repeat = 200000,
    );
    /* Warmup */
    if (bpf_prog_test_run_opts(fd, &topts) != 0) {
        perror("bpf_prog_test_run_opts warmup");
        bpf_object__close(obj);
        return -1;
    }

    double res[16];
    topts.repeat = repeat;
    for (int t = 0; t < trials; t++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
        if (bpf_prog_test_run_opts(fd, &topts) != 0) {
            perror("bpf_prog_test_run_opts");
            bpf_object__close(obj);
            return -1;
        }
        clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
        double wall_ns = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / repeat;
        /* opts.duration is integer ns per run rounded; wall_ns has sub-ns precision */
        res[t] = wall_ns;
    }
    qsort(res, trials, sizeof(double), cmp_double);
    double med = res[trials / 2];
    double minv = res[0];
    double maxv = res[trials - 1];
    printf("       -> repeat=%d x %d trials | median=%.2f ns | min=%.2f ns | max=%.2f ns | kern_dur=%u ns\n",
           repeat, trials, med, minv, maxv, topts.duration);

    bpf_object__close(obj);
    return med;
}

static void verify_single_prog(const char *obj_path, const char *prog_name)
{
    static char vlog[262144];
    memset(vlog, 0, sizeof(vlog));

    LIBBPF_OPTS(bpf_object_open_opts, open_opts,
        .kernel_log_buf = vlog,
        .kernel_log_size = sizeof(vlog),
        .kernel_log_level = 4,
    );

    struct bpf_object *obj = bpf_object__open_file(obj_path, &open_opts);
    if (!obj) { fprintf(stderr, "open failed\n"); return; }

    struct bpf_program *p;
    bpf_object__for_each_program(p, obj) {
        bpf_program__set_autoload(p, strcmp(bpf_program__name(p), prog_name) == 0);
    }

    if (bpf_object__load(obj) != 0) {
        fprintf(stderr, "  [FAIL] %s\n", prog_name);
        bpf_object__close(obj);
        return;
    }
    p = bpf_object__find_program_by_name(obj, prog_name);
    print_prog_stats(bpf_program__fd(p), prog_name, vlog);
    bpf_object__close(obj);
}

static int run_vfs_daemon(const char *obj_path, const char *variant)
{
    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);

    struct bpf_object *obj = bpf_object__open_file(obj_path, NULL);
    if (!obj) { fprintf(stderr, "open failed\n"); return 1; }

    /* Configure rodata globals before load */
    struct bpf_map *rodata = NULL;
    struct bpf_map *m;
    bpf_object__for_each_map(m, obj) {
        if (strstr(bpf_map__name(m), ".rodata")) {
            rodata = m;
            break;
        }
    }

    /* Select programs based on variant */
    int is_empty = (strcmp(variant, "empty") == 0);
    struct bpf_program *p;
    bpf_object__for_each_program(p, obj) {
        const char *n = bpf_program__name(p);
        bool load = false;
        if (is_empty) {
            load = (strcmp(n, "vfs_empty_read") == 0 || strcmp(n, "vfs_empty_write") == 0);
        } else {
            load = (strcmp(n, "vfs_kssm_read") == 0 ||
                    strcmp(n, "vfs_kssm_write") == 0 ||
                    strcmp(n, "vfs_kssm_open") == 0 ||
                    strcmp(n, "vfs_kssm_unlink") == 0);
        }
        bpf_program__set_autoload(p, load);
    }

    /* Set toggles via rodata initial value */
    if (!is_empty && rodata) {
        size_t sz = 0;
        uint32_t *vals = (uint32_t *)bpf_map__initial_value(rodata, &sz);
        if (vals && sz >= 12) {
            /* Order matches declaration: cfg_use_stackid, cfg_use_gating, cfg_use_shift_ssm */
            uint32_t use_stack = 1, use_gate = 1, ssm_mode = 0;
            if (strcmp(variant, "gated_exact") == 0)          { use_stack = 1; use_gate = 1; ssm_mode = 0; }
            else if (strcmp(variant, "gated_exact_nostk") == 0){ use_stack = 0; use_gate = 1; ssm_mode = 0; }
            else if (strcmp(variant, "ungated_exact") == 0)   { use_stack = 1; use_gate = 0; ssm_mode = 0; }
            else if (strcmp(variant, "ungated_exact_nostk") == 0){ use_stack = 0; use_gate = 0; ssm_mode = 0; }
            else if (strcmp(variant, "gated_shift") == 0)     { use_stack = 1; use_gate = 1; ssm_mode = 1; }
            else if (strcmp(variant, "gated_swar") == 0)      { use_stack = 1; use_gate = 1; ssm_mode = 2; }
            else if (strcmp(variant, "gated_swar_nostk") == 0){ use_stack = 0; use_gate = 1; ssm_mode = 2; }
            else if (strcmp(variant, "ungated_swar_nostk") == 0){ use_stack = 0; use_gate = 0; ssm_mode = 2; }
            vals[0] = use_stack;
            vals[1] = use_gate;
            vals[2] = ssm_mode;
            fprintf(stderr, "[daemon] variant=%s (stackid=%u, gating=%u, ssm_mode=%u)\n",
                    variant, use_stack, use_gate, ssm_mode);
        }
    }

    if (bpf_object__load(obj) != 0) {
        fprintf(stderr, "daemon load failed\n");
        bpf_object__close(obj);
        return 1;
    }

    if (!is_empty) {
        init_tables(bpf_object__find_map_fd_by_name(obj, "tbl_map"),
                    bpf_object__find_map_fd_by_name(obj, "readout_map"));
    }

    struct bpf_link *links[8] = {0};
    int nlinks = 0;
    bpf_object__for_each_program(p, obj) {
        if (!bpf_program__autoload(p)) continue;
        struct bpf_link *l = bpf_program__attach(p);
        if (libbpf_get_error(l)) {
            fprintf(stderr, "attach failed for %s\n", bpf_program__name(p));
            return 1;
        }
        links[nlinks++] = l;
    }

    fprintf(stderr, "READY\n");
    fflush(stderr);

    while (!g_stop)
        usleep(50000);

    /* Read kernel bpf_stats & per-CPU counters */
    printf("=== BPF DAEMON STATS (%s) ===\n", variant);
    bpf_object__for_each_program(p, obj) {
        if (!bpf_program__autoload(p)) continue;
        struct bpf_prog_info info = {};
        uint32_t ilen = sizeof(info);
        if (bpf_obj_get_info_by_fd(bpf_program__fd(p), &info, &ilen) == 0) {
            double avg_ns = info.run_cnt ? (double)info.run_time_ns / info.run_cnt : 0.0;
            printf("  prog %-20s | run_cnt=%10llu | run_time_ms=%8.2f | avg_ns=%.2f\n",
                   bpf_program__name(p),
                   (unsigned long long)info.run_cnt,
                   (double)info.run_time_ns / 1e6,
                   avg_ns);
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
                tot.scan_preds += vals[i].scan_preds;
                tot.short_life_preds += vals[i].short_life_preds;
            }
            double trig_pct = tot.total_hooks ? 100.0 * tot.ssm_updates / tot.total_hooks : 0.0;
            printf("  hooks=%llu | ssm_updates=%llu (%.2f%%) | fast_skips=%llu\n",
                   (unsigned long long)tot.total_hooks,
                   (unsigned long long)tot.ssm_updates,
                   trig_pct,
                   (unsigned long long)tot.fast_skips);
        }
        free(vals);
    }
    fflush(stdout);

    for (int i = 0; i < nlinks; i++)
        bpf_link__destroy(links[i]);
    bpf_object__close(obj);
    return 0;
}

int main(int argc, char **argv)
{
    libbpf_set_print(libbpf_quiet);
    const char *obj = "/home/yjw/workspace/research/kssm_probe/exp0a/kssm_exp0a.bpf.o";

    if (argc >= 3 && strcmp(argv[1], "--daemon") == 0) {
        return run_vfs_daemon(obj, argv[2]);
    }

    /* Pin to CPU 0 for consistent microbenchmark frequency */
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(2, &set);
    sched_setaffinity(0, sizeof(set), &set);

    printf("===========================================================================\n");
    printf(" PART 1: Linux 5.15 eBPF Verifier & JIT Check (Zero Kernel Patch)\n");
    printf("===========================================================================\n");
    verify_single_prog(obj, "lsm_kssm_native_64d");
    verify_single_prog(obj, "vfs_kssm_read");
    verify_single_prog(obj, "vfs_kssm_write");
    verify_single_prog(obj, "vfs_kssm_open");
    verify_single_prog(obj, "vfs_kssm_unlink");

    printf("\n===========================================================================\n");
    printf(" PART 2: In-Kernel Isolated Microbenchmarks (BPF_PROG_TEST_RUN, CPU 2)\n");
    printf("===========================================================================\n");
    int repeat = 500000, trials = 7;
    double base = run_xdp_bench(obj, "bench_v0_noop", repeat, trials);
    double v1   = run_xdp_bench(obj, "bench_v1_exact_q78_64d", repeat, trials);
    double v1b  = run_xdp_bench(obj, "bench_v1b_exact_q78_64d_noreadout", repeat, trials);
    double v2   = run_xdp_bench(obj, "bench_v2_shift_gate_64d", repeat, trials);
    double v2b  = run_xdp_bench(obj, "bench_v2b_shift_gate_64d_noreadout", repeat, trials);
    double v3   = run_xdp_bench(obj, "bench_v3_swar_4lane_64d", repeat, trials);
    double v4   = run_xdp_bench(obj, "bench_v4_exact_q78_32d", repeat, trials);

    printf("\n=== SUMMARY (Net SSM Latency = Variant - Noop %.2f ns) ===\n", base);
    printf("  Exact Q7.8 64D (SSM + Dual Readout): %.2f ns\n", v1 - base);
    printf("  Exact Q7.8 64D (SSM only)          : %.2f ns\n", v1b - base);
    printf("  Shift-Gate 64D (SSM + Dual Readout): %.2f ns\n", v2 - base);
    printf("  Shift-Gate 64D (SSM only)          : %.2f ns\n", v2b - base);
    printf("  SWAR 4-Lane 64D (16x u64 packed)   : %.2f ns\n", v3 - base);
    printf("  Exact Q7.8 32D (SSM + Dual Readout): %.2f ns\n", v4 - base);

    return 0;
}
