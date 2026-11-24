#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sched.h>
#include <time.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "kssm_selective.h"

#define PACK4(v) (((uint64_t)(v)) | ((uint64_t)(v) << 16) | ((uint64_t)(v) << 32) | ((uint64_t)(v) << 48))

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void fill_sel_tables(int fd, int is_shared) {
    struct kssm_selective_tables *t = calloc(1, sizeof(*t));
    uint32_t seed = 0x99887766;
    #define NR() (seed = seed * 1664525u + 1013904223u)
    uint64_t boff = is_shared ? 10240ULL : 4096ULL;
    for (int s = 0; s < KSSM_TOTAL_SLOTS; s++) {
        for (int g = 0; g < 16; g++) {
            uint64_t w = 0;
            int lim = 30;
            for (int l = 0; l < 4; l++)
                w |= ((uint64_t)(uint16_t)(2048 + ((int)(NR() % (2*lim+1)) - lim)) << (l*16));
            t->emb[s][g] = w;
        }
    }
    for (int s1 = 0; s1 < KSSM_S1_SIZE; s1++) {
        for (int b = 0; b < 4; b++) {
            struct sel_band_gate *gd = &t->gate[s1][b];
            if (s1 % 5 == 0) {
                /* Mode A: Noise-Immune HOLD (alpha=1, gamma=0) */
                gd->keep_i = ~0ULL; gd->mask_di = 0; gd->mask_ci = 0; gd->d_i = 1; gd->c_i = 2; gd->const_i = PACK4(boff);
                gd->keep_t = ~0ULL; gd->mask_dt = 0; gd->mask_ct = 0; gd->d_t = 1; gd->c_t = 2; gd->const_t = PACK4(boff);
            } else if (s1 % 5 == 1) {
                /* Mode C: Phase RESET + Strong Task->Inode Injection (alpha_i=0, gamma=0.5) */
                gd->keep_i = 0ULL; gd->mask_di = 0; gd->d_i = 1; gd->c_i = 1; gd->mask_ci = PACK4(0x7FFFULL);
                gd->const_i = PACK4(boff + 8192ULL - 16384ULL); /* Note: if boff=4096, wait: use 16384 offset safely */
                if (boff < 8192ULL) {
                    /* Safe reset for 2-slot: alpha=0.5 (d=1), gamma=0.25 (c=2) -> const = 4096 + 4096 - 8192 = 0! */
                    gd->keep_i = ~0ULL; gd->d_i = 1; gd->c_i = 2;
                    gd->mask_di = PACK4(0x7FFFULL); gd->mask_ci = PACK4(0x3FFFULL);
                    gd->const_i = PACK4(boff + 4096ULL - 8192ULL);
                }
                gd->keep_t = ~0ULL; gd->d_t = b + 1; gd->c_t = b + 2;
                gd->mask_dt = PACK4(0xFFFFULL >> gd->d_t); gd->mask_ct = PACK4(0xFFFFULL >> gd->c_t);
                gd->const_t = PACK4(boff - (16384ULL >> gd->c_t) - (16384ULL >> gd->d_t));
            } else {
                /* Mode B: Dynamic Multi-Scale Decay */
                uint32_t d = (uint32_t)(b + 1);
                uint32_t c = (uint32_t)(b + 2);
                gd->keep_i = ~0ULL; gd->d_i = d; gd->c_i = c;
                gd->mask_di = PACK4(0xFFFFULL >> d); gd->mask_ci = PACK4(0xFFFFULL >> c);
                gd->const_i = PACK4(boff + (16384ULL >> c) - (16384ULL >> d));
                gd->keep_t = ~0ULL; gd->d_t = d; gd->c_t = c;
                gd->mask_dt = PACK4(0xFFFFULL >> d); gd->mask_ct = PACK4(0xFFFFULL >> c);
                gd->const_t = PACK4(boff - (16384ULL >> c) - (16384ULL >> d));
            }
        }
    }
    uint32_t k0 = 0;
    bpf_map_update_elem(fd, &k0, t, BPF_ANY);
    free(t);
}

static double bench_one(const char *obj_path, const char *name, int is_shared) {
    static char vlog[131072];
    memset(vlog, 0, sizeof(vlog));
    LIBBPF_OPTS(bpf_object_open_opts, oopts, .kernel_log_buf = vlog, .kernel_log_size = sizeof(vlog), .kernel_log_level = 4);
    struct bpf_object *obj = bpf_object__open_file(obj_path, &oopts);
    struct bpf_program *p;
    bpf_object__for_each_program(p, obj)
        bpf_program__set_autoload(p, strcmp(bpf_program__name(p), name) == 0);
    if (bpf_object__load(obj) != 0) { printf("[FAIL] %s\n%s\n", name, vlog); return -1; }

    int tfd = bpf_object__find_map_fd_by_name(obj, "sel_tbl_map");
    if (tfd >= 0) fill_sel_tables(tfd, is_shared);

    int bfd = bpf_object__find_map_fd_by_name(obj, "bench_map");
    if (bfd >= 0) {
        int ncpus = libbpf_num_possible_cpus();
        struct bench_sel_pair *vals = calloc(ncpus, sizeof(*vals));
        for (int cpu = 0; cpu < ncpus; cpu++)
            for (int g = 0; g < 16; g++) { vals[cpu].vi[g] = PACK4(16384ULL); vals[cpu].vt[g] = PACK4(16384ULL); }
        uint32_t k0 = 0;
        bpf_map_update_elem(bfd, &k0, vals, BPF_ANY);
        free(vals);
    }

    p = bpf_object__find_program_by_name(obj, name);
    int fd = bpf_program__fd(p);
    struct bpf_prog_info info = {}; uint32_t ilen = sizeof(info);
    bpf_obj_get_info_by_fd(fd, &info, &ilen);
    const char *vl = strstr(vlog, "processed ");
    char vbuf[140] = "N/A";
    if (vl) {
        const char *nl = strchr(vl, 10);
        size_t n = nl ? (size_t)(nl - vl) : strlen(vl);
        if (n >= sizeof(vbuf)) n = sizeof(vbuf) - 1;
        memcpy(vbuf, vl, n); vbuf[n] = 0;
    }
    printf("  [PASS] %-30s | xlated=%4u | jited=%5u B | %s\n", name, info.xlated_prog_len/8, info.jited_prog_len, vbuf);

    unsigned char pkt[64] = {0};
    LIBBPF_OPTS(bpf_test_run_opts, topts, .data_in = pkt, .data_size_in = 64, .repeat = 200000);
    bpf_prog_test_run_opts(fd, &topts);

    double res[7];
    topts.repeat = 500000;
    for (int i = 0; i < 7; i++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
        bpf_prog_test_run_opts(fd, &topts);
        clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
        res[i] = ((t1.tv_sec - t0.tv_sec)*1e9 + (t1.tv_nsec - t0.tv_nsec)) / 500000.0;
    }
    qsort(res, 7, sizeof(double), cmp_d);
    printf("         -> median=%.2f ns | min=%.2f ns | kern_dur=%u ns\n", res[3], res[0], topts.duration);
    bpf_object__close(obj);
    return res[3];
}

int main(void) {
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(2, &set);
    sched_setaffinity(0, sizeof(set), &set);
    const char *obj = "kssm_selective.bpf.o";
    printf("=== True Input-Dependent Selective Guard-SWAR 64D (Linux 5.15 Verifier + ns Bench) ===\n");
    double noop = bench_one(obj, "bench_noop", 0);
    double sub  = bench_one(obj, "bench_selective_subspace_64d", 0);
    double shr  = bench_one(obj, "bench_selective_shared_64d", 1);
    printf("\n=== NET LATENCY (2.10 GHz Xeon 6530) ===\n");
    printf("  True Selective Guard-SWAR 64D (Subspace, 34.8KB L1d) : %.2f ns\n", sub - noop);
    printf("  True Selective Guard-SWAR 64D (Shared,   34.8KB L1d) : %.2f ns\n", shr - noop);
    return 0;
}
