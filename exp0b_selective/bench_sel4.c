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
struct kssm_sel4_tables { uint64_t emb[KSSM_TOTAL_SLOTS][16]; uint8_t regime[128]; };
static int cmp_d(const void *a, const void *b) { double x=*(const double*)a, y=*(const double*)b; return (x>y)-(x<y); }

static double run_one(const char *obj_path, const char *name) {
    static char vlog[131072]; memset(vlog, 0, sizeof(vlog));
    LIBBPF_OPTS(bpf_object_open_opts, oopts, .kernel_log_buf=vlog, .kernel_log_size=sizeof(vlog), .kernel_log_level=4);
    struct bpf_object *obj = bpf_object__open_file(obj_path, &oopts);
    struct bpf_program *p;
    bpf_object__for_each_program(p, obj) bpf_program__set_autoload(p, strcmp(bpf_program__name(p), name)==0);
    if (bpf_object__load(obj) != 0) { printf("[FAIL] %s\n", name); return -1; }
    int tfd = bpf_object__find_map_fd_by_name(obj, "sel4_tbl_map");
    if (tfd >= 0) {
        struct kssm_sel4_tables *t = calloc(1, sizeof(*t));
        for (int s=0; s<KSSM_TOTAL_SLOTS; s++)
            for (int g=0; g<16; g++) t->emb[s][g] = PACK4(2048ULL);
        for (int i=0; i<128; i++) t->regime[i] = (uint8_t)(i & 3); /* 25% each regime */
        uint32_t k0=0; bpf_map_update_elem(tfd, &k0, t, BPF_ANY); free(t);
    }
    int bfd = bpf_object__find_map_fd_by_name(obj, "bench4_map");
    if (bfd >= 0) {
        int nc = libbpf_num_possible_cpus();
        struct bench_sel_pair *v = calloc(nc, sizeof(*v));
        for (int c=0; c<nc; c++) for (int g=0; g<16; g++) { v[c].vi[g]=PACK4(16384ULL); v[c].vt[g]=PACK4(16384ULL); }
        uint32_t k0=0; bpf_map_update_elem(bfd, &k0, v, BPF_ANY); free(v);
    }
    p = bpf_object__find_program_by_name(obj, name);
    int fd = bpf_program__fd(p);
    struct bpf_prog_info info = {}; uint32_t ilen = sizeof(info);
    bpf_obj_get_info_by_fd(fd, &info, &ilen);
    const char *vl = strstr(vlog, "processed "); char vbuf[140]="N/A";
    if (vl) { const char *nl=strchr(vl,10); size_t n=nl?(size_t)(nl-vl):strlen(vl); if(n>=139)n=139; memcpy(vbuf,vl,n); vbuf[n]=0; }
    printf("  [PASS] %-24s | xlated=%4u | jited=%5u B | %s\n", name, info.xlated_prog_len/8, info.jited_prog_len, vbuf);

    unsigned char pkt[64]={0};
    LIBBPF_OPTS(bpf_test_run_opts, topts, .data_in=pkt, .data_size_in=64, .repeat=200000);
    bpf_prog_test_run_opts(fd, &topts);
    double res[7]; topts.repeat=500000;
    for (int i=0; i<7; i++) {
        struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
        bpf_prog_test_run_opts(fd, &topts);
        clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
        res[i] = ((t1.tv_sec-t0.tv_sec)*1e9 + (t1.tv_nsec-t0.tv_nsec))/500000.0;
    }
    qsort(res, 7, sizeof(double), cmp_d);
    printf("         -> median=%.2f ns | min=%.2f ns\n", res[3], res[0]);
    bpf_object__close(obj);
    return res[3];
}

int main(void) {
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(2, &set); sched_setaffinity(0, sizeof(set), &set);
    double noop = run_one("kssm_sel4.bpf.o", "bench_noop");
    double dyn  = run_one("kssm_sel4.bpf.o", "bench_sel4_dynamic");
    double dec  = run_one("kssm_sel4.bpf.o", "bench_sel4_worst_decay");
    double rst  = run_one("kssm_sel4.bpf.o", "bench_sel4_reset");
    double hld  = run_one("kssm_sel4.bpf.o", "bench_sel4_hold");
    printf("\n=== 4-Regime True Selective Guard-SWAR 64D Net Latency (2.10 GHz Xeon 6530) ===\n");
    printf("  Dynamic Mixed (25%% HOLD / 50%% DECAY / 25%% RESET) : %.2f ns\n", dyn - noop);
    printf("  Worst-Case DECAY Path (64D full update)          : %.2f ns\n", dec - noop);
    printf("  PHASE-RESET + Strong Task->Inode Injection       : %.2f ns\n", rst - noop);
    printf("  Noise-Immune HOLD Path (alpha=1, skip noise)     : %.2f ns\n", hld - noop);
    return 0;
}
