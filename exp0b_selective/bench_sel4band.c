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
#define PACK4(v) (((uint64_t)(v)) | ((uint64_t)(v)<<16) | ((uint64_t)(v)<<32) | ((uint64_t)(v)<<48))
struct kssm_sel4band_tables { uint64_t emb[KSSM_TOTAL_SLOTS][16]; };
static int cmp_d(const void *a, const void *b) { double x=*(const double*)a, y=*(const double*)b; return (x>y)-(x<y); }

static double run_one(const char *name, int fill_mode) {
    static char vlog[131072]; memset(vlog, 0, sizeof(vlog));
    LIBBPF_OPTS(bpf_object_open_opts, oopts, .kernel_log_buf=vlog, .kernel_log_size=sizeof(vlog), .kernel_log_level=4);
    struct bpf_object *obj = bpf_object__open_file("kssm_sel4band.bpf.o", &oopts);
    struct bpf_program *p;
    bpf_object__for_each_program(p, obj) bpf_program__set_autoload(p, strcmp(bpf_program__name(p), name)==0);
    if (bpf_object__load(obj) != 0) { printf("[FAIL] %s\n", name); return -1; }
    int tfd = bpf_object__find_map_fd_by_name(obj, "sel4b_tbl_map");
    if (tfd >= 0) {
        struct kssm_sel4band_tables *t = calloc(1, sizeof(*t));
        for (int s=0; s<KSSM_TOTAL_SLOTS; s++) for (int g=0; g<16; g++) t->emb[s][g] = PACK4(2048ULL);
        uint32_t k0=0; bpf_map_update_elem(tfd, &k0, t, BPF_ANY); free(t);
        int rfd = bpf_object__find_map_fd_by_name(obj, "regime_arr_map");
        if (rfd >= 0) {
            for (uint32_t i=0; i<1024; i++) {
                uint32_t v = (fill_mode == 1) ? 0x02020202u :
                    (((i+0)&3) | (((i+1)&3)<<8) | (((i+2)&3)<<16) | (((i+3)&3)<<24));
                bpf_map_update_elem(rfd, &i, &v, BPF_ANY);
            }
        }
    }
    int bfd = bpf_object__find_map_fd_by_name(obj, "bench4b_map");
    if (bfd >= 0) {
        int nc = libbpf_num_possible_cpus(); struct bench_sel_pair *v = calloc(nc, sizeof(*v));
        for (int c=0; c<nc; c++) for (int g=0; g<16; g++) { v[c].vi[g]=PACK4(16384ULL); v[c].vt[g]=PACK4(16384ULL); }
        uint32_t k0=0; bpf_map_update_elem(bfd, &k0, v, BPF_ANY); free(v);
    }
    p = bpf_object__find_program_by_name(obj, name); int fd = bpf_program__fd(p);
    struct bpf_prog_info info={}; uint32_t ilen=sizeof(info); bpf_obj_get_info_by_fd(fd, &info, &ilen);
    const char *vl = strstr(vlog, "processed "); char vbuf[140]="N/A";
    if (vl) { const char *nl=strchr(vl,10); size_t n=nl?(size_t)(nl-vl):strlen(vl); if(n>=139)n=139; memcpy(vbuf,vl,n); vbuf[n]=0; }
    printf("  [PASS] %-24s (mode=%d) | xlated=%4u | jited=%5u B | %s\n", name, fill_mode, info.xlated_prog_len/8, info.jited_prog_len, vbuf);
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
    bpf_object__close(obj);
    return res[3];
}
int main(void) {
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(2, &set); sched_setaffinity(0, sizeof(set), &set);
    double noop = run_one("bench_noop", 0);
    double mix  = run_one("bench_sel4band_dynamic", 0);
    double all2 = run_one("bench_sel4band_dynamic", 1);
    printf("  -> 4-Band Selective Guard-SWAR 64D (Dynamic Mix) : %.2f ns\n", mix - noop);
    printf("  -> 4-Band Selective Guard-SWAR 64D (All-4-Active): %.2f ns\n", all2 - noop);
    return 0;
}
