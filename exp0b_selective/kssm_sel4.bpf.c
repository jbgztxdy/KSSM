#include "../vmlinux.h"
#include <bpf/bpf_helpers.h>
#include "kssm_selective.h"

char LICENSE[] SEC("license") = "GPL";

#define PACK4(v) (((__u64)(v)) | ((__u64)(v) << 16) | ((__u64)(v) << 32) | ((__u64)(v) << 48))
#define MASK_SH(sh) PACK4(0xFFFFULL >> (sh))
#define CONST_POS2(d, c) PACK4(4096ULL  + (16384ULL >> (c)) - (16384ULL >> (d)))
#define CONST_NEG2(d, c) PACK4(4096ULL  - (16384ULL >> (c)) - (16384ULL >> (d)))
#define CONST_POS5(d, c) PACK4(10240ULL + (16384ULL >> (c)) - (16384ULL >> (d)))
#define CONST_NEG5(d, c) PACK4(10240ULL - (16384ULL >> (c)) - (16384ULL >> (d)))

/* 4-Regime Selective Gate Table indexed by (s1 in 0..31, is_noise_or_str in 0..3) = 128 bytes!
 * mode = 0: HOLD (alpha=1, gamma=0, skip noise -> ~4 ns!)
 * mode = 1: FAST_DECAY (multi-scale d=1,1,2,3 -> responsive adaptation)
 * mode = 2: SLOW_DECAY (long-horizon d=2,3,4,5 -> preserve long context)
 * mode = 3: PHASE_RESET (alpha_i=0, gamma=0.5 -> instant phase cut + strong task->inode inject!)
 */
struct kssm_sel4_tables {
    __u64 emb[KSSM_TOTAL_SLOTS][16]; /* 24 KB */
    __u8  regime[128];               /* 128 B: input-dependent selective gate regime */
};

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct kssm_sel4_tables);
} sel4_tbl_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct bench_sel_pair);
} bench4_map SEC(".maps");

#define SUB_STEP(vi, vt, g, d, c, ea, eb) do { \
    __u64 _oi = (vi)[g]; __u64 _ot = (vt)[g]; \
    __u64 _b  = (ea)[g] + (eb)[g]; \
    (vi)[g] = _oi - ((_oi >> (d)) & MASK_SH(d)) + ((_ot >> (c)) & MASK_SH(c)) + _b - CONST_POS2(d, c); \
    (vt)[g] = _ot - ((_ot >> (d)) & MASK_SH(d)) - ((_oi >> (c)) & MASK_SH(c)) + _b - CONST_NEG2(d, c); \
} while (0)

#define SUB_RESET_STEP(vi, vt, g, ea, eb) do { \
    __u64 _ot = (vt)[g]; \
    __u64 _b  = (ea)[g] + (eb)[g]; \
    /* alpha_i = 0 (wipe old inode), gamma = 0.5 (strong task->inode inject!):
     * (ot>>1)&0x7fff = 0.5*ht + 8192; + b (excess 4096) + 4096 -> 0.5*ht + b_real + 16384! */ \
    (vi)[g] = ((_ot >> 1) & MASK_SH(1)) + _b + PACK4(4096ULL); \
    /* task keeps 0.75*ht + b */ \
    (vt)[g] = _ot - ((_ot >> 2) & MASK_SH(2)) + _b; \
} while (0)

static __always_inline void run_sel4_subspace(struct bench_sel_pair *bp, const struct kssm_sel4_tables *tbl, __u8 force_mode)
{
    __u64 s1 = bp->s1 & 31ULL;
    __u8 mode = (force_mode < 4) ? force_mode : tbl->regime[s1];

    /* Regime 0: Noise-Immune HOLD (alpha=1, gamma=0, b=0) -> Instant Skip! */
    if (mode == 0)
        return;

    const __u64 *e1 = tbl->emb[KSSM_S1_OFF + s1];
    const __u64 *e2 = tbl->emb[KSSM_S2_OFF + (bp->s2 & 63ULL)];
    const __u64 *e3 = tbl->emb[KSSM_S3_OFF + (bp->s3 & 31ULL)];
    const __u64 *e4 = tbl->emb[KSSM_S4_OFF + (bp->s4 & 31ULL)];
    const __u64 *e5 = tbl->emb[KSSM_S5_OFF + (bp->s5 & 31ULL)];
    __u64 *vi = bp->vi;
    __u64 *vt = bp->vt;

    if (mode == 3) {
        /* Regime 3: Phase-Boundary RESET + Strong Task->Inode Injection */
        SUB_RESET_STEP(vi, vt,  0, e1, e2); SUB_RESET_STEP(vi, vt,  1, e1, e2);
        SUB_RESET_STEP(vi, vt,  2, e1, e2); SUB_RESET_STEP(vi, vt,  3, e1, e2);
        SUB_RESET_STEP(vi, vt,  4, e1, e3); SUB_RESET_STEP(vi, vt,  5, e1, e3);
        SUB_RESET_STEP(vi, vt,  6, e1, e3); SUB_RESET_STEP(vi, vt,  7, e1, e3);
        SUB_RESET_STEP(vi, vt,  8, e1, e4); SUB_RESET_STEP(vi, vt,  9, e1, e4);
        SUB_RESET_STEP(vi, vt, 10, e1, e4); SUB_RESET_STEP(vi, vt, 11, e1, e4);
        SUB_RESET_STEP(vi, vt, 12, e1, e5); SUB_RESET_STEP(vi, vt, 13, e1, e5);
        SUB_RESET_STEP(vi, vt, 14, e1, e5); SUB_RESET_STEP(vi, vt, 15, e1, e5);
        return;
    }

    if (mode == 2) {
        /* Regime 2: Long-Horizon Slow Decay (d = 2,3,4,5) */
        SUB_STEP(vi, vt,  0, 2, 3, e1, e2); SUB_STEP(vi, vt,  1, 3, 4, e1, e2);
        SUB_STEP(vi, vt,  2, 4, 5, e1, e2); SUB_STEP(vi, vt,  3, 5, 6, e1, e2);
        SUB_STEP(vi, vt,  4, 2, 3, e1, e3); SUB_STEP(vi, vt,  5, 3, 4, e1, e3);
        SUB_STEP(vi, vt,  6, 4, 5, e1, e3); SUB_STEP(vi, vt,  7, 5, 6, e1, e3);
        SUB_STEP(vi, vt,  8, 2, 3, e1, e4); SUB_STEP(vi, vt,  9, 3, 4, e1, e4);
        SUB_STEP(vi, vt, 10, 4, 5, e1, e4); SUB_STEP(vi, vt, 11, 5, 6, e1, e4);
        SUB_STEP(vi, vt, 12, 2, 3, e1, e5); SUB_STEP(vi, vt, 13, 3, 4, e1, e5);
        SUB_STEP(vi, vt, 14, 4, 5, e1, e5); SUB_STEP(vi, vt, 15, 5, 6, e1, e5);
        return;
    }

    /* Regime 1: Fast-Adapting Multi-Scale Decay (d = 1,1,2,3) */
    SUB_STEP(vi, vt,  0, 1, 2, e1, e2); SUB_STEP(vi, vt,  1, 1, 3, e1, e2);
    SUB_STEP(vi, vt,  2, 2, 3, e1, e2); SUB_STEP(vi, vt,  3, 3, 4, e1, e2);
    SUB_STEP(vi, vt,  4, 1, 2, e1, e3); SUB_STEP(vi, vt,  5, 2, 3, e1, e3);
    SUB_STEP(vi, vt,  6, 3, 4, e1, e3); SUB_STEP(vi, vt,  7, 4, 5, e1, e3);
    SUB_STEP(vi, vt,  8, 1, 2, e1, e4); SUB_STEP(vi, vt,  9, 2, 3, e1, e4);
    SUB_STEP(vi, vt, 10, 3, 4, e1, e4); SUB_STEP(vi, vt, 11, 4, 5, e1, e4);
    SUB_STEP(vi, vt, 12, 1, 2, e1, e5); SUB_STEP(vi, vt, 13, 2, 3, e1, e5);
    SUB_STEP(vi, vt, 14, 3, 4, e1, e5); SUB_STEP(vi, vt, 15, 4, 5, e1, e5);
}

SEC("xdp") int bench_noop(struct xdp_md *ctx) { return XDP_PASS; }

SEC("xdp") int bench_sel4_dynamic(struct xdp_md *ctx)
{
    __u32 k = 0;
    struct bench_sel_pair *bp = bpf_map_lookup_elem(&bench4_map, &k);
    struct kssm_sel4_tables *tbl = bpf_map_lookup_elem(&sel4_tbl_map, &k);
    if (!bp || !tbl) return XDP_PASS;
    run_sel4_subspace(bp, tbl, 255); /* dynamic lookup from tbl->regime[s1] */
    bp->s1 = (bp->s1 + 1) & 31;
    return XDP_PASS;
}

SEC("xdp") int bench_sel4_worst_decay(struct xdp_md *ctx)
{
    __u32 k = 0;
    struct bench_sel_pair *bp = bpf_map_lookup_elem(&bench4_map, &k);
    struct kssm_sel4_tables *tbl = bpf_map_lookup_elem(&sel4_tbl_map, &k);
    if (!bp || !tbl) return XDP_PASS;
    run_sel4_subspace(bp, tbl, 1); /* worst-case full DECAY path */
    bp->s1 = (bp->s1 + 1) & 31;
    return XDP_PASS;
}

SEC("xdp") int bench_sel4_reset(struct xdp_md *ctx)
{
    __u32 k = 0;
    struct bench_sel_pair *bp = bpf_map_lookup_elem(&bench4_map, &k);
    struct kssm_sel4_tables *tbl = bpf_map_lookup_elem(&sel4_tbl_map, &k);
    if (!bp || !tbl) return XDP_PASS;
    run_sel4_subspace(bp, tbl, 3); /* PHASE_RESET path */
    bp->s1 = (bp->s1 + 1) & 31;
    return XDP_PASS;
}

SEC("xdp") int bench_sel4_hold(struct xdp_md *ctx)
{
    __u32 k = 0;
    struct bench_sel_pair *bp = bpf_map_lookup_elem(&bench4_map, &k);
    struct kssm_sel4_tables *tbl = bpf_map_lookup_elem(&sel4_tbl_map, &k);
    if (!bp || !tbl) return XDP_PASS;
    run_sel4_subspace(bp, tbl, 0); /* Noise-Immune HOLD path */
    bp->s1 = (bp->s1 + 1) & 31;
    return XDP_PASS;
}
