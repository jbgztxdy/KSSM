#include "../vmlinux.h"
#include <bpf/bpf_helpers.h>
#include "kssm_selective.h"

char LICENSE[] SEC("license") = "GPL";

#define PACK4(v) (((__u64)(v)) | ((__u64)(v) << 16) | ((__u64)(v) << 32) | ((__u64)(v) << 48))
#define MASK_SH(sh) PACK4(0xFFFFULL >> (sh))
#define CONST_POS2(d, c) PACK4(4096ULL + (16384ULL >> (c)) - (16384ULL >> (d)))
#define CONST_NEG2(d, c) PACK4(4096ULL - (16384ULL >> (c)) - (16384ULL >> (d)))

/* Each (s1, s4) key maps to 4 independent band regimes packed in 4 bytes:
 *   byte b in {0=HOLD(alpha=1,0ns), 1=SLOW(d=3), 2=FAST(d=1), 3=RESET(alpha=0,gamma=0.5)}
 * Plus pre-baked 192x16 u64 table (24 KB). Total = 28 KB (fits in 48KB L1d!) */
struct kssm_sel4band_tables {
    __u64 emb[KSSM_TOTAL_SLOTS][16]; /* 24 KB */
};
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1024);
    __type(key, __u32);
    __type(value, __u32);
} regime_arr_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct kssm_sel4band_tables);
} sel4b_tbl_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct bench_sel_pair);
} bench4b_map SEC(".maps");

#define SUB_STEP(vi, vt, g, d, c, ea, eb) do { \
    __u64 _oi = (vi)[g]; __u64 _ot = (vt)[g]; __u64 _b = (ea)[g] + (eb)[g]; \
    (vi)[g] = _oi - ((_oi >> (d)) & MASK_SH(d)) + ((_ot >> (c)) & MASK_SH(c)) + _b - CONST_POS2(d, c); \
    (vt)[g] = _ot - ((_ot >> (d)) & MASK_SH(d)) - ((_oi >> (c)) & MASK_SH(c)) + _b - CONST_NEG2(d, c); \
} while (0)

#define SUB_RESET(vi, vt, g, ea, eb) do { \
    __u64 _ot = (vt)[g]; __u64 _b = (ea)[g] + (eb)[g]; \
    (vi)[g] = ((_ot >> 1) & MASK_SH(1)) + _b + PACK4(4096ULL); \
    (vt)[g] = _ot - ((_ot >> 2) & MASK_SH(2)) + _b; \
} while (0)

#define RUN_BAND(vi, vt, base, reg, ea, eb) do { \
    if ((reg) == 1) { \
        SUB_STEP(vi, vt, (base)+0, 3, 4, ea, eb); SUB_STEP(vi, vt, (base)+1, 3, 4, ea, eb); \
        SUB_STEP(vi, vt, (base)+2, 3, 4, ea, eb); SUB_STEP(vi, vt, (base)+3, 3, 4, ea, eb); \
    } else if ((reg) == 2) { \
        SUB_STEP(vi, vt, (base)+0, 1, 2, ea, eb); SUB_STEP(vi, vt, (base)+1, 1, 2, ea, eb); \
        SUB_STEP(vi, vt, (base)+2, 1, 2, ea, eb); SUB_STEP(vi, vt, (base)+3, 1, 2, ea, eb); \
    } else if ((reg) == 3) { \
        SUB_RESET(vi, vt, (base)+0, ea, eb); SUB_RESET(vi, vt, (base)+1, ea, eb); \
        SUB_RESET(vi, vt, (base)+2, ea, eb); SUB_RESET(vi, vt, (base)+3, ea, eb); \
    } \
} while (0)

SEC("xdp") int bench_noop(struct xdp_md *ctx) { return XDP_PASS; }

SEC("xdp") int bench_sel4band_dynamic(struct xdp_md *ctx)
{
    __u32 k = 0;
    struct bench_sel_pair *bp = bpf_map_lookup_elem(&bench4b_map, &k);
    struct kssm_sel4band_tables *tbl = bpf_map_lookup_elem(&sel4b_tbl_map, &k);
    if (!bp || !tbl) return XDP_PASS;

    __u64 s1 = bp->s1 & 31ULL;
    __u64 s4 = bp->s4 & 31ULL;
    __u32 gkey = ((__u32)(bp->s1 & 31) << 5) | (__u32)(bp->s4 & 31); if (gkey >= 1024) return XDP_PASS;

    const __u64 *e1 = tbl->emb[KSSM_S1_OFF + s1];
    const __u64 *e2 = tbl->emb[KSSM_S2_OFF + (bp->s2 & 63ULL)];
    const __u64 *e3 = tbl->emb[KSSM_S3_OFF + (bp->s3 & 31ULL)];
    const __u64 *e4 = tbl->emb[KSSM_S4_OFF + s4];
    const __u64 *e5 = tbl->emb[KSSM_S5_OFF + (bp->s5 & 31ULL)];

    __u32 *rp = bpf_map_lookup_elem(&regime_arr_map, &gkey);
    if (!rp) return XDP_PASS;
    __u32 rv = *rp;
    __u8 r0 = rv & 3;
    __u8 r1 = (rv >> 8) & 3;
    __u8 r2 = (rv >> 16) & 3;
    __u8 r3 = (rv >> 24) & 3;

    RUN_BAND(bp->vi, bp->vt,  0, r0, e1, e2);
    RUN_BAND(bp->vi, bp->vt,  4, r1, e1, e3);
    RUN_BAND(bp->vi, bp->vt,  8, r2, e1, e4);
    RUN_BAND(bp->vi, bp->vt, 12, r3, e1, e5);

    bp->s1 = (bp->s1 + 1) & 31;
    bp->s4 = (bp->s4 + 3) & 31;
    return XDP_PASS;
}
