#include "../vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "kssm_selective.h"

char LICENSE[] SEC("license") = "GPL";

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct kssm_selective_tables);
} sel_tbl_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct bench_sel_pair);
} bench_map SEC(".maps");

/* Branchless input-dependent Selective Guard-SWAR step (4 lanes per u64) */
#define SEL_STEP_SUB(vi, vt, g, gdesc, ea, eb) do { \
    __u64 _oi = (vi)[g]; \
    __u64 _ot = (vt)[g]; \
    __u64 _b  = (ea)[g] + (eb)[g]; \
    (vi)[g] = (_oi & (gdesc)->keep_i) - ((_oi >> (gdesc)->d_i) & (gdesc)->mask_di) \
              + ((_ot >> (gdesc)->c_i) & (gdesc)->mask_ci) + _b - (gdesc)->const_i; \
    (vt)[g] = (_ot & (gdesc)->keep_t) - ((_ot >> (gdesc)->d_t) & (gdesc)->mask_dt) \
              - ((_oi >> (gdesc)->c_t) & (gdesc)->mask_ct) + _b - (gdesc)->const_t; \
} while (0)

#define SEL_STEP_SHARED(vi, vt, g, gdesc, e1, e2, e3, e4, e5) do { \
    __u64 _oi = (vi)[g]; \
    __u64 _ot = (vt)[g]; \
    __u64 _b  = (e1)[g] + (e2)[g] + (e3)[g] + (e4)[g] + (e5)[g]; \
    (vi)[g] = (_oi & (gdesc)->keep_i) - ((_oi >> (gdesc)->d_i) & (gdesc)->mask_di) \
              + ((_ot >> (gdesc)->c_i) & (gdesc)->mask_ci) + _b - (gdesc)->const_i; \
    (vt)[g] = (_ot & (gdesc)->keep_t) - ((_ot >> (gdesc)->d_t) & (gdesc)->mask_dt) \
              - ((_oi >> (gdesc)->c_t) & (gdesc)->mask_ct) + _b - (gdesc)->const_t; \
} while (0)

SEC("xdp")
int bench_noop(struct xdp_md *ctx) { return XDP_PASS; }

SEC("xdp")
int bench_selective_subspace_64d(struct xdp_md *ctx)
{
    __u32 k = 0;
    struct bench_sel_pair *bp = bpf_map_lookup_elem(&bench_map, &k);
    struct kssm_selective_tables *tbl = bpf_map_lookup_elem(&sel_tbl_map, &k);
    if (!bp || !tbl) return XDP_PASS;

    __u64 s1 = bp->s1 & 31ULL;
    __u64 i1 = KSSM_S1_OFF + s1;
    __u64 i2 = KSSM_S2_OFF + (bp->s2 & 63ULL);
    __u64 i3 = KSSM_S3_OFF + (bp->s3 & 31ULL);
    __u64 i4 = KSSM_S4_OFF + (bp->s4 & 31ULL);
    __u64 i5 = KSSM_S5_OFF + (bp->s5 & 31ULL);

    const __u64 *e1 = tbl->emb[i1];
    const __u64 *e2 = tbl->emb[i2];
    const __u64 *e3 = tbl->emb[i3];
    const __u64 *e4 = tbl->emb[i4];
    const __u64 *e5 = tbl->emb[i5];

    const struct sel_band_gate *g0 = &tbl->gate[s1][0];
    const struct sel_band_gate *g1 = &tbl->gate[s1][1];
    const struct sel_band_gate *g2 = &tbl->gate[s1][2];
    const struct sel_band_gate *g3 = &tbl->gate[s1][3];

    __u64 *vi = bp->vi;
    __u64 *vt = bp->vt;

    SEL_STEP_SUB(vi, vt,  0, g0, e1, e2);
    SEL_STEP_SUB(vi, vt,  1, g0, e1, e2);
    SEL_STEP_SUB(vi, vt,  2, g0, e1, e2);
    SEL_STEP_SUB(vi, vt,  3, g0, e1, e2);

    SEL_STEP_SUB(vi, vt,  4, g1, e1, e3);
    SEL_STEP_SUB(vi, vt,  5, g1, e1, e3);
    SEL_STEP_SUB(vi, vt,  6, g1, e1, e3);
    SEL_STEP_SUB(vi, vt,  7, g1, e1, e3);

    SEL_STEP_SUB(vi, vt,  8, g2, e1, e4);
    SEL_STEP_SUB(vi, vt,  9, g2, e1, e4);
    SEL_STEP_SUB(vi, vt, 10, g2, e1, e4);
    SEL_STEP_SUB(vi, vt, 11, g2, e1, e4);

    SEL_STEP_SUB(vi, vt, 12, g3, e1, e5);
    SEL_STEP_SUB(vi, vt, 13, g3, e1, e5);
    SEL_STEP_SUB(vi, vt, 14, g3, e1, e5);
    SEL_STEP_SUB(vi, vt, 15, g3, e1, e5);

    bp->s1 = (bp->s1 + 1) & 31;
    return XDP_PASS;
}

SEC("xdp")
int bench_selective_shared_64d(struct xdp_md *ctx)
{
    __u32 k = 0;
    struct bench_sel_pair *bp = bpf_map_lookup_elem(&bench_map, &k);
    struct kssm_selective_tables *tbl = bpf_map_lookup_elem(&sel_tbl_map, &k);
    if (!bp || !tbl) return XDP_PASS;

    __u64 s1 = bp->s1 & 31ULL;
    __u64 i1 = KSSM_S1_OFF + s1;
    __u64 i2 = KSSM_S2_OFF + (bp->s2 & 63ULL);
    __u64 i3 = KSSM_S3_OFF + (bp->s3 & 31ULL);
    __u64 i4 = KSSM_S4_OFF + (bp->s4 & 31ULL);
    __u64 i5 = KSSM_S5_OFF + (bp->s5 & 31ULL);

    const __u64 *e1 = tbl->emb[i1];
    const __u64 *e2 = tbl->emb[i2];
    const __u64 *e3 = tbl->emb[i3];
    const __u64 *e4 = tbl->emb[i4];
    const __u64 *e5 = tbl->emb[i5];

    const struct sel_band_gate *g0 = &tbl->gate[s1][0];
    const struct sel_band_gate *g1 = &tbl->gate[s1][1];
    const struct sel_band_gate *g2 = &tbl->gate[s1][2];
    const struct sel_band_gate *g3 = &tbl->gate[s1][3];

    __u64 *vi = bp->vi;
    __u64 *vt = bp->vt;

    SEL_STEP_SHARED(vi, vt,  0, g0, e1, e2, e3, e4, e5);
    SEL_STEP_SHARED(vi, vt,  1, g0, e1, e2, e3, e4, e5);
    SEL_STEP_SHARED(vi, vt,  2, g0, e1, e2, e3, e4, e5);
    SEL_STEP_SHARED(vi, vt,  3, g0, e1, e2, e3, e4, e5);

    SEL_STEP_SHARED(vi, vt,  4, g1, e1, e2, e3, e4, e5);
    SEL_STEP_SHARED(vi, vt,  5, g1, e1, e2, e3, e4, e5);
    SEL_STEP_SHARED(vi, vt,  6, g1, e1, e2, e3, e4, e5);
    SEL_STEP_SHARED(vi, vt,  7, g1, e1, e2, e3, e4, e5);

    SEL_STEP_SHARED(vi, vt,  8, g2, e1, e2, e3, e4, e5);
    SEL_STEP_SHARED(vi, vt,  9, g2, e1, e2, e3, e4, e5);
    SEL_STEP_SHARED(vi, vt, 10, g2, e1, e2, e3, e4, e5);
    SEL_STEP_SHARED(vi, vt, 11, g2, e1, e2, e3, e4, e5);

    SEL_STEP_SHARED(vi, vt, 12, g3, e1, e2, e3, e4, e5);
    SEL_STEP_SHARED(vi, vt, 13, g3, e1, e2, e3, e4, e5);
    SEL_STEP_SHARED(vi, vt, 14, g3, e1, e2, e3, e4, e5);
    SEL_STEP_SHARED(vi, vt, 15, g3, e1, e2, e3, e4, e5);

    bp->s1 = (bp->s1 + 1) & 31;
    return XDP_PASS;
}
