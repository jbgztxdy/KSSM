#include "../vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "kssm_common.h"

char LICENSE[] SEC("license") = "GPL";

volatile const __u32 cfg_use_stackid = 1;
volatile const __u32 cfg_use_gating = 1;
volatile const __u32 cfg_use_shift_ssm = 0;

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct kssm_tables);
} tbl_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct kssm_readout);
} readout_map SEC(".maps");

/* Zero template to avoid putting 184B struct on BPF stack */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct kssm_inode_state);
} zero_inode_map SEC(".maps");

struct bench_pair {
    struct kssm_hidden h_i;
    struct kssm_hidden h_t;
    __u32 s1, s2, s3, s4, s5;
    __s32 sink;
};

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct bench_pair);
} bench_state_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_TASK_STORAGE);
    __uint(map_flags, BPF_F_NO_PREALLOC);
    __type(key, int);
    __type(value, struct kssm_task_state);
} task_storage SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_INODE_STORAGE);
    __uint(map_flags, BPF_F_NO_PREALLOC);
    __type(key, int);
    __type(value, struct kssm_inode_state);
} inode_storage SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(map_flags, BPF_F_NO_COMMON_LRU);
    __uint(max_entries, 131072);
    __type(key, __u64);
    __type(value, struct kssm_inode_state);
} inode_lru_map SEC(".maps");

struct dir_churn {
    __u64 last_ts_ns;
    __u32 ema_events;
    __u32 pad;
};

struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(map_flags, BPF_F_NO_COMMON_LRU);
    __uint(max_entries, 16384);
    __type(key, __u64);
    __type(value, struct dir_churn);
} dir_churn_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_STACK_TRACE);
    __uint(key_size, sizeof(__u32));
    __uint(value_size, 16 * sizeof(__u64));
    __uint(max_entries, 8192);
} stack_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct kssm_cpu_stats);
} stats_map SEC(".maps");

static __always_inline __s16 sat16_branchless(__s32 x)
{
    __s32 s = x >> 31;
    __s32 clipped = (x + 32768) >> 16;
    __u32 ov = (__u32)(clipped | -clipped) >> 31;
    __s32 mask = -(__s32)ov;
    __s32 sat_val = 32767 ^ s;
    return (__s16)((x & ~mask) | (sat_val & mask));
}

static __always_inline __u32 fast_log2_u64(__u64 v)
{
    __u32 r = 0;
    if (v >= (1ULL << 32)) { v >>= 32; r += 32; }
    if (v >= (1ULL << 16)) { v >>= 16; r += 16; }
    if (v >= (1ULL << 8))  { v >>= 8;  r += 8;  }
    if (v >= (1ULL << 4))  { v >>= 4;  r += 4;  }
    if (v >= (1ULL << 2))  { v >>= 2;  r += 2;  }
    if (v >= (1ULL << 1))  { r += 1; }
    return r;
}

/* --------------------------------------------------------------------------
 * Core SSM Kernel 1: Exact Q7.8 Fixed-Point Diagonal Coupled SSM (D=64)
 * Keep register pressure minimal -> zero stack overflow, tiny L1i footprint!
 * -------------------------------------------------------------------------- */
static __always_inline void kssm_step_exact_64d(
    struct kssm_hidden *hi,
    struct kssm_hidden *ht,
    const struct kssm_tables *tbl,
    const struct kssm_readout *ro,
    __u32 s1, __u32 s2, __u32 s3, __u32 s4, __u32 s5,
    __s16 *out_scan, __s16 *out_life,
    int do_readout)
{
    s1 &= 31;
    __u32 i1 = KSSM_S1_OFF + s1;
    __u32 i2 = KSSM_S2_OFF + (s2 & 63);
    __u32 i3 = KSSM_S3_OFF + (s3 & 31);
    __u32 i4 = KSSM_S4_OFF + (s4 & 31);
    __u32 i5 = KSSM_S5_OFF + (s5 & 31);

    __s32 acc_scan = (do_readout && ro) ? ro->b_scan : 0;
    __s32 acc_life = (do_readout && ro) ? ro->b_life : 0;

    #pragma clang loop unroll(disable)
    for (int c = 0; c < KSSM_DIM; c++) {
        __s32 oi = hi->h[c];
        __s32 ot = ht->h[c];

        __s32 bi = (__s32)tbl->emb_inode[i1][c]
                 + (__s32)tbl->emb_inode[i2][c]
                 + (__s32)tbl->emb_inode[i3][c]
                 + (__s32)tbl->emb_inode[i4][c]
                 + (__s32)tbl->emb_inode[i5][c];

        __s32 bt = (__s32)tbl->emb_task[i1][c]
                 + (__s32)tbl->emb_task[i2][c]
                 + (__s32)tbl->emb_task[i3][c]
                 + (__s32)tbl->emb_task[i4][c]
                 + (__s32)tbl->emb_task[i5][c];

        __s32 ni = ((oi * (__s32)tbl->alpha_inode[s1][c]
                   + ot * (__s32)tbl->gamma_t2i[s1][c]) >> 8) + bi;
        __s32 nt = ((ot * (__s32)tbl->alpha_task[s1][c]
                   + oi * (__s32)tbl->eta_i2t[s1][c]) >> 8) + bt;

        __s16 si = sat16_branchless(ni);
        __s16 st = sat16_branchless(nt);

        hi->h[c] = si;
        ht->h[c] = st;

        if (do_readout && ro) {
            acc_scan += (__s32)si * (__s32)ro->w_scan_i[c]
                      + (__s32)st * (__s32)ro->w_scan_t[c];
            acc_life += (__s32)si * (__s32)ro->w_life_i[c]
                      + (__s32)st * (__s32)ro->w_life_t[c];
        }
    }

    if (do_readout && out_scan && out_life) {
        *out_scan = sat16_branchless(acc_scan >> 8);
        *out_life = sat16_branchless(acc_life >> 8);
    }
}

/* --------------------------------------------------------------------------
 * Core SSM Kernel 2: Shift-Gate Fixed-Point Coupled SSM (D=64, zero multiplies!)
 * -------------------------------------------------------------------------- */
static __always_inline void kssm_step_shift_64d(
    struct kssm_hidden *hi,
    struct kssm_hidden *ht,
    const struct kssm_tables *tbl,
    const struct kssm_readout *ro,
    __u32 s1, __u32 s2, __u32 s3, __u32 s4, __u32 s5,
    __s16 *out_scan, __s16 *out_life,
    int do_readout)
{
    s1 &= 31;
    __u32 i1 = KSSM_S1_OFF + s1;
    __u32 i2 = KSSM_S2_OFF + (s2 & 63);
    __u32 i3 = KSSM_S3_OFF + (s3 & 31);
    __u32 i4 = KSSM_S4_OFF + (s4 & 31);
    __u32 i5 = KSSM_S5_OFF + (s5 & 31);

    __s32 acc_scan = (do_readout && ro) ? ro->b_scan : 0;
    __s32 acc_life = (do_readout && ro) ? ro->b_life : 0;

    #pragma clang loop unroll(disable)
    for (int c = 0; c < KSSM_DIM; c++) {
        __s32 oi = hi->h[c];
        __s32 ot = ht->h[c];

        __u32 di = tbl->sh_decay_i[s1][c] & 7;
        __u32 ci = tbl->sh_coup_i[s1][c] & 7;
        __u32 dt = tbl->sh_decay_t[s1][c] & 7;
        __u32 ct = tbl->sh_coup_t[s1][c] & 7;

        __s32 bi = (__s32)tbl->emb_inode[i1][c]
                 + (__s32)tbl->emb_inode[i2][c]
                 + (__s32)tbl->emb_inode[i3][c]
                 + (__s32)tbl->emb_inode[i4][c]
                 + (__s32)tbl->emb_inode[i5][c];

        __s32 bt = (__s32)tbl->emb_task[i1][c]
                 + (__s32)tbl->emb_task[i2][c]
                 + (__s32)tbl->emb_task[i3][c]
                 + (__s32)tbl->emb_task[i4][c]
                 + (__s32)tbl->emb_task[i5][c];

        __s32 ni = oi - (oi >> di) + (ot >> ci) + bi;
        __s32 nt = ot - (ot >> dt) + (oi >> ct) + bt;

        __s16 si = sat16_branchless(ni);
        __s16 st = sat16_branchless(nt);

        hi->h[c] = si;
        ht->h[c] = st;

        if (do_readout && ro) {
            acc_scan += (__s32)si * (__s32)ro->w_scan_i[c]
                      + (__s32)st * (__s32)ro->w_scan_t[c];
            acc_life += (__s32)si * (__s32)ro->w_life_i[c]
                      + (__s32)st * (__s32)ro->w_life_t[c];
        }
    }

    if (do_readout && out_scan && out_life) {
        *out_scan = sat16_branchless(acc_scan >> 8);
        *out_life = sat16_branchless(acc_life >> 8);
    }
}

/* --------------------------------------------------------------------------
 * Core SSM Kernel 3: SWAR 4-Lane 64-bit Parallel Packed SSM (D=64 = 16x u64)
 * Pure 64-bit integer SWAR: 4x Q7.8 channels updated simultaneously per u64!
 * -------------------------------------------------------------------------- */
static __always_inline __u64 swar_add4_s16(__u64 a, __u64 b)
{
    const __u64 lo_mask = 0x7fff7fff7fff7fffULL;
    __u64 lo = (a & lo_mask) + (b & lo_mask);
    return lo ^ ((a ^ b) & ~lo_mask);
}

static __always_inline __u64 swar_sub4_s16(__u64 a, __u64 b)
{
    const __u64 hi_mask = 0x8000800080008000ULL;
    const __u64 lo_mask = 0x7fff7fff7fff7fffULL;
    __u64 diff = (a | hi_mask) - (b & lo_mask);
    return diff ^ ((a ^ ~b) & hi_mask);
}

static __always_inline __u64 swar_asr4_s16(__u64 x, __u32 sh)
{
    __u64 mask = (0xFFFFULL >> sh);
    mask = mask | (mask << 16) | (mask << 32) | (mask << 48);
    __u64 sign_bit = (0x8000ULL >> sh);
    sign_bit = sign_bit | (sign_bit << 16) | (sign_bit << 32) | (sign_bit << 48);
    __u64 shifted = (x >> sh) & mask;
    return (shifted ^ sign_bit) - sign_bit;
}

static __always_inline void kssm_step_swar_64d(
    struct kssm_hidden *hi,
    struct kssm_hidden *ht,
    const struct kssm_tables *tbl,
    __u32 s1, __u32 s2, __u32 s3, __u32 s4, __u32 s5)
{
    s1 &= 31;
    __u32 i1 = KSSM_S1_OFF + s1;
    __u32 i2 = KSSM_S2_OFF + (s2 & 63);
    __u32 i3 = KSSM_S3_OFF + (s3 & 31);
    __u32 i4 = KSSM_S4_OFF + (s4 & 31);
    __u32 i5 = KSSM_S5_OFF + (s5 & 31);

    const __u64 *ei1 = (const __u64 *)tbl->emb_inode[i1];
    const __u64 *ei2 = (const __u64 *)tbl->emb_inode[i2];
    const __u64 *ei3 = (const __u64 *)tbl->emb_inode[i3];
    const __u64 *ei4 = (const __u64 *)tbl->emb_inode[i4];
    const __u64 *ei5 = (const __u64 *)tbl->emb_inode[i5];

    const __u64 *et1 = (const __u64 *)tbl->emb_task[i1];
    const __u64 *et2 = (const __u64 *)tbl->emb_task[i2];
    const __u64 *et3 = (const __u64 *)tbl->emb_task[i3];
    const __u64 *et4 = (const __u64 *)tbl->emb_task[i4];
    const __u64 *et5 = (const __u64 *)tbl->emb_task[i5];

    __u64 *vi = (__u64 *)hi->h;
    __u64 *vt = (__u64 *)ht->h;

    /* 4 multi-timescale groups x 4 packs = 16 packs (64 dims) */
    #pragma clang loop unroll(full)
    for (int grp = 0; grp < 4; grp++) {
        __u32 dsh = grp + 1; /* 1, 2, 3, 4 -> decay 0.5, 0.75, 0.875, 0.9375 */
        __u32 csh = grp + 2; /* 2, 3, 4, 5 -> coupling 0.25, 0.125, 0.0625, 0.03125 */
        #pragma clang loop unroll(disable)
        for (int k = 0; k < 4; k++) {
            int g = (grp << 2) + k;
            __u64 oi = vi[g];
            __u64 ot = vt[g];

            __u64 bi = swar_add4_s16(swar_add4_s16(ei1[g], ei2[g]),
                                     swar_add4_s16(swar_add4_s16(ei3[g], ei4[g]), ei5[g]));
            __u64 bt = swar_add4_s16(swar_add4_s16(et1[g], et2[g]),
                                     swar_add4_s16(swar_add4_s16(et3[g], et4[g]), et5[g]));

            vi[g] = swar_add4_s16(swar_sub4_s16(oi, swar_asr4_s16(oi, dsh)),
                                  swar_add4_s16(swar_asr4_s16(ot, csh), bi));
            vt[g] = swar_add4_s16(swar_sub4_s16(ot, swar_asr4_s16(ot, dsh)),
                                  swar_add4_s16(swar_asr4_s16(oi, csh), bt));
        }
    }
}

/* --------------------------------------------------------------------------
 * Core SSM Kernel 4: Exact Q7.8 Fixed-Point Diagonal Coupled SSM (D=32)
 * -------------------------------------------------------------------------- */
static __always_inline void kssm_step_exact_32d(
    struct kssm_hidden *hi,
    struct kssm_hidden *ht,
    const struct kssm_tables *tbl,
    const struct kssm_readout *ro,
    __u32 s1, __u32 s2, __u32 s3, __u32 s4, __u32 s5,
    __s16 *out_scan, __s16 *out_life)
{
    s1 &= 31;
    __u32 i1 = KSSM_S1_OFF + s1;
    __u32 i2 = KSSM_S2_OFF + (s2 & 63);
    __u32 i3 = KSSM_S3_OFF + (s3 & 31);
    __u32 i4 = KSSM_S4_OFF + (s4 & 31);
    __u32 i5 = KSSM_S5_OFF + (s5 & 31);

    __s32 acc_scan = ro->b_scan;
    __s32 acc_life = ro->b_life;

    #pragma clang loop unroll(disable)
    for (int c = 0; c < KSSM_DIM_32; c++) {
        __s32 oi = hi->h[c];
        __s32 ot = ht->h[c];

        __s32 bi = (__s32)tbl->emb_inode[i1][c]
                 + (__s32)tbl->emb_inode[i2][c]
                 + (__s32)tbl->emb_inode[i3][c]
                 + (__s32)tbl->emb_inode[i4][c]
                 + (__s32)tbl->emb_inode[i5][c];

        __s32 bt = (__s32)tbl->emb_task[i1][c]
                 + (__s32)tbl->emb_task[i2][c]
                 + (__s32)tbl->emb_task[i3][c]
                 + (__s32)tbl->emb_task[i4][c]
                 + (__s32)tbl->emb_task[i5][c];

        __s16 si = sat16_branchless(((oi * (__s32)tbl->alpha_inode[s1][c]
                                    + ot * (__s32)tbl->gamma_t2i[s1][c]) >> 8) + bi);
        __s16 st = sat16_branchless(((ot * (__s32)tbl->alpha_task[s1][c]
                                    + oi * (__s32)tbl->eta_i2t[s1][c]) >> 8) + bt);

        hi->h[c] = si;
        ht->h[c] = st;

        acc_scan += (__s32)si * (__s32)ro->w_scan_i[c] + (__s32)st * (__s32)ro->w_scan_t[c];
        acc_life += (__s32)si * (__s32)ro->w_life_i[c] + (__s32)st * (__s32)ro->w_life_t[c];
    }

    *out_scan = sat16_branchless(acc_scan >> 8);
    *out_life = sat16_branchless(acc_life >> 8);
}

/* ==========================================================================
 * PART 1: Isolated In-Kernel Microbenchmarks via BPF_PROG_TEST_RUN (XDP)
 * ========================================================================== */
SEC("xdp")
int bench_v0_noop(struct xdp_md *ctx)
{
    return XDP_PASS;
}

SEC("xdp")
int bench_v1_exact_q78_64d(struct xdp_md *ctx)
{
    __u32 k = 0;
    struct bench_pair *bp = bpf_map_lookup_elem(&bench_state_map, &k);
    struct kssm_tables *tbl = bpf_map_lookup_elem(&tbl_map, &k);
    struct kssm_readout *ro = bpf_map_lookup_elem(&readout_map, &k);
    if (!bp || !tbl || !ro)
        return XDP_PASS;

    __s16 sc = 0, lf = 0;
    kssm_step_exact_64d(&bp->h_i, &bp->h_t, tbl, ro,
                        bp->s1, bp->s2, bp->s3, bp->s4, bp->s5,
                        &sc, &lf, 1);
    bp->s1 = (bp->s1 + 1) & 31;
    bp->sink = sc + lf;
    return XDP_PASS;
}

SEC("xdp")
int bench_v1b_exact_q78_64d_noreadout(struct xdp_md *ctx)
{
    __u32 k = 0;
    struct bench_pair *bp = bpf_map_lookup_elem(&bench_state_map, &k);
    struct kssm_tables *tbl = bpf_map_lookup_elem(&tbl_map, &k);
    if (!bp || !tbl)
        return XDP_PASS;

    kssm_step_exact_64d(&bp->h_i, &bp->h_t, tbl, 0,
                        bp->s1, bp->s2, bp->s3, bp->s4, bp->s5,
                        0, 0, 0);
    bp->s1 = (bp->s1 + 1) & 31;
    return XDP_PASS;
}

SEC("xdp")
int bench_v2_shift_gate_64d(struct xdp_md *ctx)
{
    __u32 k = 0;
    struct bench_pair *bp = bpf_map_lookup_elem(&bench_state_map, &k);
    struct kssm_tables *tbl = bpf_map_lookup_elem(&tbl_map, &k);
    struct kssm_readout *ro = bpf_map_lookup_elem(&readout_map, &k);
    if (!bp || !tbl || !ro)
        return XDP_PASS;

    __s16 sc = 0, lf = 0;
    kssm_step_shift_64d(&bp->h_i, &bp->h_t, tbl, ro,
                        bp->s1, bp->s2, bp->s3, bp->s4, bp->s5,
                        &sc, &lf, 1);
    bp->s1 = (bp->s1 + 1) & 31;
    bp->sink = sc + lf;
    return XDP_PASS;
}

SEC("xdp")
int bench_v2b_shift_gate_64d_noreadout(struct xdp_md *ctx)
{
    __u32 k = 0;
    struct bench_pair *bp = bpf_map_lookup_elem(&bench_state_map, &k);
    struct kssm_tables *tbl = bpf_map_lookup_elem(&tbl_map, &k);
    if (!bp || !tbl)
        return XDP_PASS;

    kssm_step_shift_64d(&bp->h_i, &bp->h_t, tbl, 0,
                        bp->s1, bp->s2, bp->s3, bp->s4, bp->s5,
                        0, 0, 0);
    bp->s1 = (bp->s1 + 1) & 31;
    return XDP_PASS;
}

SEC("xdp")
int bench_v3_swar_4lane_64d(struct xdp_md *ctx)
{
    __u32 k = 0;
    struct bench_pair *bp = bpf_map_lookup_elem(&bench_state_map, &k);
    struct kssm_tables *tbl = bpf_map_lookup_elem(&tbl_map, &k);
    if (!bp || !tbl)
        return XDP_PASS;

    kssm_step_swar_64d(&bp->h_i, &bp->h_t, tbl,
                       bp->s1, bp->s2, bp->s3, bp->s4, bp->s5);
    bp->s1 = (bp->s1 + 1) & 31;
    return XDP_PASS;
}

SEC("xdp")
int bench_v4_exact_q78_32d(struct xdp_md *ctx)
{
    __u32 k = 0;
    struct bench_pair *bp = bpf_map_lookup_elem(&bench_state_map, &k);
    struct kssm_tables *tbl = bpf_map_lookup_elem(&tbl_map, &k);
    struct kssm_readout *ro = bpf_map_lookup_elem(&readout_map, &k);
    if (!bp || !tbl || !ro)
        return XDP_PASS;

    __s16 sc = 0, lf = 0;
    kssm_step_exact_32d(&bp->h_i, &bp->h_t, tbl, ro,
                        bp->s1, bp->s2, bp->s3, bp->s4, bp->s5,
                        &sc, &lf);
    bp->s1 = (bp->s1 + 1) & 31;
    bp->sink = sc + lf;
    return XDP_PASS;
}

/* ==========================================================================
 * PART 2: Native TASK_STORAGE + INODE_STORAGE LSM Verifier Verification (5.15)
 * ========================================================================== */
SEC("lsm/file_open")
int BPF_PROG(lsm_kssm_native_64d, struct file *file)
{
    struct task_struct *t = bpf_get_current_task_btf();
    struct kssm_task_state *ts = bpf_task_storage_get(&task_storage, t, 0,
                                                      BPF_LOCAL_STORAGE_GET_F_CREATE);
    if (!ts)
        return 0;

    struct inode *ino = file->f_inode;
    if (!ino)
        return 0;

    struct kssm_inode_state *is = bpf_inode_storage_get(&inode_storage, ino, 0,
                                                        BPF_LOCAL_STORAGE_GET_F_CREATE);
    if (!is)
        return 0;

    __u32 k = 0;
    struct kssm_tables *tbl = bpf_map_lookup_elem(&tbl_map, &k);
    struct kssm_readout *ro = bpf_map_lookup_elem(&readout_map, &k);
    if (!tbl || !ro)
        return 0;

    __u32 flags = file->f_flags;
    __u32 s1 = (flags & 0100) ? 1 : ((flags & 3) ? 2 : 0);
    __u32 s2 = (ts->recent_writes > ts->recent_reads) ? 16 : 0;
    __u64 sz = (__u64)ino->i_size;
    __u32 s3 = (fast_log2_u64(sz + 1) >> 2) & 31;
    __u32 s4 = 0;
    __u32 s5 = is->parent_churn_bucket & 31;

    kssm_step_exact_64d(&is->hidden, &ts->hidden, tbl, ro,
                        s1, s2, s3, s4, s5,
                        &is->score_scan, &is->score_life, 1);
    return 0;
}

/* ==========================================================================
 * PART 3: Production-Active Linux 5.15 VFS fentry Engine
 * ========================================================================== */
static __always_inline __u32 compute_s2_recurrence(void *ctx, struct kssm_task_state *ts, int is_write)
{
    if (is_write)
        ts->recent_writes++;
    else
        ts->recent_reads++;

    if ((ts->recent_reads + ts->recent_writes) > 256) {
        ts->recent_reads >>= 1;
        ts->recent_writes >>= 1;
    }

    __u32 total = ts->recent_reads + ts->recent_writes + 1;
    __u32 wr_q = (ts->recent_writes * 4) / total;
    __u32 rw_bucket = wr_q & 3;

    __u32 dist = 15;
    if (cfg_use_stackid) {
        long sid = bpf_get_stackid(ctx, &stack_map, BPF_F_FAST_STACK_CMP);
        if (sid >= 0) {
            __u32 u_sid = (__u32)sid;
            __u32 head = ts->ring_head & 15;
            #pragma clang loop unroll(disable)
            for (int d = 0; d < KSSM_STACK_RING_SZ; d++) {
                __u32 idx = (head - d) & 15;
                if (ts->stack_ring[idx] == u_sid && dist == 15)
                    dist = d;
            }
            __u32 next_head = (head + 1) & 15;
            ts->stack_ring[next_head] = u_sid;
            ts->ring_head = next_head;
        }
    }
    return ((rw_bucket << 4) | (dist & 15)) & 63;
}

static __always_inline struct kssm_inode_state *get_or_init_inode_state(__u64 ino_key)
{
    struct kssm_inode_state *is = bpf_map_lookup_elem(&inode_lru_map, &ino_key);
    if (is)
        return is;

    __u32 k0 = 0;
    struct kssm_inode_state *zero = bpf_map_lookup_elem(&zero_inode_map, &k0);
    if (!zero)
        return 0;
    bpf_map_update_elem(&inode_lru_map, &ino_key, zero, BPF_NOEXIST);
    return bpf_map_lookup_elem(&inode_lru_map, &ino_key);
}

static __always_inline int handle_vfs_rw(void *ctx, struct file *file, size_t count, loff_t *pos, int is_write)
{
    __u32 k0 = 0;
    struct kssm_cpu_stats *st = bpf_map_lookup_elem(&stats_map, &k0);
    if (st)
        st->total_hooks++;

    if (!file)
        return 0;

    struct inode *ino = file->f_inode;
    if (!ino)
        return 0;

    __u16 mode = ino->i_mode;
    if ((mode & 0170000) != 0100000)
        return 0;

    __u64 off = 0;
    if (pos)
        bpf_probe_read_kernel(&off, sizeof(off), pos);

    __u64 ino_key = (__u64)ino;
    struct kssm_inode_state *is = get_or_init_inode_state(ino_key);
    if (!is)
        return 0;

    __u64 cluster = off >> KSSM_CLUSTER_SHIFT;
    __s64 stride = (__s64)off - (__s64)is->last_off;
    __s64 expected_stride = (__s64)is->last_size;
    int is_seq = (stride == expected_stride) || (stride == 0);
    int op_changed = (is->last_op != (__u8)(is_write ? 2 : 1));
    int cluster_crossed = (cluster != is->last_cluster);
    int stride_inflection = !is_seq;

    is->last_off = off;
    is->last_size = (__u32)count;
    is->last_op = is_write ? 2 : 1;

    if (cfg_use_gating && is->initialized && !op_changed && !cluster_crossed && !stride_inflection) {
        if (st)
            st->fast_skips++;
        return 0;
    }

    is->last_cluster = cluster;
    is->last_stride = stride;
    is->initialized = 1;

    struct task_struct *t = bpf_get_current_task_btf();
    struct kssm_task_state *ts = bpf_task_storage_get(&task_storage, t, 0,
                                                      BPF_LOCAL_STORAGE_GET_F_CREATE);
    if (!ts)
        return 0;

    struct kssm_tables *tbl = bpf_map_lookup_elem(&tbl_map, &k0);
    struct kssm_readout *ro = bpf_map_lookup_elem(&readout_map, &k0);
    if (!tbl || !ro)
        return 0;

    __u64 fsz = (__u64)ino->i_size;

    __u32 s1;
    if (!is_write) {
        s1 = is_seq ? 4 : 5;
    } else {
        s1 = (off >= fsz) ? 6 : (is_seq ? 7 : 8);
    }

    __u32 s2 = compute_s2_recurrence(ctx, ts, is_write);

    __u32 sz_b = fast_log2_u64(fsz + 1) >> 2;
    if (sz_b > 7) sz_b = 7;
    __u32 region = 0;
    if (off < 65536)
        region = 0;
    else if (off + count >= fsz)
        region = 3;
    else if (off < (fsz >> 1))
        region = 1;
    else
        region = 2;
    __u32 s3 = ((sz_b << 2) | region) & 31;

    __u32 sz_log = fast_log2_u64((__u64)count + 1);
    if (sz_log > 20) sz_log = 20;
    __u32 req_b = (sz_log >= 12) ? ((sz_log - 12) >> 1) : 0;
    if (req_b > 3) req_b = 3;
    __u32 str_b = is_seq ? 0 : (stride < 0 ? 1 : (stride > 131072 ? 3 : 2));
    __u32 s4 = ((req_b << 3) | str_b) & 31;

    __u32 s5 = is->parent_churn_bucket & 31;

    if (cfg_use_shift_ssm == 1) {
        kssm_step_shift_64d(&is->hidden, &ts->hidden, tbl, ro,
                            s1, s2, s3, s4, s5,
                            &is->score_scan, &is->score_life, 1);
    } else if (cfg_use_shift_ssm == 2) {
        kssm_step_swar_64d(&is->hidden, &ts->hidden, tbl,
                           s1, s2, s3, s4, s5);
        is->score_scan = is->hidden.h[0];
        is->score_life = is->hidden.h[1];
    } else {
        kssm_step_exact_64d(&is->hidden, &ts->hidden, tbl, ro,
                            s1, s2, s3, s4, s5,
                            &is->score_scan, &is->score_life, 1);
    }

    if (st) {
        st->ssm_updates++;
        if (is->score_scan > 0) st->scan_preds++;
        if (is->score_life > 0) st->short_life_preds++;
    }
    return 0;
}

SEC("fentry/vfs_read")
int BPF_PROG(vfs_empty_read, struct file *file, char *buf, size_t count, loff_t *pos)
{
    return 0;
}

SEC("fentry/vfs_write")
int BPF_PROG(vfs_empty_write, struct file *file, const char *buf, size_t count, loff_t *pos)
{
    return 0;
}

SEC("fentry/vfs_read")
int BPF_PROG(vfs_kssm_read, struct file *file, char *buf, size_t count, loff_t *pos)
{
    return handle_vfs_rw(ctx, file, count, pos, 0);
}

SEC("fentry/vfs_write")
int BPF_PROG(vfs_kssm_write, struct file *file, const char *buf, size_t count, loff_t *pos)
{
    return handle_vfs_rw(ctx, file, count, pos, 1);
}

SEC("fentry/vfs_open")
int BPF_PROG(vfs_kssm_open, const struct path *path, struct file *file)
{
    if (!path || !file)
        return 0;
    struct dentry *dentry = path->dentry;
    if (!dentry)
        return 0;
    struct inode *ino = dentry->d_inode;
    if (!ino)
        return 0;
    if ((ino->i_mode & 0170000) != 0100000)
        return 0;

    struct kssm_inode_state *is = get_or_init_inode_state((__u64)ino);
    if (!is)
        return 0;

    __u64 parent_key = (__u64)dentry->d_parent;
    struct dir_churn *dc = bpf_map_lookup_elem(&dir_churn_map, &parent_key);
    if (dc)
        is->parent_churn_bucket = fast_log2_u64(dc->ema_events + 1) & 31;

    struct task_struct *t = bpf_get_current_task_btf();
    struct kssm_task_state *ts = bpf_task_storage_get(&task_storage, t, 0,
                                                      BPF_LOCAL_STORAGE_GET_F_CREATE);
    if (!ts)
        return 0;

    __u32 k0 = 0;
    struct kssm_tables *tbl = bpf_map_lookup_elem(&tbl_map, &k0);
    struct kssm_readout *ro = bpf_map_lookup_elem(&readout_map, &k0);
    if (!tbl || !ro)
        return 0;

    __u32 flags = file->f_flags;
    __u32 s1 = (flags & 0100) ? 1 : ((flags & 3) ? 2 : 0);
    __u32 s2 = compute_s2_recurrence(ctx, ts, (flags & 3) != 0);
    __u32 s3 = (fast_log2_u64((__u64)ino->i_size + 1) >> 2) & 31;
    __u32 s4 = 0;
    __u32 s5 = is->parent_churn_bucket & 31;

    if (cfg_use_shift_ssm == 1) {
        kssm_step_shift_64d(&is->hidden, &ts->hidden, tbl, ro,
                            s1, s2, s3, s4, s5,
                            &is->score_scan, &is->score_life, 1);
    } else if (cfg_use_shift_ssm == 2) {
        kssm_step_swar_64d(&is->hidden, &ts->hidden, tbl,
                           s1, s2, s3, s4, s5);
    } else {
        kssm_step_exact_64d(&is->hidden, &ts->hidden, tbl, ro,
                            s1, s2, s3, s4, s5,
                            &is->score_scan, &is->score_life, 1);
    }
    return 0;
}

SEC("fentry/vfs_unlink")
int BPF_PROG(vfs_kssm_unlink, struct user_namespace *mnt_userns, struct inode *dir, struct dentry *dentry)
{
    if (!dir || !dentry)
        return 0;
    __u64 pkey = (__u64)dentry->d_parent;
    struct dir_churn *dc = bpf_map_lookup_elem(&dir_churn_map, &pkey);
    __u64 now = bpf_ktime_get_ns();
    if (dc) {
        if (now - dc->last_ts_ns > 5000000000ULL)
            dc->ema_events >>= 1;
        dc->ema_events += 1;
        dc->last_ts_ns = now;
    } else {
        struct dir_churn init = { .last_ts_ns = now, .ema_events = 1 };
        bpf_map_update_elem(&dir_churn_map, &pkey, &init, BPF_NOEXIST);
    }
    if (dentry->d_inode) {
        __u64 ino_key = (__u64)dentry->d_inode;
        bpf_map_delete_elem(&inode_lru_map, &ino_key);
    }
    return 0;
}
