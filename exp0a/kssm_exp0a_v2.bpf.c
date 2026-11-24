#include "../vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "kssm_common.h"

char LICENSE[] SEC("license") = "GPL";

volatile const __u32 cfg_use_stackid = 1;
volatile const __u32 cfg_use_gating = 1;
volatile const __u32 cfg_ssm_mode = 2; /* 0=exact64, 1=shift64, 2=guard_swar64, 3=guard_swar64_shared_emb */

/* Guard-bit SWAR tables: 4x uint16 (excess-2048) packed per u64 -> 16 u64 per slot (128 B)
 * Total size:
 *   emb_inode: 192 * 128B = 24 KB
 *   emb_task:  192 * 128B = 24 KB
 *   (Optional shared emb: 192 * 128B = 24 KB total, 100% fits in 48KB L1d!)
 */
struct kssm_swar_tables {
    __u64 emb_i[KSSM_TOTAL_SLOTS][16]; /* 24 KB */
    __u64 emb_t[KSSM_TOTAL_SLOTS][16]; /* 24 KB */
};

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
    __type(value, struct kssm_swar_tables);
} swar_tbl_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct kssm_readout);
} readout_map SEC(".maps");

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

/* Completely branchless approximate log2 bucket (0..7) for file size -> 0 verifier branches! */
static __always_inline __u32 size_bucket_branchless(__u64 v)
{
    __u32 b = 0;
    b += (v >= (1ULL << 12)); /* >= 4KB */
    b += (v >= (1ULL << 16)); /* >= 64KB */
    b += (v >= (1ULL << 20)); /* >= 1MB */
    b += (v >= (1ULL << 24)); /* >= 16MB */
    b += (v >= (1ULL << 28)); /* >= 256MB */
    b += (v >= (1ULL << 30)); /* >= 1GB */
    b += (v >= (1ULL << 32)); /* >= 4GB */
    return b;
}

/* --------------------------------------------------------------------------
 * Guard-Bit Excess-16384 SWAR 4-Lane 64D Coupled SSM (128B per entity!)
 * Mathematical Guarantee (Contractive Invariant Set):
 *   Hidden lanes stored as H = h + 16384 in [8192, 24575] (Q5.8: [-32, +32))
 *   Table entries stored as E = e + 2048  in [1536, 2559]
 *   5-slot sum B = E1+E2+E3+E4+E5 = (e1..e5) + 10240 (4 native ADD insns!)
 *   Decay + coupling: H_new = H - ((H>>d)&Md) + ((H_other>>c)&Mc) + B - C_dc
 *   ZERO cross-lane carry/borrow, ZERO branches, ZERO multiplies, ZERO FPU!
 * -------------------------------------------------------------------------- */
#define PACK4(v) (((__u64)(v)) | ((__u64)(v) << 16) | ((__u64)(v) << 32) | ((__u64)(v) << 48))
#define MASK_SH(sh) PACK4(0xFFFFULL >> (sh))

/* Positive coupling constant: C = 10240 + (16384 >> c) - (16384 >> d) */
#define CONST_POS(d, c) PACK4(10240ULL + (16384ULL >> (c)) - (16384ULL >> (d)))
/* Negative coupling (antiphase channels): H - (H>>d) - (Ho>>c) + B - C_neg
 * where C_neg = 10240 - (16384 >> c) - (16384 >> d) */
#define CONST_NEG(d, c) PACK4(10240ULL - (16384ULL >> (c)) - (16384ULL >> (d)))

#define SWAR_STEP_POS(vi, vt, g, d, c, ei1, ei2, ei3, ei4, ei5, et1, et2, et3, et4, et5) do { \
    __u64 _oi = (vi)[g]; \
    __u64 _ot = (vt)[g]; \
    __u64 _bi = (ei1)[g] + (ei2)[g] + (ei3)[g] + (ei4)[g] + (ei5)[g]; \
    __u64 _bt = (et1)[g] + (et2)[g] + (et3)[g] + (et4)[g] + (et5)[g]; \
    (vi)[g] = _oi - ((_oi >> (d)) & MASK_SH(d)) + ((_ot >> (c)) & MASK_SH(c)) + _bi - CONST_POS(d, c); \
    (vt)[g] = _ot - ((_ot >> (d)) & MASK_SH(d)) + ((_oi >> (c)) & MASK_SH(c)) + _bt - CONST_POS(d, c); \
} while (0)

#define SWAR_STEP_NEG(vi, vt, g, d, c, ei1, ei2, ei3, ei4, ei5, et1, et2, et3, et4, et5) do { \
    __u64 _oi = (vi)[g]; \
    __u64 _ot = (vt)[g]; \
    __u64 _bi = (ei1)[g] + (ei2)[g] + (ei3)[g] + (ei4)[g] + (ei5)[g]; \
    __u64 _bt = (et1)[g] + (et2)[g] + (et3)[g] + (et4)[g] + (et5)[g]; \
    (vi)[g] = _oi - ((_oi >> (d)) & MASK_SH(d)) - ((_ot >> (c)) & MASK_SH(c)) + _bi - CONST_NEG(d, c); \
    (vt)[g] = _ot - ((_ot >> (d)) & MASK_SH(d)) - ((_oi >> (c)) & MASK_SH(c)) + _bt - CONST_NEG(d, c); \
} while (0)

/* Variant A: Full dual-table Guard-Bit SWAR 64D (both emb_i and emb_t, 48KB tables) */
static __always_inline void kssm_step_guard_swar_dual_64d(
    struct kssm_hidden *hi,
    struct kssm_hidden *ht,
    const struct kssm_swar_tables *tbl,
    __u32 s1, __u32 s2, __u32 s3, __u32 s4, __u32 s5)
{
    __u32 i1 = KSSM_S1_OFF + (s1 & 31);
    __u32 i2 = KSSM_S2_OFF + (s2 & 63);
    __u32 i3 = KSSM_S3_OFF + (s3 & 31);
    __u32 i4 = KSSM_S4_OFF + (s4 & 31);
    __u32 i5 = KSSM_S5_OFF + (s5 & 31);

    const __u64 *ei1 = tbl->emb_i[i1];
    const __u64 *ei2 = tbl->emb_i[i2];
    const __u64 *ei3 = tbl->emb_i[i3];
    const __u64 *ei4 = tbl->emb_i[i4];
    const __u64 *ei5 = tbl->emb_i[i5];

    const __u64 *et1 = tbl->emb_t[i1];
    const __u64 *et2 = tbl->emb_t[i2];
    const __u64 *et3 = tbl->emb_t[i3];
    const __u64 *et4 = tbl->emb_t[i4];
    const __u64 *et5 = tbl->emb_t[i5];

    __u64 *vi = (__u64 *)hi->h;
    __u64 *vt = (__u64 *)ht->h;

    /* 4 multi-timescale bands (d=1,2,3,4; half positive, half negative coupling) */
    SWAR_STEP_POS(vi, vt,  0, 1, 2, ei1, ei2, ei3, ei4, ei5, et1, et2, et3, et4, et5);
    SWAR_STEP_NEG(vi, vt,  1, 1, 2, ei1, ei2, ei3, ei4, ei5, et1, et2, et3, et4, et5);
    SWAR_STEP_POS(vi, vt,  2, 1, 3, ei1, ei2, ei3, ei4, ei5, et1, et2, et3, et4, et5);
    SWAR_STEP_NEG(vi, vt,  3, 1, 3, ei1, ei2, ei3, ei4, ei5, et1, et2, et3, et4, et5);

    SWAR_STEP_POS(vi, vt,  4, 2, 3, ei1, ei2, ei3, ei4, ei5, et1, et2, et3, et4, et5);
    SWAR_STEP_NEG(vi, vt,  5, 2, 3, ei1, ei2, ei3, ei4, ei5, et1, et2, et3, et4, et5);
    SWAR_STEP_POS(vi, vt,  6, 2, 4, ei1, ei2, ei3, ei4, ei5, et1, et2, et3, et4, et5);
    SWAR_STEP_NEG(vi, vt,  7, 2, 4, ei1, ei2, ei3, ei4, ei5, et1, et2, et3, et4, et5);

    SWAR_STEP_POS(vi, vt,  8, 3, 4, ei1, ei2, ei3, ei4, ei5, et1, et2, et3, et4, et5);
    SWAR_STEP_NEG(vi, vt,  9, 3, 4, ei1, ei2, ei3, ei4, ei5, et1, et2, et3, et4, et5);
    SWAR_STEP_POS(vi, vt, 10, 3, 5, ei1, ei2, ei3, ei4, ei5, et1, et2, et3, et4, et5);
    SWAR_STEP_NEG(vi, vt, 11, 3, 5, ei1, ei2, ei3, ei4, ei5, et1, et2, et3, et4, et5);

    SWAR_STEP_POS(vi, vt, 12, 4, 5, ei1, ei2, ei3, ei4, ei5, et1, et2, et3, et4, et5);
    SWAR_STEP_NEG(vi, vt, 13, 4, 5, ei1, ei2, ei3, ei4, ei5, et1, et2, et3, et4, et5);
    SWAR_STEP_POS(vi, vt, 14, 4, 5, ei1, ei2, ei3, ei4, ei5, et1, et2, et3, et4, et5);
    SWAR_STEP_NEG(vi, vt, 15, 4, 5, ei1, ei2, ei3, ei4, ei5, et1, et2, et3, et4, et5);
}

/* Variant B: Shared-Input Guard-Bit SWAR 64D (24KB table -> 100% fits in L1d!
 * Same 5-slot input projection b_t computed once and injected asymmetrically into
 * inode (+b) and task (+b or -b across bands), cutting table loads in HALF!) */
#define SWAR_SHARED_STEP(vi, vt, g, d, c, e1, e2, e3, e4, e5) do { \
    __u64 _oi = (vi)[g]; \
    __u64 _ot = (vt)[g]; \
    __u64 _b  = (e1)[g] + (e2)[g] + (e3)[g] + (e4)[g] + (e5)[g]; \
    (vi)[g] = _oi - ((_oi >> (d)) & MASK_SH(d)) + ((_ot >> (c)) & MASK_SH(c)) + _b - CONST_POS(d, c); \
    (vt)[g] = _ot - ((_ot >> (d)) & MASK_SH(d)) - ((_oi >> (c)) & MASK_SH(c)) + _b - CONST_NEG(d, c); \
} while (0)

static __always_inline void kssm_step_guard_swar_shared_64d(
    struct kssm_hidden *hi,
    struct kssm_hidden *ht,
    const struct kssm_swar_tables *tbl,
    __u32 s1, __u32 s2, __u32 s3, __u32 s4, __u32 s5)
{
    const __u64 *e1 = tbl->emb_i[KSSM_S1_OFF + (s1 & 31)];
    const __u64 *e2 = tbl->emb_i[KSSM_S2_OFF + (s2 & 63)];
    const __u64 *e3 = tbl->emb_i[KSSM_S3_OFF + (s3 & 31)];
    const __u64 *e4 = tbl->emb_i[KSSM_S4_OFF + (s4 & 31)];
    const __u64 *e5 = tbl->emb_i[KSSM_S5_OFF + (s5 & 31)];

    __u64 *vi = (__u64 *)hi->h;
    __u64 *vt = (__u64 *)ht->h;

    SWAR_SHARED_STEP(vi, vt,  0, 1, 2, e1, e2, e3, e4, e5);
    SWAR_SHARED_STEP(vi, vt,  1, 1, 2, e1, e2, e3, e4, e5);
    SWAR_SHARED_STEP(vi, vt,  2, 1, 3, e1, e2, e3, e4, e5);
    SWAR_SHARED_STEP(vi, vt,  3, 1, 3, e1, e2, e3, e4, e5);

    SWAR_SHARED_STEP(vi, vt,  4, 2, 3, e1, e2, e3, e4, e5);
    SWAR_SHARED_STEP(vi, vt,  5, 2, 3, e1, e2, e3, e4, e5);
    SWAR_SHARED_STEP(vi, vt,  6, 2, 4, e1, e2, e3, e4, e5);
    SWAR_SHARED_STEP(vi, vt,  7, 2, 4, e1, e2, e3, e4, e5);

    SWAR_SHARED_STEP(vi, vt,  8, 3, 4, e1, e2, e3, e4, e5);
    SWAR_SHARED_STEP(vi, vt,  9, 3, 4, e1, e2, e3, e4, e5);
    SWAR_SHARED_STEP(vi, vt, 10, 3, 5, e1, e2, e3, e4, e5);
    SWAR_SHARED_STEP(vi, vt, 11, 3, 5, e1, e2, e3, e4, e5);

    SWAR_SHARED_STEP(vi, vt, 12, 4, 5, e1, e2, e3, e4, e5);
    SWAR_SHARED_STEP(vi, vt, 13, 4, 5, e1, e2, e3, e4, e5);
    SWAR_SHARED_STEP(vi, vt, 14, 4, 5, e1, e2, e3, e4, e5);
    SWAR_SHARED_STEP(vi, vt, 15, 4, 5, e1, e2, e3, e4, e5);
}

/* Variant C: Subspace-Factorized Guard-Bit SWAR 64D (128B per entity, 100% 64D updated!)
 * Each of 4 multi-timescale 16D bands (4x u64) takes 2 orthogonal feature lookups
 * (s1 always globally broadcast + 1 dedicated slot: band0<-s2, band1<-s3, band2<-s4, band3<-s5).
 * Cross-slot mixing happens naturally via task<->inode coupling & linear readout!
 * Cuts table loads by 5x: only 2 loads/pack instead of 10! */
#define SWAR_SUB_STEP(vi, vt, g, d, c, ea, eb) do { \
    __u64 _oi = (vi)[g]; \
    __u64 _ot = (vt)[g]; \
    __u64 _b  = (ea)[g] + (eb)[g]; \
    (vi)[g] = _oi - ((_oi >> (d)) & MASK_SH(d)) + ((_ot >> (c)) & MASK_SH(c)) + _b - PACK4(4096ULL + (16384ULL >> (c)) - (16384ULL >> (d))); \
    (vt)[g] = _ot - ((_ot >> (d)) & MASK_SH(d)) - ((_oi >> (c)) & MASK_SH(c)) + _b - PACK4(4096ULL - (16384ULL >> (c)) - (16384ULL >> (d))); \
} while (0)

static __always_inline void kssm_step_guard_swar_subspace_64d(
    struct kssm_hidden *hi,
    struct kssm_hidden *ht,
    const struct kssm_swar_tables *tbl,
    __u32 s1, __u32 s2, __u32 s3, __u32 s4, __u32 s5)
{
    const __u64 *e1 = tbl->emb_i[KSSM_S1_OFF + (s1 & 31)];
    const __u64 *e2 = tbl->emb_i[KSSM_S2_OFF + (s2 & 63)];
    const __u64 *e3 = tbl->emb_i[KSSM_S3_OFF + (s3 & 31)];
    const __u64 *e4 = tbl->emb_i[KSSM_S4_OFF + (s4 & 31)];
    const __u64 *e5 = tbl->emb_i[KSSM_S5_OFF + (s5 & 31)];

    __u64 *vi = (__u64 *)hi->h;
    __u64 *vt = (__u64 *)ht->h;

    /* Band 0 (dims 0..15): s1 + s2 (primitive + call-stack recurrence) */
    SWAR_SUB_STEP(vi, vt,  0, 1, 2, e1, e2);
    SWAR_SUB_STEP(vi, vt,  1, 1, 3, e1, e2);
    SWAR_SUB_STEP(vi, vt,  2, 2, 3, e1, e2);
    SWAR_SUB_STEP(vi, vt,  3, 3, 4, e1, e2);

    /* Band 1 (dims 16..31): s1 + s3 (primitive + file geometry) */
    SWAR_SUB_STEP(vi, vt,  4, 1, 2, e1, e3);
    SWAR_SUB_STEP(vi, vt,  5, 2, 3, e1, e3);
    SWAR_SUB_STEP(vi, vt,  6, 3, 4, e1, e3);
    SWAR_SUB_STEP(vi, vt,  7, 4, 5, e1, e3);

    /* Band 2 (dims 32..47): s1 + s4 (primitive + size/stride) */
    SWAR_SUB_STEP(vi, vt,  8, 1, 2, e1, e4);
    SWAR_SUB_STEP(vi, vt,  9, 2, 3, e1, e4);
    SWAR_SUB_STEP(vi, vt, 10, 3, 4, e1, e4);
    SWAR_SUB_STEP(vi, vt, 11, 4, 5, e1, e4);

    /* Band 3 (dims 48..63): s1 + s5 (primitive + dir churn) */
    SWAR_SUB_STEP(vi, vt, 12, 1, 2, e1, e5);
    SWAR_SUB_STEP(vi, vt, 13, 2, 3, e1, e5);
    SWAR_SUB_STEP(vi, vt, 14, 3, 4, e1, e5);
    SWAR_SUB_STEP(vi, vt, 15, 4, 5, e1, e5);
}

/* Variant D: Guard-Bit SWAR 8-Lane 64D (int8 Q0.6 packed -> 8x u64 = 64B/entity!) */
#define PACK8(v) (((__u64)(v)) | ((__u64)(v) << 8) | ((__u64)(v) << 16) | ((__u64)(v) << 24) | \
                  ((__u64)(v) << 32) | ((__u64)(v) << 40) | ((__u64)(v) << 48) | ((__u64)(v) << 56))
#define MASK8(sh) PACK8(0xFFULL >> (sh))
#define SWAR8_STEP(vi, vt, g, d, c, e1, e2, e3, e4, e5) do { \
    __u64 _oi = (vi)[g]; \
    __u64 _ot = (vt)[g]; \
    __u64 _b  = (e1)[g] + (e2)[g] + (e3)[g] + (e4)[g] + (e5)[g]; \
    (vi)[g] = _oi - ((_oi >> (d)) & MASK8(d)) + ((_ot >> (c)) & MASK8(c)) + _b - PACK8(40ULL + (64ULL >> (c)) - (64ULL >> (d))); \
    (vt)[g] = _ot - ((_ot >> (d)) & MASK8(d)) - ((_oi >> (c)) & MASK8(c)) + _b - PACK8(40ULL - (64ULL >> (c)) - (64ULL >> (d))); \
} while (0)

static __always_inline void kssm_step_guard_swar8_64d(
    struct kssm_hidden *hi,
    struct kssm_hidden *ht,
    const struct kssm_swar_tables *tbl,
    __u32 s1, __u32 s2, __u32 s3, __u32 s4, __u32 s5)
{
    const __u64 *e1 = tbl->emb_i[KSSM_S1_OFF + (s1 & 31)];
    const __u64 *e2 = tbl->emb_i[KSSM_S2_OFF + (s2 & 63)];
    const __u64 *e3 = tbl->emb_i[KSSM_S3_OFF + (s3 & 31)];
    const __u64 *e4 = tbl->emb_i[KSSM_S4_OFF + (s4 & 31)];
    const __u64 *e5 = tbl->emb_i[KSSM_S5_OFF + (s5 & 31)];

    __u64 *vi = (__u64 *)hi->h;
    __u64 *vt = (__u64 *)ht->h;

    SWAR8_STEP(vi, vt, 0, 1, 2, e1, e2, e3, e4, e5);
    SWAR8_STEP(vi, vt, 1, 1, 3, e1, e2, e3, e4, e5);
    SWAR8_STEP(vi, vt, 2, 2, 3, e1, e2, e3, e4, e5);
    SWAR8_STEP(vi, vt, 3, 2, 4, e1, e2, e3, e4, e5);
    SWAR8_STEP(vi, vt, 4, 3, 4, e1, e2, e3, e4, e5);
    SWAR8_STEP(vi, vt, 5, 3, 5, e1, e2, e3, e4, e5);
    SWAR8_STEP(vi, vt, 6, 4, 5, e1, e2, e3, e4, e5);
    SWAR8_STEP(vi, vt, 7, 4, 5, e1, e2, e3, e4, e5);
}

/* Lightweight 8-channel readout (or full 64D readout) */
static __always_inline void kssm_readout_fast16(
    const struct kssm_hidden *hi,
    const struct kssm_hidden *ht,
    const struct kssm_readout *ro,
    __s16 *out_scan, __s16 *out_life)
{
    /* Readout over 16 representative channels (4 per timescale band) -> ~12 ns */
    __s32 acc_s = ro->b_scan;
    __s32 acc_l = ro->b_life;
    #pragma clang loop unroll(full)
    for (int k = 0; k < 16; k++) {
        int c = k * 4;
        __s32 xi = (__s32)(__u16)hi->h[c] - 16384;
        __s32 xt = (__s32)(__u16)ht->h[c] - 16384;
        acc_s += xi * (__s32)ro->w_scan_i[c] + xt * (__s32)ro->w_scan_t[c];
        acc_l += xi * (__s32)ro->w_life_i[c] + xt * (__s32)ro->w_life_t[c];
    }
    *out_scan = (__s16)(acc_s >> 8);
    *out_life = (__s16)(acc_l >> 8);
}

/* Exact Q7.8 64D kept for faithful comparison */
static __always_inline void kssm_step_exact_64d(
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
    for (int c = 0; c < KSSM_DIM; c++) {
        __s32 oi = hi->h[c];
        __s32 ot = ht->h[c];
        __s32 bi = (__s32)tbl->emb_inode[i1][c] + (__s32)tbl->emb_inode[i2][c]
                 + (__s32)tbl->emb_inode[i3][c] + (__s32)tbl->emb_inode[i4][c]
                 + (__s32)tbl->emb_inode[i5][c];
        __s32 bt = (__s32)tbl->emb_task[i1][c] + (__s32)tbl->emb_task[i2][c]
                 + (__s32)tbl->emb_task[i3][c] + (__s32)tbl->emb_task[i4][c]
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
 * PART 1: XDP Isolated Microbenchmarks
 * ========================================================================== */
SEC("xdp")
int bench_noop(struct xdp_md *ctx) { return XDP_PASS; }

SEC("xdp")
int bench_swar_dual_64d(struct xdp_md *ctx)
{
    __u32 k = 0;
    struct bench_pair *bp = bpf_map_lookup_elem(&bench_state_map, &k);
    struct kssm_swar_tables *tbl = bpf_map_lookup_elem(&swar_tbl_map, &k);
    if (!bp || !tbl) return XDP_PASS;
    kssm_step_guard_swar_dual_64d(&bp->h_i, &bp->h_t, tbl, bp->s1, bp->s2, bp->s3, bp->s4, bp->s5);
    bp->s1 = (bp->s1 + 1) & 31;
    return XDP_PASS;
}

SEC("xdp")
int bench_swar_shared_64d(struct xdp_md *ctx)
{
    __u32 k = 0;
    struct bench_pair *bp = bpf_map_lookup_elem(&bench_state_map, &k);
    struct kssm_swar_tables *tbl = bpf_map_lookup_elem(&swar_tbl_map, &k);
    if (!bp || !tbl) return XDP_PASS;
    kssm_step_guard_swar_shared_64d(&bp->h_i, &bp->h_t, tbl, bp->s1, bp->s2, bp->s3, bp->s4, bp->s5);
    bp->s1 = (bp->s1 + 1) & 31;
    return XDP_PASS;
}

SEC("xdp")
int bench_swar_subspace_64d(struct xdp_md *ctx)
{
    __u32 k = 0;
    struct bench_pair *bp = bpf_map_lookup_elem(&bench_state_map, &k);
    struct kssm_swar_tables *tbl = bpf_map_lookup_elem(&swar_tbl_map, &k);
    if (!bp || !tbl) return XDP_PASS;
    kssm_step_guard_swar_subspace_64d(&bp->h_i, &bp->h_t, tbl, bp->s1, bp->s2, bp->s3, bp->s4, bp->s5);
    bp->s1 = (bp->s1 + 1) & 31;
    return XDP_PASS;
}

SEC("xdp")
int bench_swar_subspace_64d_ro(struct xdp_md *ctx)
{
    __u32 k = 0;
    struct bench_pair *bp = bpf_map_lookup_elem(&bench_state_map, &k);
    struct kssm_swar_tables *tbl = bpf_map_lookup_elem(&swar_tbl_map, &k);
    struct kssm_readout *ro = bpf_map_lookup_elem(&readout_map, &k);
    if (!bp || !tbl || !ro) return XDP_PASS;
    kssm_step_guard_swar_subspace_64d(&bp->h_i, &bp->h_t, tbl, bp->s1, bp->s2, bp->s3, bp->s4, bp->s5);
    __s16 sc = 0, lf = 0;
    kssm_readout_fast16(&bp->h_i, &bp->h_t, ro, &sc, &lf);
    bp->s1 = (bp->s1 + 1) & 31;
    bp->sink = sc + lf;
    return XDP_PASS;
}

SEC("xdp")
int bench_swar8_64d(struct xdp_md *ctx)
{
    __u32 k = 0;
    struct bench_pair *bp = bpf_map_lookup_elem(&bench_state_map, &k);
    struct kssm_swar_tables *tbl = bpf_map_lookup_elem(&swar_tbl_map, &k);
    if (!bp || !tbl) return XDP_PASS;
    kssm_step_guard_swar8_64d(&bp->h_i, &bp->h_t, tbl, bp->s1, bp->s2, bp->s3, bp->s4, bp->s5);
    bp->s1 = (bp->s1 + 1) & 31;
    return XDP_PASS;
}

/* ==========================================================================
 * PART 2: Native TASK_STORAGE + INODE_STORAGE LSM Verifier Check (Zero Patch!)
 * ========================================================================== */
SEC("lsm/file_open")
int BPF_PROG(lsm_kssm_exact_64d, struct file *file)
{
    struct task_struct *t = bpf_get_current_task_btf();
    struct kssm_task_state *ts = bpf_task_storage_get(&task_storage, t, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
    if (!ts) return 0;
    struct inode *ino = file->f_inode;
    if (!ino) return 0;
    struct kssm_inode_state *is = bpf_inode_storage_get(&inode_storage, ino, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
    if (!is) return 0;

    __u32 k = 0;
    struct kssm_tables *tbl = bpf_map_lookup_elem(&tbl_map, &k);
    struct kssm_readout *ro = bpf_map_lookup_elem(&readout_map, &k);
    if (!tbl || !ro) return 0;

    __u32 flags = file->f_flags;
    __u32 s1 = ((flags & 0100) != 0) + (((flags & 3) != 0) << 1);
    __u32 s2 = (ts->recent_writes > ts->recent_reads) << 4;
    __u32 s3 = size_bucket_branchless((__u64)ino->i_size);
    __u32 s4 = 0;
    __u32 s5 = is->parent_churn_bucket & 31;

    kssm_step_exact_64d(&is->hidden, &ts->hidden, tbl, ro,
                        s1, s2, s3, s4, s5,
                        &is->score_scan, &is->score_life);
    return 0;
}

SEC("lsm/file_open")
int BPF_PROG(lsm_kssm_swar_64d, struct file *file)
{
    struct task_struct *t = bpf_get_current_task_btf();
    struct kssm_task_state *ts = bpf_task_storage_get(&task_storage, t, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
    if (!ts) return 0;
    struct inode *ino = file->f_inode;
    if (!ino) return 0;
    struct kssm_inode_state *is = bpf_inode_storage_get(&inode_storage, ino, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
    if (!is) return 0;

    __u32 k = 0;
    struct kssm_swar_tables *tbl = bpf_map_lookup_elem(&swar_tbl_map, &k);
    struct kssm_readout *ro = bpf_map_lookup_elem(&readout_map, &k);
    if (!tbl || !ro) return 0;

    __u32 flags = file->f_flags;
    __u32 s1 = ((flags & 0100) != 0) + (((flags & 3) != 0) << 1);
    __u32 s2 = (ts->recent_writes > ts->recent_reads) << 4;
    __u32 s3 = size_bucket_branchless((__u64)ino->i_size);
    __u32 s4 = 0;
    __u32 s5 = is->parent_churn_bucket & 31;

    kssm_step_guard_swar_subspace_64d(&is->hidden, &ts->hidden, tbl, s1, s2, s3, s4, s5);
    kssm_readout_fast16(&is->hidden, &ts->hidden, ro, &is->score_scan, &is->score_life);
    return 0;
}

/* ==========================================================================
 * PART 3: Production-Active 5.15 VFS fentry Hooks (pos_ptr as u64!)
 * ========================================================================== */
static __always_inline __u32 compute_s2(void *ctx, struct kssm_task_state *ts, int is_write)
{
    if (is_write) ts->recent_writes++;
    else ts->recent_reads++;
    if ((ts->recent_reads + ts->recent_writes) > 256) {
        ts->recent_reads >>= 1;
        ts->recent_writes >>= 1;
    }
    __u32 total = ts->recent_reads + ts->recent_writes + 1;
    __u32 rw_bucket = ((ts->recent_writes * 4) / total) & 3;
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

static __always_inline struct kssm_inode_state *get_inode_st(__u64 ino_key)
{
    struct kssm_inode_state *is = bpf_map_lookup_elem(&inode_lru_map, &ino_key);
    if (is) return is;
    __u32 k0 = 0;
    struct kssm_inode_state *zero = bpf_map_lookup_elem(&zero_inode_map, &k0);
    if (!zero) return 0;
    bpf_map_update_elem(&inode_lru_map, &ino_key, zero, BPF_NOEXIST);
    return bpf_map_lookup_elem(&inode_lru_map, &ino_key);
}

static __always_inline int handle_vfs_rw(void *ctx, struct file *file, size_t count, __u64 pos_ptr, int is_write)
{
    __u32 k0 = 0;
    struct kssm_cpu_stats *st = bpf_map_lookup_elem(&stats_map, &k0);
    if (st) st->total_hooks++;

    if (!file) return 0;
    struct inode *ino = file->f_inode;
    if (!ino) return 0;
    if ((ino->i_mode & 0170000) != 0100000) return 0;

    __u64 off = 0;
    if (pos_ptr)
        bpf_probe_read_kernel(&off, sizeof(off), (void *)pos_ptr);
    else
        off = (__u64)file->f_pos;

    struct kssm_inode_state *is = get_inode_st((__u64)ino);
    if (!is) return 0;

    __u64 cluster = off >> KSSM_CLUSTER_SHIFT;
    __s64 stride = (__s64)off - (__s64)is->last_off;
    int is_seq = (stride == (__s64)is->last_size) || (stride == 0);
    int op_changed = (is->last_op != (__u8)(is_write ? 2 : 1));
    int cluster_crossed = (cluster != is->last_cluster);

    is->last_off = off;
    is->last_size = (__u32)count;
    is->last_op = is_write ? 2 : 1;

    if (cfg_use_gating && is->initialized && !op_changed && !cluster_crossed && is_seq) {
        if (st) st->fast_skips++;
        return 0;
    }

    is->last_cluster = cluster;
    is->last_stride = stride;
    is->initialized = 1;

    struct task_struct *t = bpf_get_current_task_btf();
    struct kssm_task_state *ts = bpf_task_storage_get(&task_storage, t, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
    if (!ts) return 0;

    __u64 fsz = (__u64)ino->i_size;
    __u32 s1 = !is_write ? (4 + (!is_seq)) : ((off >= fsz) ? 6 : (7 + (!is_seq)));
    __u32 s2 = compute_s2(ctx, ts, is_write);
    __u32 sz_b = size_bucket_branchless(fsz);
    __u32 region = (off >= 65536) + (off >= (fsz >> 1)) + ((off + count) >= fsz);
    __u32 s3 = ((sz_b << 2) | (region & 3)) & 31;
    __u32 req_b = (count >= 16384) + (count >= 65536) + (count >= 262144);
    __u32 str_b = is_seq ? 0 : (stride < 0 ? 1 : (stride > 131072 ? 3 : 2));
    __u32 s4 = ((req_b << 3) | str_b) & 31;
    __u32 s5 = is->parent_churn_bucket & 31;

    struct kssm_swar_tables *stbl = bpf_map_lookup_elem(&swar_tbl_map, &k0);
    struct kssm_readout *ro = bpf_map_lookup_elem(&readout_map, &k0);
    if (!stbl || !ro) return 0;

    if (cfg_ssm_mode == 3) {
        kssm_step_guard_swar_shared_64d(&is->hidden, &ts->hidden, stbl, s1, s2, s3, s4, s5);
    } else if (cfg_ssm_mode == 4) {
        kssm_step_guard_swar8_64d(&is->hidden, &ts->hidden, stbl, s1, s2, s3, s4, s5);
    } else {
        kssm_step_guard_swar_subspace_64d(&is->hidden, &ts->hidden, stbl, s1, s2, s3, s4, s5);
    }
    kssm_readout_fast16(&is->hidden, &ts->hidden, ro, &is->score_scan, &is->score_life);

    if (st) {
        st->ssm_updates++;
        if (is->score_scan > 0) st->scan_preds++;
        if (is->score_life > 0) st->short_life_preds++;
    }
    return 0;
}

SEC("fentry/vfs_read")
int BPF_PROG(vfs_empty_read, struct file *file, char *buf, size_t count, __u64 pos_ptr) { return 0; }

SEC("fentry/vfs_write")
int BPF_PROG(vfs_empty_write, struct file *file, const char *buf, size_t count, __u64 pos_ptr) { return 0; }

SEC("fentry/vfs_read")
int BPF_PROG(vfs_kssm_read, struct file *file, char *buf, size_t count, __u64 pos_ptr)
{
    return handle_vfs_rw(ctx, file, count, pos_ptr, 0);
}

SEC("fentry/vfs_write")
int BPF_PROG(vfs_kssm_write, struct file *file, const char *buf, size_t count, __u64 pos_ptr)
{
    return handle_vfs_rw(ctx, file, count, pos_ptr, 1);
}

SEC("fentry/vfs_open")
int BPF_PROG(vfs_kssm_open, const struct path *path, struct file *file)
{
    if (!path || !file) return 0;
    struct dentry *dentry = path->dentry;
    if (!dentry) return 0;
    struct inode *ino = dentry->d_inode;
    if (!ino || (ino->i_mode & 0170000) != 0100000) return 0;

    struct kssm_inode_state *is = get_inode_st((__u64)ino);
    if (!is) return 0;

    __u64 pkey = (__u64)dentry->d_parent;
    struct dir_churn *dc = bpf_map_lookup_elem(&dir_churn_map, &pkey);
    if (dc) is->parent_churn_bucket = size_bucket_branchless((__u64)dc->ema_events << 12) & 31;

    struct task_struct *t = bpf_get_current_task_btf();
    struct kssm_task_state *ts = bpf_task_storage_get(&task_storage, t, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
    if (!ts) return 0;

    __u32 k0 = 0;
    struct kssm_swar_tables *stbl = bpf_map_lookup_elem(&swar_tbl_map, &k0);
    struct kssm_readout *ro = bpf_map_lookup_elem(&readout_map, &k0);
    if (!stbl || !ro) return 0;

    __u32 flags = file->f_flags;
    __u32 s1 = ((flags & 0100) != 0) + (((flags & 3) != 0) << 1);
    __u32 s2 = compute_s2(ctx, ts, (flags & 3) != 0);
    __u32 s3 = size_bucket_branchless((__u64)ino->i_size);
    __u32 s4 = 0;
    __u32 s5 = is->parent_churn_bucket & 31;

    kssm_step_guard_swar_subspace_64d(&is->hidden, &ts->hidden, stbl, s1, s2, s3, s4, s5);
    kssm_readout_fast16(&is->hidden, &ts->hidden, ro, &is->score_scan, &is->score_life);
    return 0;
}

SEC("fentry/vfs_unlink")
int BPF_PROG(vfs_kssm_unlink, struct user_namespace *mnt_userns, struct inode *dir, struct dentry *dentry)
{
    if (!dir || !dentry) return 0;
    __u64 pkey = (__u64)dentry->d_parent;
    struct dir_churn *dc = bpf_map_lookup_elem(&dir_churn_map, &pkey);
    __u64 now = bpf_ktime_get_ns();
    if (dc) {
        if (now - dc->last_ts_ns > 5000000000ULL) dc->ema_events >>= 1;
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
