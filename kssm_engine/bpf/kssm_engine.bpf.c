#include "../../vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "../include/kssm_abi.h"

char LICENSE[] SEC("license") = "GPL";

volatile const __u32 cfg_target_sdev = 0;
volatile const __u32 cfg_gating_mode = KSSM_GATING_TL_PERCPU;
volatile const __u32 cfg_enable_actions = 1; /* 1=emit ringbuf hints, 0=pure inference overhead test */
volatile const __s32 cfg_scan_thresh_q8 = 0;

/* Per-task private state in native BPF_MAP_TYPE_TASK_STORAGE (zero lock, L1d-private!) */
struct kssm_task_st {
    __u64 h[16];           /* 128 B: 64D Excess-16384 Q5.8 hidden state */
    __u64 last_ino;
    __u64 last_off;
    __u32 last_cnt;
    __u32 bytes_acc;       /* Thread-local 128KB fast-path budget -> ZERO cross-core bouncing! */
    __u32 seq_run;
    __u32 stack_ring[16];
    __u16 ring_head;
    __u16 recent_reads;
    __u16 recent_writes;
    __u8  last_mode;
    __u8  initialized;
};

/* Per-inode state (Per-CPU LRU_HASH on 5.15 fentry; native INODE_STORAGE on LSM/6.4+) */
struct kssm_inode_st {
    __u64 h[16];           /* 128 B: 64D Excess-16384 Q5.8 hidden state */
    __u64 birth_hi[16];    /* Birth snapshot for zero-BPTT slow loop */
    __u64 birth_ht[16];
    __u64 birth_ts_ns;
    __u64 last_off;
    __u64 last_drop_off;
    __u32 last_size;
    __u32 bytes_acc;
    __u32 consec_scan;
    __u32 parent_churn;
    __u8  last_mode;
    __u8  pred_class;
    __u8  has_birth;
    __u8  pad;
};

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct kssm_base_model);
} base_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct kssm_dual_head);
} head_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct kssm_inode_st);
} zero_ino_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_TASK_STORAGE);
    __uint(map_flags, BPF_F_NO_PREALLOC);
    __type(key, int);
    __type(value, struct kssm_task_st);
} task_storage SEC(".maps");

/* Native INODE_STORAGE (verified via LSM hook on 5.15; natively usable in fentry on 6.4+) */
struct {
    __uint(type, BPF_MAP_TYPE_INODE_STORAGE);
    __uint(map_flags, BPF_F_NO_PREALLOC);
    __type(key, int);
    __type(value, struct kssm_inode_st);
} inode_storage SEC(".maps");

/* Production 5.15 fentry inode map: Per-CPU LRU free list (NO global LRU lock!) */
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(map_flags, BPF_F_NO_COMMON_LRU);
    __uint(max_entries, 131072);
    __type(key, __u64);
    __type(value, struct kssm_inode_st);
} inode_lru_nocommon SEC(".maps");

/* Contention comparison baseline: Common-LRU global lock map */
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 131072);
    __type(key, __u64);
    __type(value, struct kssm_inode_st);
} inode_lru_common SEC(".maps");

struct dir_st { __u64 last_ts; __u32 ema; __u32 pad; };
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(map_flags, BPF_F_NO_COMMON_LRU);
    __uint(max_entries, 16384);
    __type(key, __u64);
    __type(value, struct dir_st);
} dir_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_STACK_TRACE);
    __uint(key_size, sizeof(__u32));
    __uint(value_size, 16 * sizeof(__u64));
    __uint(max_entries, 16384);
} stack_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 8 * 1024 * 1024);
} ctrl_rb SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct kssm_cpu_metrics);
} metrics_map SEC(".maps");

struct evt_scr { __u32 s1, s2, s3, s4, s5, pad; };
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct evt_scr);
} scr_map SEC(".maps");

/* Per-CPU scratch for isolated XDP microbenchmark of pruned g_s1[32][4] kernel */
struct bench_u_pair { __u64 vi[16]; __u64 vt[16]; __u32 s1, s2, s3, s4, s5, pad; };
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct bench_u_pair);
} bench_u_map SEC(".maps");

#define PACK4(v) (((__u64)(v)) | ((__u64)(v) << 16) | ((__u64)(v) << 32) | ((__u64)(v) << 48))
#define MASK_SH(sh) PACK4(0xFFFFULL >> (sh))
#define CONST_POS2(d, c) PACK4(4096ULL + (16384ULL >> (c)) - (16384ULL >> (d)))
#define CONST_NEG2(d, c) PACK4(4096ULL - (16384ULL >> (c)) - (16384ULL >> (d)))

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

static __always_inline int check_dev(struct inode *ino)
{
    if (!ino) return 0;
    if ((ino->i_mode & 0170000) != 0100000) return 0;
    if (cfg_target_sdev) {
        struct super_block *sb = ino->i_sb;
        if (!sb || sb->s_dev != cfg_target_sdev) return 0;
    }
    return 1;
}

static __always_inline __u32 fast_sz_b(__u64 v)
{
    if (v < (1ULL << 16)) return (v >= (1ULL << 12)) ? 1 : 0;
    if (v < (1ULL << 24)) return (v >= (1ULL << 20)) ? 3 : 2;
    if (v < (1ULL << 30)) return (v >= (1ULL << 28)) ? 5 : 4;
    return (v >= (1ULL << 32)) ? 7 : 6;
}

static __always_inline void init_task_neutral(struct kssm_task_st *ts)
{
    if (ts->initialized) return;
    __u64 n = PACK4(16384ULL);
    #pragma clang loop unroll(full)
    for (int g = 0; g < 16; g++) ts->h[g] = n;
    ts->initialized = 1;
}

static __always_inline struct kssm_inode_st *get_inode_st(__u64 ino_num)
{
    void *map_ptr = (cfg_gating_mode == KSSM_GATING_COMMON_LRU) ?
                    (void *)&inode_lru_common : (void *)&inode_lru_nocommon;
    struct kssm_inode_st *is = bpf_map_lookup_elem(map_ptr, &ino_num);
    if (is) return is;
    __u32 k0 = 0;
    struct kssm_inode_st *z = bpf_map_lookup_elem(&zero_ino_map, &k0);
    if (!z) return 0;
    bpf_map_update_elem(map_ptr, &ino_num, z, BPF_NOEXIST);
    return bpf_map_lookup_elem(map_ptr, &ino_num);
}

static __always_inline __u32 bump_dir_churn(__u64 pino, int bump)
{
    if (!pino) return 0;
    __u64 now = bpf_ktime_get_ns();
    struct dir_st *ds = bpf_map_lookup_elem(&dir_map, &pino);
    if (ds) {
        if (now > ds->last_ts && (now - ds->last_ts) > 5000000000ULL) ds->ema >>= 1;
        if (bump) { ds->ema += 1; ds->last_ts = now; }
        return ds->ema;
    }
    if (bump) {
        struct dir_st init = { .last_ts = now, .ema = 1 };
        bpf_map_update_elem(&dir_map, &pino, &init, BPF_NOEXIST);
        return 1;
    }
    return 0;
}

static __always_inline __u32 compute_s2(void *ctx, struct kssm_task_st *ts, int is_write)
{
    if (is_write) ts->recent_writes++; else ts->recent_reads++;
    if ((ts->recent_reads + ts->recent_writes) > 256) {
        ts->recent_reads >>= 1; ts->recent_writes >>= 1;
    }
    __u32 rw_q = ((ts->recent_writes * 4) / (ts->recent_reads + ts->recent_writes + 1)) & 3;
    __u32 dist = 15;
    long sid = bpf_get_stackid(ctx, &stack_map, BPF_F_USER_STACK | BPF_F_FAST_STACK_CMP);
    if (sid < 0) sid = bpf_get_stackid(ctx, &stack_map, BPF_F_FAST_STACK_CMP);
    if (sid >= 0) {
        __u32 usid = (__u32)sid; __u32 hd = ts->ring_head & 15;
        #pragma clang loop unroll(full)
        for (int d = 15; d >= 0; d--) {
            __u32 idx = (hd - d) & 15;
            __u32 diff = ts->stack_ring[idx] ^ usid;
            __u32 nz = (diff | (__u32)(-(__s32)diff)) >> 31;
            __u32 m = (__u32)(-(__s32)(1U - nz));
            dist = ((__u32)d & m) | (dist & ~m);
        }
        __u32 nh = (hd + 1) & 15; ts->stack_ring[nh] = usid; ts->ring_head = nh;
    }
    return ((rw_q << 4) | (dist & 15)) & 63;
}

/* Core Pruned 4-Band Selective Guard-SWAR 64D (g_s1[32][4] = 128 B, ~35-48 ns!) */
static __always_inline void kssm_sel4_step(
    __u64 *vi, __u64 *vt,
    const struct kssm_base_model *bm,
    const struct evt_scr *ev)
{
    const volatile struct evt_scr *vev = ev;
    __u64 s1 = (__u64)vev->s1 & 31ULL;
    __u64 i1 = KSSM_S1_OFF + s1;
    __u64 i2 = KSSM_S2_OFF + ((__u64)vev->s2 & 63ULL);
    __u64 i3 = KSSM_S3_OFF + ((__u64)vev->s3 & 31ULL);
    __u64 i4 = KSSM_S4_OFF + ((__u64)vev->s4 & 31ULL);
    __u64 i5 = KSSM_S5_OFF + ((__u64)vev->s5 & 31ULL);

    const __u64 *e1 = bm->emb[i1];
    const __u64 *e2 = bm->emb[i2];
    const __u64 *e3 = bm->emb[i3];
    const __u64 *e4 = bm->emb[i4];
    const __u64 *e5 = bm->emb[i5];

    __u8 r0 = bm->reg_s1[s1][0] & 3;
    __u8 r1 = bm->reg_s1[s1][1] & 3;
    __u8 r2 = bm->reg_s1[s1][2] & 3;
    __u8 r3 = bm->reg_s1[s1][3] & 3;

    RUN_BAND(vi, vt,  0, r0, e1, e2);
    RUN_BAND(vi, vt,  4, r1, e1, e3);
    RUN_BAND(vi, vt,  8, r2, e1, e4);
    RUN_BAND(vi, vt, 12, r3, e1, e5);
}

static __always_inline __s32 readout_scan_q8(const __u64 *vi, const __u64 *vt, const struct kssm_dual_head *hd)
{
    const __u16 *hi = (const __u16 *)vi;
    const __u16 *ht = (const __u16 *)vt;
    __s32 acc = hd->b_scan;
    #pragma clang loop unroll(disable)
    for (int c = 0; c < KSSM_DIM; c++) {
        acc += ((__s32)hi[c] - 16384) * (__s32)hd->w_scan_i[c]
             + ((__s32)ht[c] - 16384) * (__s32)hd->w_scan_t[c];
    }
    return acc >> 8;
}

static __always_inline __u8 readout_life_cls(const __u64 *vi, const __u64 *vt, const struct kssm_dual_head *hd)
{
    const __u16 *hi = (const __u16 *)vi;
    const __u16 *ht = (const __u16 *)vt;
    __s32 s0 = hd->b_life[0], s1 = hd->b_life[1], s2 = hd->b_life[2];
    #pragma clang loop unroll(disable)
    for (int c = 0; c < KSSM_DIM; c++) {
        __s32 xi = (__s32)hi[c] - 16384;
        __s32 xt = (__s32)ht[c] - 16384;
        s0 += xi * (__s32)hd->w_life_i[0][c] + xt * (__s32)hd->w_life_t[0][c];
        s1 += xi * (__s32)hd->w_life_i[1][c] + xt * (__s32)hd->w_life_t[1][c];
        s2 += xi * (__s32)hd->w_life_i[2][c] + xt * (__s32)hd->w_life_t[2][c];
    }
    if (s2 >= s1 && s2 >= s0) return 2;
    if (s1 >= s0) return 1;
    return 0;
}

/* ==========================================================================
 * PART A: Isolated XDP Microbenchmark for Pruned g_s1[32][4] Kernel
 * ========================================================================== */
SEC("xdp") int bench_noop(struct xdp_md *ctx) { return XDP_PASS; }

SEC("xdp") int bench_unified_sel4(struct xdp_md *ctx)
{
    __u32 k0 = 0;
    struct bench_u_pair *bp = bpf_map_lookup_elem(&bench_u_map, &k0);
    struct kssm_base_model *bm = bpf_map_lookup_elem(&base_map, &k0);
    struct evt_scr *ev = bpf_map_lookup_elem(&scr_map, &k0);
    if (!bp || !bm || !ev) return XDP_PASS;
    ev->s1 = bp->s1; ev->s2 = bp->s2; ev->s3 = bp->s3; ev->s4 = bp->s4; ev->s5 = bp->s5;
    kssm_sel4_step(bp->vi, bp->vt, bm, ev);
    bp->s1 = (bp->s1 + 1) & 31;
    return XDP_PASS;
}

/* ==========================================================================
 * PART B: Native TASK_STORAGE + INODE_STORAGE LSM Verifier Verification (5.15)
 * ========================================================================== */
SEC("lsm/file_open")
int BPF_PROG(lsm_kssm_unified_native, struct file *file)
{
    struct task_struct *t = bpf_get_current_task_btf();
    struct kssm_task_st *ts = bpf_task_storage_get(&task_storage, t, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
    if (!ts || !file || !file->f_inode) return 0;
    struct inode *ino = file->f_inode;
    struct kssm_inode_st *is = bpf_inode_storage_get(&inode_storage, ino, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
    if (!is) return 0;
    __u32 k0 = 0;
    struct kssm_base_model *bm = bpf_map_lookup_elem(&base_map, &k0);
    struct kssm_dual_head *hd = bpf_map_lookup_elem(&head_map, &k0);
    struct evt_scr *ev = bpf_map_lookup_elem(&scr_map, &k0);
    if (!bm || !hd || !ev) return 0;
    init_task_neutral(ts);
    ev->s1 = ((file->f_flags & 0100) ? 1 : 0);
    ev->s2 = 0;
    ev->s3 = fast_sz_b((__u64)ino->i_size) << 2;
    ev->s4 = 0; ev->s5 = 8;
    kssm_sel4_step(is->h, ts->h, bm, ev);
    is->pred_class = readout_life_cls(is->h, ts->h, hd);
    return 0;
}

/* ==========================================================================
 * PART C: Production Unified 5-Hook VFS/Ext4 Engine (Dual-Head + Slow Loop!)
 * ========================================================================== */
static __always_inline int handle_rw(void *ctx, struct kiocb *iocb, struct iov_iter *iter, int is_write)
{
    if (!iocb || !iter || !iocb->ki_filp) return 0;
    struct inode *ino = iocb->ki_filp->f_inode;
    if (!check_dev(ino)) return 0;

    __u32 k0 = 0;
    struct kssm_cpu_metrics *mt = bpf_map_lookup_elem(&metrics_map, &k0);
    if (mt) mt->total_hooks++;

    __u64 ino_num = ino->i_ino;
    __u64 off = (__u64)iocb->ki_pos;
    __u32 cnt = (__u32)iter->count;

    struct task_struct *t = bpf_get_current_task_btf();
    struct kssm_task_st *ts = bpf_task_storage_get(&task_storage, t, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
    if (!ts) return 0;
    init_task_neutral(ts);

    __s64 stride = 0;
    int is_seq = 0;
    int same_file = (ts->last_ino == ino_num);
    if (same_file) {
        stride = (__s64)off - (__s64)ts->last_off;
        __u64 lim = ((__u64)ts->last_cnt << 1);
        if (lim < 131072ULL) lim = 131072ULL;
        is_seq = (stride > 0 && (__u64)stride <= lim) || (stride == 0 && cnt >= 65536U);
    }
    __u8 cur_mode = is_write ? (is_seq ? 3 : 4) : (is_seq ? 1 : 2);

    /* FAST-PATH GATING:
     * Mode 0 (TL_PERCPU): checks thread-private TASK_STORAGE ONLY -> ZERO shared inode_map
     * lookup or cache-line write on 96.9% of I/Os -> linear scaling even when 64 cores pound 1 hot file! */
    if (cfg_gating_mode == KSSM_GATING_TL_PERCPU) {
        __u32 acc = ts->bytes_acc + cnt;
        int skip = same_file && (ts->last_mode == cur_mode) && (acc < 131072U);
        ts->last_ino = ino_num;
        ts->last_off = off;
        ts->last_cnt = cnt;
        if (is_seq) { if (ts->seq_run < 31) ts->seq_run++; } else { ts->seq_run = 0; }
        if (skip) {
            ts->bytes_acc = acc;
            if (mt) mt->fast_skips++;
            return 0;
        }
        ts->last_mode = cur_mode;
        ts->bytes_acc = 0;
    }

    struct kssm_inode_st *is = get_inode_st(ino_num);
    if (!is) return 0;

    if (cfg_gating_mode != KSSM_GATING_TL_PERCPU) {
        __u32 acc = is->bytes_acc + cnt;
        int skip = (is->last_mode == cur_mode) && (acc < 131072U);
        is->last_off = off; is->last_size = cnt;
        ts->last_ino = ino_num; ts->last_off = off; ts->last_cnt = cnt;
        if (is_seq) { if (ts->seq_run < 31) ts->seq_run++; } else { ts->seq_run = 0; }
        if (skip) {
            is->bytes_acc = acc;
            if (mt) mt->fast_skips++;
            return 0;
        }
        is->last_mode = cur_mode;
        is->bytes_acc = 0;
    }

    struct kssm_base_model *bm = bpf_map_lookup_elem(&base_map, &k0);
    struct kssm_dual_head *hd = bpf_map_lookup_elem(&head_map, &k0);
    struct evt_scr *ev = bpf_map_lookup_elem(&scr_map, &k0);
    if (!bm || !hd || !ev) return 0;

    __u64 fsz = (__u64)ino->i_size;
    ev->s1 = !is_write ? (is_seq ? 8 : 9) : ((off >= fsz) ? 10 : (is_seq ? 11 : 12));
    ev->s2 = compute_s2(ctx, ts, is_write);
    __u32 reg = (off < 65536ULL) ? 0 : (((off + cnt) >= fsz) ? 3 : ((off < (fsz >> 1)) ? 1 : 2));
    ev->s3 = ((fast_sz_b(fsz) << 2) | reg) & 31;
    __u32 req_b = (cnt >= 16384U) + (cnt >= 65536U) + (cnt >= 262144U);
    __u32 str_b = is_seq ? ((ts->seq_run >= 3) ? 0 : 4) : ((stride < 0) ? 1 : 5);
    ev->s4 = ((req_b << 3) | str_b) & 31;
    ev->s5 = 8;

    kssm_sel4_step(is->h, ts->h, bm, ev);
    if (mt) mt->ssm_updates++;

    __s32 sc_q8 = readout_scan_q8(is->h, ts->h, hd);
    if (!is_write) {
        if (sc_q8 > cfg_scan_thresh_q8) {
            is->consec_scan++;
            if (mt) mt->scan_votes++;
        } else {
            is->consec_scan = 0;
            if (mt) mt->hot_votes++;
        }
        if (cfg_enable_actions) {
            if (off < is->last_drop_off) is->last_drop_off = off;
            if (is->consec_scan >= 4 && off > (24ULL << 20)) {
                __u64 safe_end = off - (16ULL << 20);
                if (safe_end >= is->last_drop_off + (8ULL << 20)) {
                    __u64 dstart = is->last_drop_off;
                    __u64 dlen = safe_end - dstart;
                    if (dlen > (32ULL << 20)) { dstart = safe_end - (16ULL << 20); dlen = (16ULL << 20); }
                    is->last_drop_off = safe_end;
                    struct kssm_ctrl_msg *m = bpf_ringbuf_reserve(&ctrl_rb, sizeof(*m), 0);
                    if (m) {
                        m->ino = ino_num; m->arg0 = dstart; m->arg1 = dlen;
                        m->pid = (__u32)(bpf_get_current_pid_tgid() >> 32);
                        m->score_q8 = sc_q8; m->msg_type = KSSM_MSG_DROP_HINT; m->is_write = 0;
                        bpf_ringbuf_submit(m, 0);
                        if (mt) mt->msgs_sent++;
                    } else if (mt) mt->rb_drops++;
                }
            }
        }
    }
    return 0;
}

SEC("fentry/ext4_file_read_iter")
int BPF_PROG(kssm_ext4_read, struct kiocb *iocb, struct iov_iter *to) { return handle_rw(ctx, iocb, to, 0); }

SEC("fentry/ext4_file_write_iter")
int BPF_PROG(kssm_ext4_write, struct kiocb *iocb, struct iov_iter *from) { return handle_rw(ctx, iocb, from, 1); }

SEC("fentry/vfs_open")
int BPF_PROG(kssm_vfs_open, const struct path *path, struct file *file)
{
    if (!path || !file || !path->dentry) return 0;
    struct inode *ino = path->dentry->d_inode;
    if (!check_dev(ino)) return 0;

    __u32 k0 = 0;
    struct kssm_cpu_metrics *mt = bpf_map_lookup_elem(&metrics_map, &k0);
    if (mt) mt->total_hooks++;

    __u32 fl = file->f_flags;
    int is_creat = ((fl & 0100) != 0) && (ino->i_size == 0);
    __u64 pino = (path->dentry->d_parent && path->dentry->d_parent->d_inode) ? path->dentry->d_parent->d_inode->i_ino : 0;
    __u32 churn = bump_dir_churn(pino, is_creat);

    struct kssm_inode_st *is = get_inode_st(ino->i_ino);
    struct task_struct *t = bpf_get_current_task_btf();
    struct kssm_task_st *ts = bpf_task_storage_get(&task_storage, t, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
    struct kssm_base_model *bm = bpf_map_lookup_elem(&base_map, &k0);
    struct kssm_dual_head *hd = bpf_map_lookup_elem(&head_map, &k0);
    struct evt_scr *ev = bpf_map_lookup_elem(&scr_map, &k0);
    if (!is || !ts || !bm || !hd || !ev) return 0;
    init_task_neutral(ts);

    ev->s1 = is_creat ? (1 + ((fl & 02000) ? 1 : 0) + ((fl & 01000) ? 2 : 0)) : (((fl & 3) != 0) ? 5 : 0);
    ev->s2 = compute_s2(ctx, ts, (fl & 3) != 0);
    ev->s3 = (fast_sz_b((__u64)ino->i_size) << 2) & 31;
    ev->s4 = 0;
    ev->s5 = (((churn > 4) ? 16 : 0) + (fast_sz_b((__u64)churn << 12) << 1)) & 31;

    kssm_sel4_step(is->h, ts->h, bm, ev);
    if (mt) mt->ssm_updates++;

    if (is_creat) {
        __u8 pred = readout_life_cls(is->h, ts->h, hd);
        is->pred_class = pred;
        is->birth_ts_ns = bpf_ktime_get_ns();
        is->has_birth = 1;
        #pragma clang loop unroll(full)
        for (int g = 0; g < 16; g++) { is->birth_hi[g] = is->h[g]; is->birth_ht[g] = ts->h[g]; }
        if (mt) mt->births_pred++;

        if (cfg_enable_actions) {
            struct kssm_ctrl_msg *m = bpf_ringbuf_reserve(&ctrl_rb, sizeof(*m), 0);
            if (m) {
                m->ino = ino->i_ino; m->arg0 = is->birth_ts_ns; m->arg1 = (__u64)ino->i_size;
                m->pid = (__u32)(bpf_get_current_pid_tgid() >> 32);
                m->msg_type = KSSM_MSG_BIRTH_HINT; m->pred_class = pred;
                bpf_ringbuf_submit(m, 0);
                if (mt) mt->msgs_sent++;
            } else if (mt) mt->rb_drops++;
        }
    }
    return 0;
}

SEC("fentry/vfs_unlink")
int BPF_PROG(kssm_vfs_unlink, struct user_namespace *mnt_userns, struct inode *dir, struct dentry *dentry)
{
    if (!dir || !dentry || !dentry->d_inode) return 0;
    struct inode *ino = dentry->d_inode;
    if (!check_dev(ino)) return 0;

    bump_dir_churn(dir->i_ino, 1);
    __u64 ino_num = ino->i_ino;
    void *map_ptr = (cfg_gating_mode == KSSM_GATING_COMMON_LRU) ?
                    (void *)&inode_lru_common : (void *)&inode_lru_nocommon;
    struct kssm_inode_st *is = bpf_map_lookup_elem(map_ptr, &ino_num);
    if (is && is->has_birth) {
        __u32 k0 = 0;
        struct kssm_cpu_metrics *mt = bpf_map_lookup_elem(&metrics_map, &k0);
        __u64 now = bpf_ktime_get_ns();
        __u64 dt = (now > is->birth_ts_ns) ? (now - is->birth_ts_ns) : 0;
        __u8 tcls = (dt < 10000000000ULL) ? 0 : ((dt < 50000000000ULL) ? 1 : 2);
        if (mt) mt->deaths_obs++;

        if (cfg_enable_actions) {
            struct kssm_ctrl_msg *m = bpf_ringbuf_reserve(&ctrl_rb, sizeof(*m), 0);
            if (m) {
                m->ino = ino_num; m->arg0 = dt; m->pid = (__u32)(bpf_get_current_pid_tgid() >> 32);
                m->msg_type = KSSM_MSG_DEATH_FEEDBACK; m->pred_class = is->pred_class; m->true_class = tcls;
                const __u16 *bhi = (const __u16 *)is->birth_hi;
                const __u16 *bht = (const __u16 *)is->birth_ht;
                #pragma clang loop unroll(disable)
                for (int c = 0; c < KSSM_DIM; c++) {
                    m->snap_hi[c] = (__s16)((__s32)bhi[c] - 16384);
                    m->snap_ht[c] = (__s16)((__s32)bht[c] - 16384);
                }
                bpf_ringbuf_submit(m, 0);
                if (mt) mt->msgs_sent++;
            } else if (mt) mt->rb_drops++;
        }
    }
    bpf_map_delete_elem(map_ptr, &ino_num);
    return 0;
}

SEC("fentry/vfs_fsync_range")
int BPF_PROG(kssm_vfs_fsync, struct file *file, loff_t start, loff_t end, int datasync)
{
    if (!file || !file->f_inode) return 0;
    struct inode *ino = file->f_inode;
    if (!check_dev(ino)) return 0;
    struct kssm_inode_st *is = get_inode_st(ino->i_ino);
    struct task_struct *t = bpf_get_current_task_btf();
    struct kssm_task_st *ts = bpf_task_storage_get(&task_storage, t, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
    __u32 k0 = 0;
    struct kssm_base_model *bm = bpf_map_lookup_elem(&base_map, &k0);
    struct evt_scr *ev = bpf_map_lookup_elem(&scr_map, &k0);
    if (!is || !ts || !bm || !ev) return 0;
    init_task_neutral(ts);
    ev->s1 = 13; ev->s2 = compute_s2(ctx, ts, 1); ev->s3 = fast_sz_b((__u64)ino->i_size) << 2; ev->s4 = 0; ev->s5 = 8;
    kssm_sel4_step(is->h, ts->h, bm, ev);
    return 0;
}
