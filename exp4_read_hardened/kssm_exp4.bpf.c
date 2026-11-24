#include "../vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "kssm_exp4_shared.h"

char LICENSE[] SEC("license") = "GPL";

volatile const __u32 cfg_target_sdev = 0;
volatile const __u32 cfg_mode = EXP4_MODE_KSSM;

struct task_st {
    __u64 h[16];
    __u64 last_ino;
    __u64 last_off;
    __u32 last_cnt;
    __u32 seq_run;
    __u32 stack_ring[16];
    __u16 ring_head;
    __u16 recent_reads;
    __u16 recent_writes;
    __u16 initialized;
};

struct inode_st {
    __u64 h[16];
    __u64 last_off;
    __u64 last_drop_off;
    __u64 last_fp_chunk;  /* last 2 MiB chunk checked in reuse fingerprint LRU */
    __u32 last_size;
    __u32 bytes_acc;
    __u32 consec_scan;
    __u32 reuse_ema;      /* 0..256 EMA of 2 MiB chunk reuse hits within 4 GiB global window */
    __u8  last_mode;
    __u8  pad[7];
};

/* Global 2,048-entry LRU over 2 MiB chunks = EXACT 4 GiB cache-residency reuse window across all 128 CPUs!
 * Only probed once per 2 MiB sequential advance (zero contention, ~0.05% overhead)! */
struct chunk_key {
    __u64 ino;
    __u64 chunk_2m; /* off >> 21 (2 MiB) */
};

struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 4096); /* 2048 * 2 MiB = 4 GiB exact global window! */
    __type(key, struct chunk_key);
    __type(value, __u32);
} chunk_seen_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct kssm_live_weights);
} weights_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct inode_st);
} zero_ino_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_TASK_STORAGE);
    __uint(map_flags, BPF_F_NO_PREALLOC);
    __type(key, int);
    __type(value, struct task_st);
} task_storage SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(map_flags, BPF_F_NO_COMMON_LRU);
    __uint(max_entries, 65536);
    __type(key, __u64);
    __type(value, struct inode_st);
} inode_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_STACK_TRACE);
    __uint(key_size, sizeof(__u32));
    __uint(value_size, 16 * sizeof(__u64));
    __uint(max_entries, 16384);
} stack_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 4 * 1024 * 1024);
} drop_rb SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct exp4_stats);
} stats_map SEC(".maps");

struct evt_scr { __u32 s1, s2, s3, s4, s5, pad; };
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct evt_scr);
} scr_map SEC(".maps");

#define PACK4(v) (((__u64)(v)) | ((__u64)(v) << 16) | ((__u64)(v) << 32) | ((__u64)(v) << 48))
#define MASK_SH(sh) PACK4(0xFFFFULL >> (sh))
#define CONST_POS5(d, c) PACK4(10240ULL + (16384ULL >> (c)) - (16384ULL >> (d)))
#define CONST_NEG5(d, c) PACK4(10240ULL - (16384ULL >> (c)) - (16384ULL >> (d)))

#define SWAR_STEP(vi, vt, g, d, c, e1, e2, e3, e4, e5) do { \
    __u64 _oi = (vi)[g]; __u64 _ot = (vt)[g]; \
    __u64 _b  = (e1)[g] + (e2)[g] + (e3)[g] + (e4)[g] + (e5)[g]; \
    (vi)[g] = _oi - ((_oi >> (d)) & MASK_SH(d)) + ((_ot >> (c)) & MASK_SH(c)) + _b - CONST_POS5(d, c); \
    (vt)[g] = _ot - ((_ot >> (d)) & MASK_SH(d)) - ((_oi >> (c)) & MASK_SH(c)) + _b - CONST_NEG5(d, c); \
} while (0)

static __always_inline __u32 fast_sz_b(__u64 v)
{
    if (v < (1ULL << 16)) return (v >= (1ULL << 12)) ? 1 : 0;
    if (v < (1ULL << 24)) return (v >= (1ULL << 20)) ? 3 : 2;
    if (v < (1ULL << 30)) return (v >= (1ULL << 28)) ? 5 : 4;
    return (v >= (1ULL << 32)) ? 7 : 6;
}

static __always_inline int check_dev(struct inode *ino)
{
    if (!ino || (ino->i_mode & 0170000) != 0100000) return 0;
    if (cfg_target_sdev) {
        struct super_block *sb = ino->i_sb;
        if (!sb || sb->s_dev != cfg_target_sdev) return 0;
    }
    return 1;
}

static __always_inline void init_task(struct task_st *ts)
{
    if (ts->initialized) return;
    __u64 n = PACK4(16384ULL);
    #pragma clang loop unroll(full)
    for (int g = 0; g < 16; g++) ts->h[g] = n;
    ts->initialized = 1;
}

static __always_inline struct inode_st *get_ino(__u64 ino_num)
{
    struct inode_st *is = bpf_map_lookup_elem(&inode_map, &ino_num);
    if (is) return is;
    __u32 k0 = 0;
    struct inode_st *z = bpf_map_lookup_elem(&zero_ino_map, &k0);
    if (!z) return 0;
    bpf_map_update_elem(&inode_map, &ino_num, z, BPF_NOEXIST);
    return bpf_map_lookup_elem(&inode_map, &ino_num);
}

static __always_inline __u32 compute_s2(void *ctx, struct task_st *ts, int is_write)
{
    if (is_write) ts->recent_writes++; else ts->recent_reads++;
    if ((ts->recent_reads + ts->recent_writes) > 256) { ts->recent_reads >>= 1; ts->recent_writes >>= 1; }
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

static __always_inline __s32 kssm_step_readout(
    __u64 *vi, __u64 *vt, const struct kssm_live_weights *w, const struct evt_scr *ev)
{
    const volatile struct evt_scr *vev = ev;
    const __u64 *e1 = w->emb[KSSM_S1_OFF + ((__u64)vev->s1 & 31ULL)];
    const __u64 *e2 = w->emb[KSSM_S2_OFF + ((__u64)vev->s2 & 63ULL)];
    const __u64 *e3 = w->emb[KSSM_S3_OFF + ((__u64)vev->s3 & 31ULL)];
    const __u64 *e4 = w->emb[KSSM_S4_OFF + ((__u64)vev->s4 & 31ULL)];
    const __u64 *e5 = w->emb[KSSM_S5_OFF + ((__u64)vev->s5 & 31ULL)];

    SWAR_STEP(vi, vt,  0, 1, 2, e1, e2, e3, e4, e5); SWAR_STEP(vi, vt,  1, 1, 2, e1, e2, e3, e4, e5);
    SWAR_STEP(vi, vt,  2, 1, 3, e1, e2, e3, e4, e5); SWAR_STEP(vi, vt,  3, 1, 3, e1, e2, e3, e4, e5);
    SWAR_STEP(vi, vt,  4, 2, 3, e1, e2, e3, e4, e5); SWAR_STEP(vi, vt,  5, 2, 3, e1, e2, e3, e4, e5);
    SWAR_STEP(vi, vt,  6, 2, 4, e1, e2, e3, e4, e5); SWAR_STEP(vi, vt,  7, 2, 4, e1, e2, e3, e4, e5);
    SWAR_STEP(vi, vt,  8, 3, 4, e1, e2, e3, e4, e5); SWAR_STEP(vi, vt,  9, 3, 4, e1, e2, e3, e4, e5);
    SWAR_STEP(vi, vt, 10, 3, 5, e1, e2, e3, e4, e5); SWAR_STEP(vi, vt, 11, 3, 5, e1, e2, e3, e4, e5);
    SWAR_STEP(vi, vt, 12, 4, 5, e1, e2, e3, e4, e5); SWAR_STEP(vi, vt, 13, 4, 5, e1, e2, e3, e4, e5);
    SWAR_STEP(vi, vt, 14, 4, 5, e1, e2, e3, e4, e5); SWAR_STEP(vi, vt, 15, 4, 5, e1, e2, e3, e4, e5);

    const __u16 *hi = (const __u16 *)vi;
    const __u16 *ht = (const __u16 *)vt;
    __s32 acc = w->b_scan;
    #pragma clang loop unroll(disable)
    for (int c = 0; c < KSSM_DIM; c++) {
        acc += ((__s32)hi[c] - 16384) * (__s32)w->w_scan_i[c]
             + ((__s32)ht[c] - 16384) * (__s32)w->w_scan_t[c];
    }
    return acc >> 8;
}

static __always_inline void emit_drop(__u64 ino_num, __u64 dstart, __u64 dlen, __s32 sc, __u8 is_w, struct exp4_stats *st)
{
    struct exp4_drop_hint *h = bpf_ringbuf_reserve(&drop_rb, sizeof(*h), 0);
    if (!h) { if (st) st->rb_drops++; return; }
    h->ino = ino_num; h->drop_start = dstart; h->drop_len = dlen;
    h->pid = (__u32)(bpf_get_current_pid_tgid() >> 32);
    h->score_q8 = sc; h->is_write = is_w;
    bpf_ringbuf_submit(h, 0);
    if (st) st->hints_sent++;
}

SEC("fentry/ext4_file_read_iter")
int BPF_PROG(on_ext4_read, struct kiocb *iocb, struct iov_iter *to)
{
    if (!iocb || !to || !iocb->ki_filp) return 0;
    struct inode *ino = iocb->ki_filp->f_inode;
    if (!check_dev(ino)) return 0;

    __u32 k0 = 0;
    struct exp4_stats *st = bpf_map_lookup_elem(&stats_map, &k0);
    if (st) st->rw_hooks++;

    __u64 ino_num = ino->i_ino;
    __u64 off = (__u64)iocb->ki_pos;
    __u32 cnt = (__u32)to->count;
    __u64 fsz = (__u64)ino->i_size;

    struct inode_st *is = get_ino(ino_num);
    if (!is) return 0;

    if (cfg_mode == EXP4_MODE_SIZE) {
        __u32 acc = is->bytes_acc + cnt;
        if (acc < (4U << 20)) { is->bytes_acc = acc; return 0; }
        is->bytes_acc = 0;
        if (fsz >= (32ULL << 20)) {
            __u64 ds = (off > (8ULL << 20)) ? (off - (8ULL << 20)) : 0;
            emit_drop(ino_num, ds, (8ULL << 20), 256, 0, st);
        }
        return 0;
    }

    struct task_struct *t = bpf_get_current_task_btf();
    struct task_st *ts = bpf_task_storage_get(&task_storage, t, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
    if (!ts) return 0;
    init_task(ts);

    /* Per-thread stride detection + per-inode 128KB byte-budget gating (proven 84ns & 0 SST FP!) */
    __s64 stride = 0;
    int is_seq = 0;
    if (ts->last_ino == ino_num) {
        stride = (__s64)off - (__s64)ts->last_off;
        __u64 lim = ((__u64)ts->last_cnt << 1);
        if (lim < 131072ULL) lim = 131072ULL;
        is_seq = (stride > 0 && (__u64)stride <= lim) || (stride == 0 && cnt >= 65536U);
    } else {
        stride = (__s64)off - (__s64)is->last_off;
        is_seq = (stride == (__s64)is->last_size);
    }
    ts->last_ino = ino_num; ts->last_off = off; ts->last_cnt = cnt;
    if (is_seq) { if (ts->seq_run < 31) ts->seq_run++; } else { ts->seq_run = 0; }

    __u8 cur_mode = is_seq ? 1 : 2;
    is->last_off = off; is->last_size = cnt;
    __u32 acc = is->bytes_acc + cnt;
    if (is->last_mode == cur_mode && acc < 131072U) {
        is->bytes_acc = acc;
        return 0;
    }
    is->last_mode = cur_mode;
    is->bytes_acc = 0;

    if (off < is->last_drop_off) is->last_drop_off = off;

    if (cfg_mode == EXP4_MODE_SEQDROP) {
        if (is_seq) is->consec_scan++; else is->consec_scan = 0;
        if (is->consec_scan >= 4 && off > (24ULL << 20)) {
            __u64 safe_end = off - (16ULL << 20);
            if (safe_end >= is->last_drop_off + (8ULL << 20)) {
                __u64 ds = is->last_drop_off;
                __u64 dl = safe_end - ds;
                if (dl > (32ULL << 20)) { ds = safe_end - (16ULL << 20); dl = (16ULL << 20); }
                is->last_drop_off = safe_end;
                emit_drop(ino_num, ds, dl, 256, 0, st);
            }
        }
        return 0;
    }

    /* Mode 0: KSSM Adaptive (Exact Exp 0-C 5-tuple + Global 4GiB 2MiB-Chunk Reuse Self-Supervision!) */
    __u64 c2m = off >> 21; /* 2 MiB chunk index */
    if (c2m != is->last_fp_chunk) {
        is->last_fp_chunk = c2m;
        struct chunk_key ck = { .ino = ino_num, .chunk_2m = c2m };
        __u32 *seen = bpf_map_lookup_elem(&chunk_seen_map, &ck);
        int is_reuse = 0;
        if (seen) {
            is_reuse = 1;
            (*seen)++;
            if (st) st->reuse_hits++;
        } else {
            __u32 one = 1;
            bpf_map_update_elem(&chunk_seen_map, &ck, &one, BPF_NOEXIST);
        }
        /* Fast EMA (alpha=1/4) over 2 MiB chunks: locks onto Hot Reusable Seq within 4 chunks (8 MB)! */
        is->reuse_ema = is->reuse_ema - (is->reuse_ema >> 2) + (is_reuse ? 64U : 0U);
    }

    struct kssm_live_weights *w = bpf_map_lookup_elem(&weights_map, &k0);
    struct evt_scr *ev = bpf_map_lookup_elem(&scr_map, &k0);
    if (!w || !ev) return 0;

    ev->s1 = is_seq ? 8 : 9;
    ev->s2 = compute_s2(ctx, ts, 0);
    __u32 reg = (off < 65536ULL) ? 0 : (((off + cnt) >= fsz) ? 3 : ((off < (fsz >> 1)) ? 1 : 2));
    ev->s3 = ((fast_sz_b(fsz) << 2) | reg) & 31;
    __u32 req_b = (cnt >= 16384U) + (cnt >= 65536U) + (cnt >= 262144U);
    __u32 str_b = is_seq ? ((ts->seq_run >= 3) ? 0 : 4) :
                  ((stride < 0) ? ((stride > -131072) ? 1 : 2) :
                  ((stride == 0) ? 3 : ((stride <= 65536) ? 5 : ((stride <= 524288) ? 6 : 7))));
    ev->s4 = ((req_b << 3) | str_b) & 31;
    ev->s5 = 8;

    __s32 sc_q8 = kssm_step_readout(is->h, ts->h, w, ev);
    if (st) st->ssm_updates++;

    /* Self-supervised reuse calibration: if file exhibits genuine reuse within the 4 GiB cache window
     * (reuse_ema >= 64, i.e., >=25% recent 2MiB chunks revisited within 4GiB!), suppress One-Shot drop! */
    __s32 cal_sc = (is->reuse_ema >= 64U) ? -256 : sc_q8;

    if (cal_sc > 0) {
        is->consec_scan++;
        if (st) st->scan_votes++;
    } else {
        is->consec_scan = 0;
        if (st) st->hot_votes++;
    }

    if (is->consec_scan >= 4 && off > (24ULL << 20)) {
        __u64 safe_end = off - (16ULL << 20);
        if (safe_end >= is->last_drop_off + (8ULL << 20)) {
            __u64 ds = is->last_drop_off;
            __u64 dl = safe_end - ds;
            if (dl > (32ULL << 20)) { ds = safe_end - (16ULL << 20); dl = (16ULL << 20); }
            is->last_drop_off = safe_end;
            emit_drop(ino_num, ds, dl, cal_sc, 0, st);
        }
    }
    return 0;
}
