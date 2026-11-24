#include "../vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "kssm_exp0c_shared.h"

char LICENSE[] SEC("license") = "GPL";

volatile const __u32 cfg_target_sdev = 0;   /* kernel MKDEV of /mnt/nvme12 */
volatile const __u32 cfg_mode = MODE_KSSM;  /* 0=KSSM, 1=Heuristic(size>=32MB) */
volatile const __s32 cfg_scan_thresh_q8 = 0; /* Q7.8 threshold (0 = logit > 0 -> prob > 0.5) */

struct kssm_task_st {
    __u64 h[16];          /* 128 B: 64D Excess-16384 Q5.8 hidden state */
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

struct kssm_inode_st {
    __u64 h[16];          /* 128 B: 64D Excess-16384 Q5.8 hidden state */
    __u64 last_off;
    __u64 last_drop_off;  /* end offset already advised for drop-behind */
    __u32 last_size;
    __u32 bytes_acc;      /* byte budget for 128KB gating */
    __u32 consec_scan;    /* consecutive positive One-Shot Scan windows */
    __u32 parent_churn;
    __u8  last_mode;      /* 0=init, 1=seqR, 2=rndR, 3=seqW, 4=rndW */
    __u8  initialized;
    __u8  pad[6];
};

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
    __type(value, struct kssm_inode_st);
} zero_inode_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_TASK_STORAGE);
    __uint(map_flags, BPF_F_NO_PREALLOC);
    __type(key, int);
    __type(value, struct kssm_task_st);
} task_storage SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(map_flags, BPF_F_NO_COMMON_LRU);
    __uint(max_entries, 65536);
    __type(key, __u64);
    __type(value, struct kssm_inode_st);
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
    __type(value, struct kssm_exp0c_stats);
} stats_map SEC(".maps");

struct evt_scratch {
    __u32 s1, s2, s3, s4, s5, pad;
};

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct evt_scratch);
} scratch_map SEC(".maps");

#define PACK4(v) (((__u64)(v)) | ((__u64)(v) << 16) | ((__u64)(v) << 32) | ((__u64)(v) << 48))
#define MASK_SH(sh) PACK4(0xFFFFULL >> (sh))
#define CONST_POS5(d, c) PACK4(10240ULL + (16384ULL >> (c)) - (16384ULL >> (d)))
#define CONST_NEG5(d, c) PACK4(10240ULL - (16384ULL >> (c)) - (16384ULL >> (d)))

#define SWAR_SHARED_STEP(vi, vt, g, d, c, e1, e2, e3, e4, e5) do { \
    __u64 _oi = (vi)[g]; \
    __u64 _ot = (vt)[g]; \
    __u64 _b  = (e1)[g] + (e2)[g] + (e3)[g] + (e4)[g] + (e5)[g]; \
    (vi)[g] = _oi - ((_oi >> (d)) & MASK_SH(d)) + ((_ot >> (c)) & MASK_SH(c)) + _b - CONST_POS5(d, c); \
    (vt)[g] = _ot - ((_ot >> (d)) & MASK_SH(d)) - ((_oi >> (c)) & MASK_SH(c)) + _b - CONST_NEG5(d, c); \
} while (0)

static __always_inline __u32 fast_size_bucket(__u64 v)
{
    if (v < (1ULL << 16)) return (v >= (1ULL << 12)) ? 1 : 0;
    if (v < (1ULL << 24)) return (v >= (1ULL << 20)) ? 3 : 2;
    if (v < (1ULL << 30)) return (v >= (1ULL << 28)) ? 5 : 4;
    return (v >= (1ULL << 32)) ? 7 : 6;
}

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

static __always_inline void init_task_neutral(struct kssm_task_st *ts)
{
    if (ts->initialized) return;
    __u64 neut = PACK4(16384ULL);
    #pragma clang loop unroll(full)
    for (int g = 0; g < 16; g++)
        ts->h[g] = neut;
    ts->initialized = 1;
}

static __always_inline struct kssm_inode_st *get_inode_st(__u64 ino_num)
{
    struct kssm_inode_st *is = bpf_map_lookup_elem(&inode_map, &ino_num);
    if (is) return is;
    __u32 k0 = 0;
    struct kssm_inode_st *z = bpf_map_lookup_elem(&zero_inode_map, &k0);
    if (!z) return 0;
    bpf_map_update_elem(&inode_map, &ino_num, z, BPF_NOEXIST);
    return bpf_map_lookup_elem(&inode_map, &ino_num);
}

static __always_inline __u32 compute_s2(void *ctx, struct kssm_task_st *ts, int is_write)
{
    if (is_write) ts->recent_writes++;
    else ts->recent_reads++;
    if ((ts->recent_reads + ts->recent_writes) > 256) {
        ts->recent_reads >>= 1;
        ts->recent_writes >>= 1;
    }
    __u32 total = ts->recent_reads + ts->recent_writes + 1;
    __u32 rw_q = ((ts->recent_writes * 4) / total) & 3;
    __u32 dist = 15;
    long sid = bpf_get_stackid(ctx, &stack_map, BPF_F_USER_STACK | BPF_F_FAST_STACK_CMP);
    if (sid < 0)
        sid = bpf_get_stackid(ctx, &stack_map, BPF_F_FAST_STACK_CMP);
    if (sid >= 0) {
        __u32 usid = (__u32)sid;
        __u32 head = ts->ring_head & 15;
        #pragma clang loop unroll(full)
        for (int d = 15; d >= 0; d--) {
            __u32 idx = (head - d) & 15;
            __u32 diff = ts->stack_ring[idx] ^ usid;
            __u32 nz = (diff | (__u32)(-(__s32)diff)) >> 31;
            __u32 mask = (__u32)(-(__s32)(1U - nz));
            dist = ((__u32)d & mask) | (dist & ~mask);
        }
        __u32 nh = (head + 1) & 15;
        ts->stack_ring[nh] = usid;
        ts->ring_head = nh;
    }
    return ((rw_q << 4) | (dist & 15)) & 63;
}

static __always_inline void kssm_step_and_readout(
    __u64 *vi, __u64 *vt,
    const struct kssm_live_weights *w,
    const struct evt_scratch *ev,
    __s32 *out_score_q8)
{
    const volatile struct evt_scratch *vev = ev;
    __u64 i1 = KSSM_S1_OFF + ((__u64)vev->s1 & 31ULL);
    __u64 i2 = KSSM_S2_OFF + ((__u64)vev->s2 & 63ULL);
    __u64 i3 = KSSM_S3_OFF + ((__u64)vev->s3 & 31ULL);
    __u64 i4 = KSSM_S4_OFF + ((__u64)vev->s4 & 31ULL);
    __u64 i5 = KSSM_S5_OFF + ((__u64)vev->s5 & 31ULL);

    const __u64 *e1 = w->emb[i1];
    const __u64 *e2 = w->emb[i2];
    const __u64 *e3 = w->emb[i3];
    const __u64 *e4 = w->emb[i4];
    const __u64 *e5 = w->emb[i5];

    /* Bit-exact Guard-SWAR-Shared 64D matching Exp 0-B trained weights! */
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

    /* Exact full 64D + 64D linear scan readout (only runs once per 128KB window -> ~3ns amortized!) */
    const __u16 *hi16 = (const __u16 *)vi;
    const __u16 *ht16 = (const __u16 *)vt;
    __s32 acc = w->b_scan;
    #pragma clang loop unroll(disable)
    for (int c = 0; c < KSSM_DIM; c++) {
        __s32 xi = (__s32)hi16[c] - 16384;
        __s32 xt = (__s32)ht16[c] - 16384;
        acc += xi * (__s32)w->w_scan_i[c] + xt * (__s32)w->w_scan_t[c];
    }
    *out_score_q8 = acc >> 8; /* Q7.8 logit */
}

static __always_inline void emit_drop_hint(
    __u64 ino_num, __u64 drop_start, __u64 drop_len, __u64 fsz,
    __s32 score_q8, __u8 is_write, struct kssm_exp0c_stats *st)
{
    struct kssm_drop_hint *h = bpf_ringbuf_reserve(&drop_rb, sizeof(*h), 0);
    if (!h) {
        if (st) st->hints_dropped++;
        return;
    }
    h->ino = ino_num;
    h->drop_start = drop_start;
    h->drop_len = drop_len;
    h->file_size = fsz;
    h->pid = (__u32)(bpf_get_current_pid_tgid() >> 32);
    h->score_q8 = score_q8;
    h->is_write = is_write;
    bpf_ringbuf_submit(h, 0);
    if (st) st->hints_emitted++;
}

static __always_inline int handle_rw(void *ctx, struct kiocb *iocb, struct iov_iter *iter, int is_write)
{
    if (!iocb || !iter) return 0;
    struct file *file = iocb->ki_filp;
    if (!file) return 0;
    struct inode *ino = file->f_inode;
    if (!check_dev(ino)) return 0;

    __u32 k0 = 0;
    struct kssm_exp0c_stats *st = bpf_map_lookup_elem(&stats_map, &k0);
    if (st) st->total_rw_hooks++;

    __u64 ino_num = ino->i_ino;
    __u64 off = (__u64)iocb->ki_pos;
    __u32 cnt = (__u32)iter->count;
    __u64 fsz = (__u64)ino->i_size;

    struct kssm_inode_st *is = get_inode_st(ino_num);
    if (!is) return 0;

    /* Baseline comparison mode: naive static size rule (>= 32MB -> drop-behind!) */
    if (cfg_mode == MODE_HEURISTIC_SIZE) {
        __u32 acc = is->bytes_acc + cnt;
        if (acc < (4U * 1024U * 1024U)) {
            is->bytes_acc = acc;
            return 0;
        }
        is->bytes_acc = 0;
        if (fsz >= (32ULL * 1024ULL * 1024ULL)) {
            /* Blindly drops cache of ANY large file >= 32MB (including 64MB RocksDB SSTs!) */
            __u64 dstart = (off > (8ULL << 20)) ? (off - (8ULL << 20)) : 0;
            emit_drop_hint(ino_num, dstart, (8ULL << 20), fsz, 256, (__u8)is_write, st);
        }
        return 0;
    }

    /* KSSM Mode: Mode-Transition + 128 KB Byte-Budget Gating + Per-Thread Stride */
    struct task_struct *t = bpf_get_current_task_btf();
    struct kssm_task_st *ts = bpf_task_storage_get(&task_storage, t, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
    if (!ts) return 0;
    init_task_neutral(ts);

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
    ts->last_ino = ino_num;
    ts->last_off = off;
    ts->last_cnt = cnt;
    if (is_seq) {
        if (ts->seq_run < 31) ts->seq_run++;
    } else {
        ts->seq_run = 0;
    }

    __u8 cur_mode = is_write ? (is_seq ? 3 : 4) : (is_seq ? 1 : 2);
    is->last_off = off;
    is->last_size = cnt;
    __u32 acc = is->bytes_acc + cnt;

    if (is->last_mode == cur_mode && acc < 131072U) {
        is->bytes_acc = acc;
        return 0;
    }
    is->last_mode = cur_mode;
    is->bytes_acc = 0;

    struct kssm_live_weights *w = bpf_map_lookup_elem(&weights_map, &k0);
    struct evt_scratch *ev = bpf_map_lookup_elem(&scratch_map, &k0);
    if (!w || !ev) return 0;

    /* Construct exact 5-tuple grammar matching Exp 0-B prepare_dataset_v3.py! */
    __u32 s1 = !is_write ? (is_seq ? 8 : 9) : ((off >= fsz) ? 10 : (is_seq ? 11 : 12));
    __u32 s2 = compute_s2(ctx, ts, is_write);
    __u32 sz_b = fast_size_bucket(fsz);
    __u32 reg = (off < 65536ULL) ? 0 : (((off + cnt) >= fsz) ? 3 : ((off < (fsz >> 1)) ? 1 : 2));
    __u32 s3 = ((sz_b << 2) | reg) & 31;
    __u32 req_b = (cnt >= 16384U) + (cnt >= 65536U) + (cnt >= 262144U);
    __u32 str_b = is_seq ? ((ts->seq_run >= 3) ? 0 : 4) :
                  ((stride < 0) ? ((stride > -131072) ? 1 : 2) :
                  ((stride == 0) ? 3 : ((stride <= 65536) ? 5 : ((stride <= 524288) ? 6 : 7))));
    __u32 s4 = ((req_b << 3) | str_b) & 31;
    __u32 s5 = 8; /* neutral relative churn for read/write streams */

    ev->s1 = s1; ev->s2 = s2; ev->s3 = s3; ev->s4 = s4; ev->s5 = s5;

    __s32 score_q8 = 0;
    kssm_step_and_readout(is->h, ts->h, w, ev, &score_q8);
    if (st) st->ssm_updates++;

    if (!is_write) {
        /* Read path: KSSM decides One-Shot Scan vs Hot Reusable! */
        if (score_q8 > cfg_scan_thresh_q8) {
            is->consec_scan++;
            if (st) st->scan_pos_votes++;
        } else {
            is->consec_scan = 0;
            if (st) st->hot_neg_votes++;
        }
        /* Trigger drop-behind ONLY when KSSM sustains One-Shot Scan (>=4 windows = 512KB)
         * and leave a 16 MB safety margin behind current read offset so Linux readahead stays 100% full speed! */
        if (off < is->last_drop_off) is->last_drop_off = off;
        if (is->consec_scan >= 4 && off > (24ULL << 20)) {
            __u64 safe_end = off - (16ULL << 20);
            if (safe_end >= is->last_drop_off + (8ULL << 20)) {
                __u64 dstart = is->last_drop_off;
                __u64 dlen = safe_end - dstart;
                if (dlen > (32ULL << 20)) {
                    dstart = safe_end - (16ULL << 20);
                    dlen = (16ULL << 20);
                }
                is->last_drop_off = safe_end;
                emit_drop_hint(ino_num, dstart, dlen, fsz, score_q8, 0, st);
            }
        }
    } else {
        /* Write path: streaming sequential append (e.g. large Checkpoint dump) -> async writeback + drop-behind */
        if (is_seq && s1 == 10 && off > (24ULL << 20)) {
            __u64 safe_end = off - (16ULL << 20);
            if (safe_end >= is->last_drop_off + (8ULL << 20)) {
                __u64 dstart = is->last_drop_off;
                __u64 dlen = safe_end - dstart;
                if (dlen > (32ULL << 20)) {
                    dstart = safe_end - (16ULL << 20);
                    dlen = (16ULL << 20);
                }
                is->last_drop_off = safe_end;
                emit_drop_hint(ino_num, dstart, dlen, fsz, score_q8, 1, st);
            }
        }
    }
    return 0;
}

SEC("fentry/ext4_file_read_iter")
int BPF_PROG(on_ext4_read, struct kiocb *iocb, struct iov_iter *to)
{
    return handle_rw(ctx, iocb, to, 0);
}

SEC("fentry/ext4_file_write_iter")
int BPF_PROG(on_ext4_write, struct kiocb *iocb, struct iov_iter *from)
{
    return handle_rw(ctx, iocb, from, 1);
}

SEC("fentry/vfs_unlink")
int BPF_PROG(on_vfs_unlink, struct user_namespace *mnt_userns, struct inode *dir, struct dentry *dentry)
{
    if (!dentry || !dentry->d_inode) return 0;
    if (!check_dev(dentry->d_inode)) return 0;
    __u64 ino_num = dentry->d_inode->i_ino;
    bpf_map_delete_elem(&inode_map, &ino_num);
    return 0;
}
