#include "../vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "kssm_write_shared.h"

char LICENSE[] SEC("license") = "GPL";

volatile const __u32 cfg_target_sdev = 0;

struct kssm_task_w {
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

struct kssm_inode_w {
    __u64 h[16];          /* current hidden state */
    __u64 birth_hi[16];   /* snapshot at 0th second for slow-loop feedback */
    __u64 birth_ht[16];   /* snapshot at 0th second for slow-loop feedback */
    __u64 birth_ts_ns;
    __u64 last_off;
    __u32 last_size;
    __u32 bytes_acc;
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
    __type(value, struct kssm_life_base);
} base_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct kssm_life_head);
} head_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct kssm_inode_w);
} zero_ino_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_TASK_STORAGE);
    __uint(map_flags, BPF_F_NO_PREALLOC);
    __type(key, int);
    __type(value, struct kssm_task_w);
} task_storage SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(map_flags, BPF_F_NO_COMMON_LRU);
    __uint(max_entries, 65536);
    __type(key, __u64);
    __type(value, struct kssm_inode_w);
} inode_map SEC(".maps");

struct dir_st {
    __u64 last_ts;
    __u32 ema;
    __u32 pad;
};

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
} msg_rb SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct kssm_write_stats);
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

static __always_inline void init_task_w(struct kssm_task_w *ts)
{
    if (ts->initialized) return;
    __u64 n = PACK4(16384ULL);
    #pragma clang loop unroll(full)
    for (int g = 0; g < 16; g++) ts->h[g] = n;
    ts->initialized = 1;
}

static __always_inline struct kssm_inode_w *get_ino_w(__u64 ino_num)
{
    struct kssm_inode_w *is = bpf_map_lookup_elem(&inode_map, &ino_num);
    if (is) return is;
    __u32 k0 = 0;
    struct kssm_inode_w *z = bpf_map_lookup_elem(&zero_ino_map, &k0);
    if (!z) return 0;
    bpf_map_update_elem(&inode_map, &ino_num, z, BPF_NOEXIST);
    return bpf_map_lookup_elem(&inode_map, &ino_num);
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

static __always_inline __u32 compute_s2(void *ctx, struct kssm_task_w *ts, int is_write)
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

static __always_inline void ssm_update_64d(__u64 *vi, __u64 *vt, const struct kssm_life_base *base, const struct evt_scr *ev)
{
    const volatile struct evt_scr *vev = ev;
    const __u64 *e1 = base->emb[KSSM_S1_OFF + ((__u64)vev->s1 & 31ULL)];
    const __u64 *e2 = base->emb[KSSM_S2_OFF + ((__u64)vev->s2 & 63ULL)];
    const __u64 *e3 = base->emb[KSSM_S3_OFF + ((__u64)vev->s3 & 31ULL)];
    const __u64 *e4 = base->emb[KSSM_S4_OFF + ((__u64)vev->s4 & 31ULL)];
    const __u64 *e5 = base->emb[KSSM_S5_OFF + ((__u64)vev->s5 & 31ULL)];

    SWAR_STEP(vi, vt,  0, 1, 2, e1, e2, e3, e4, e5); SWAR_STEP(vi, vt,  1, 1, 2, e1, e2, e3, e4, e5);
    SWAR_STEP(vi, vt,  2, 1, 3, e1, e2, e3, e4, e5); SWAR_STEP(vi, vt,  3, 1, 3, e1, e2, e3, e4, e5);
    SWAR_STEP(vi, vt,  4, 2, 3, e1, e2, e3, e4, e5); SWAR_STEP(vi, vt,  5, 2, 3, e1, e2, e3, e4, e5);
    SWAR_STEP(vi, vt,  6, 2, 4, e1, e2, e3, e4, e5); SWAR_STEP(vi, vt,  7, 2, 4, e1, e2, e3, e4, e5);
    SWAR_STEP(vi, vt,  8, 3, 4, e1, e2, e3, e4, e5); SWAR_STEP(vi, vt,  9, 3, 4, e1, e2, e3, e4, e5);
    SWAR_STEP(vi, vt, 10, 3, 5, e1, e2, e3, e4, e5); SWAR_STEP(vi, vt, 11, 3, 5, e1, e2, e3, e4, e5);
    SWAR_STEP(vi, vt, 12, 4, 5, e1, e2, e3, e4, e5); SWAR_STEP(vi, vt, 13, 4, 5, e1, e2, e3, e4, e5);
    SWAR_STEP(vi, vt, 14, 4, 5, e1, e2, e3, e4, e5); SWAR_STEP(vi, vt, 15, 4, 5, e1, e2, e3, e4, e5);
}

static __always_inline __u8 predict_life_class(const __u64 *vi, const __u64 *vt, const struct kssm_life_head *hd)
{
    const __u16 *hi = (const __u16 *)vi;
    const __u16 *ht = (const __u16 *)vt;
    __s32 s0 = hd->b[0], s1 = hd->b[1], s2 = hd->b[2];
    #pragma clang loop unroll(disable)
    for (int c = 0; c < KSSM_DIM; c++) {
        __s32 xi = (__s32)hi[c] - 16384;
        __s32 xt = (__s32)ht[c] - 16384;
        s0 += xi * (__s32)hd->w_i[0][c] + xt * (__s32)hd->w_t[0][c];
        s1 += xi * (__s32)hd->w_i[1][c] + xt * (__s32)hd->w_t[1][c];
        s2 += xi * (__s32)hd->w_i[2][c] + xt * (__s32)hd->w_t[2][c];
    }
    if (s2 >= s1 && s2 >= s0) return 2;
    if (s1 >= s0) return 1;
    return 0;
}

/* 0th-Second Birth Prediction Hook: Predicts Short/Med/Long stream at open(O_CREAT) byte 0! */
SEC("fentry/vfs_open")
int BPF_PROG(on_vfs_open, const struct path *path, struct file *file)
{
    if (!path || !file || !path->dentry) return 0;
    struct inode *ino = path->dentry->d_inode;
    if (!check_dev(ino)) return 0;

    __u32 k0 = 0;
    struct kssm_write_stats *st = bpf_map_lookup_elem(&stats_map, &k0);
    if (st) st->total_hooks++;

    __u32 fl = file->f_flags;
    int is_creat = ((fl & 0100) != 0) && (ino->i_size == 0);
    __u64 pino = (path->dentry->d_parent && path->dentry->d_parent->d_inode) ? path->dentry->d_parent->d_inode->i_ino : 0;
    __u32 churn = bump_dir_churn(pino, is_creat);

    struct kssm_inode_w *is = get_ino_w(ino->i_ino);
    struct task_struct *t = bpf_get_current_task_btf();
    struct kssm_task_w *ts = bpf_task_storage_get(&task_storage, t, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
    struct kssm_life_base *base = bpf_map_lookup_elem(&base_map, &k0);
    struct kssm_life_head *head = bpf_map_lookup_elem(&head_map, &k0);
    struct evt_scr *ev = bpf_map_lookup_elem(&scr_map, &k0);
    if (!is || !ts || !base || !head || !ev) return 0;
    init_task_w(ts);

    __u32 s1 = is_creat ? (1 + ((fl & 02000) ? 1 : 0) + ((fl & 01000) ? 2 : 0)) : (((fl & 3) != 0) ? 5 : 0);
    ev->s1 = s1;
    ev->s2 = compute_s2(ctx, ts, (fl & 3) != 0);
    ev->s3 = (fast_sz_b((__u64)ino->i_size) << 2) & 31;
    ev->s4 = 0;
    ev->s5 = (((churn > 4) ? 16 : 0) + (fast_sz_b((__u64)churn << 12) << 1)) & 31;

    ssm_update_64d(is->h, ts->h, base, ev);
    if (st) st->ssm_updates++;

    if (is_creat) {
        __u8 pred = predict_life_class(is->h, ts->h, head);
        is->pred_class = pred;
        is->birth_ts_ns = bpf_ktime_get_ns();
        is->has_birth = 1;
        /* Snapshot birth state for delayed zero-BPTT slow-loop calibration upon unlink! */
        #pragma clang loop unroll(full)
        for (int g = 0; g < 16; g++) {
            is->birth_hi[g] = is->h[g];
            is->birth_ht[g] = ts->h[g];
        }
        if (st) st->births_predicted++;

        struct kssm_write_msg *m = bpf_ringbuf_reserve(&msg_rb, sizeof(*m), 0);
        if (m) {
            m->ino = ino->i_ino;
            m->ts_ns = is->birth_ts_ns;
            m->lifetime_ns = 0;
            m->pid = (__u32)(bpf_get_current_pid_tgid() >> 32);
            m->msg_type = MSG_BIRTH_HINT;
            m->pred_class = pred;
            m->true_class = 0;
            bpf_ringbuf_submit(m, 0);
        } else if (st) {
            st->rb_drops++;
        }
    }
    return 0;
}

/* Delayed Self-Supervised Death Ground-Truth Hook -> Feeds Zero-BPTT Slow Loop! */
SEC("fentry/vfs_unlink")
int BPF_PROG(on_vfs_unlink, struct user_namespace *mnt_userns, struct inode *dir, struct dentry *dentry)
{
    if (!dir || !dentry || !dentry->d_inode) return 0;
    struct inode *ino = dentry->d_inode;
    if (!check_dev(ino)) return 0;

    bump_dir_churn(dir->i_ino, 1);
    __u64 ino_num = ino->i_ino;
    struct kssm_inode_w *is = bpf_map_lookup_elem(&inode_map, &ino_num);
    if (is && is->has_birth) {
        __u32 k0 = 0;
        struct kssm_write_stats *st = bpf_map_lookup_elem(&stats_map, &k0);
        __u64 now = bpf_ktime_get_ns();
        __u64 dt = (now > is->birth_ts_ns) ? (now - is->birth_ts_ns) : 0;
        __u8 tcls = (dt < 10000000000ULL) ? 0 : ((dt < 50000000000ULL) ? 1 : 2);
        if (st) st->deaths_observed++;

        struct kssm_write_msg *m = bpf_ringbuf_reserve(&msg_rb, sizeof(*m), 0);
        if (m) {
            m->ino = ino_num;
            m->ts_ns = now;
            m->lifetime_ns = dt;
            m->pid = (__u32)(bpf_get_current_pid_tgid() >> 32);
            m->msg_type = MSG_DEATH_LABEL;
            m->pred_class = is->pred_class;
            m->true_class = tcls;
            const __u16 *bhi = (const __u16 *)is->birth_hi;
            const __u16 *bht = (const __u16 *)is->birth_ht;
            #pragma clang loop unroll(disable)
            for (int c = 0; c < KSSM_DIM; c++) {
                m->snap_hi[c] = (__s16)((__s32)bhi[c] - 16384);
                m->snap_ht[c] = (__s16)((__s32)bht[c] - 16384);
            }
            bpf_ringbuf_submit(m, 0);
        } else if (st) {
            st->rb_drops++;
        }
    }
    bpf_map_delete_elem(&inode_map, &ino_num);
    return 0;
}

static __always_inline int on_rw(void *ctx, struct kiocb *iocb, struct iov_iter *iter, int is_write)
{
    if (!iocb || !iter || !iocb->ki_filp) return 0;
    struct inode *ino = iocb->ki_filp->f_inode;
    if (!check_dev(ino)) return 0;

    __u32 k0 = 0;
    struct kssm_write_stats *st = bpf_map_lookup_elem(&stats_map, &k0);
    if (st) st->total_hooks++;

    __u64 ino_num = ino->i_ino;
    __u64 off = (__u64)iocb->ki_pos;
    __u32 cnt = (__u32)iter->count;
    struct kssm_inode_w *is = get_ino_w(ino_num);
    if (!is) return 0;

    __s64 stride = (__s64)off - (__s64)is->last_off;
    int is_seq = (stride == (__s64)is->last_size) || (stride == 0);
    __u8 mode = is_write ? (is_seq ? 3 : 4) : (is_seq ? 1 : 2);
    is->last_off = off; is->last_size = cnt;
    __u32 acc = is->bytes_acc + cnt;
    if (is->last_mode == mode && acc < 131072U) {
        is->bytes_acc = acc;
        return 0;
    }
    is->last_mode = mode; is->bytes_acc = 0;

    struct task_struct *t = bpf_get_current_task_btf();
    struct kssm_task_w *ts = bpf_task_storage_get(&task_storage, t, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
    struct kssm_life_base *base = bpf_map_lookup_elem(&base_map, &k0);
    struct evt_scr *ev = bpf_map_lookup_elem(&scr_map, &k0);
    if (!ts || !base || !ev) return 0;
    init_task_w(ts);

    __u64 fsz = (__u64)ino->i_size;
    ev->s1 = !is_write ? (is_seq ? 8 : 9) : ((off >= fsz) ? 10 : (is_seq ? 11 : 12));
    ev->s2 = compute_s2(ctx, ts, is_write);
    ev->s3 = ((fast_sz_b(fsz) << 2) | ((off < 65536ULL) ? 0 : (((off + cnt) >= fsz) ? 3 : 1))) & 31;
    ev->s4 = (((cnt >= 16384U) + (cnt >= 65536U) + (cnt >= 262144U)) << 3) | (is_seq ? 0 : 2);
    ev->s5 = 8;
    ssm_update_64d(is->h, ts->h, base, ev);
    if (st) st->ssm_updates++;
    return 0;
}

SEC("fentry/ext4_file_read_iter")
int BPF_PROG(on_ext4_read, struct kiocb *iocb, struct iov_iter *to) { return on_rw(ctx, iocb, to, 0); }

SEC("fentry/ext4_file_write_iter")
int BPF_PROG(on_ext4_write, struct kiocb *iocb, struct iov_iter *from) { return on_rw(ctx, iocb, from, 1); }
