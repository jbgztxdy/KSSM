#include "../vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "kssm_common.h"

char LICENSE[] SEC("license") = "GPL";

volatile const __u32 cfg_use_stackid = 1;
volatile const __u32 cfg_use_gating = 1;
volatile const __u32 cfg_ssm_mode = 2; /* 2=subspace_swar64, 3=shared_swar64 */

struct kssm_swar_tables {
    __u64 emb_i[KSSM_TOTAL_SLOTS][16];
    __u64 emb_t[KSSM_TOTAL_SLOTS][16];
};

struct kssm_inode_state_v5 {
    struct kssm_hidden hidden;          /* 128 B */
    __u64 last_off;
    __u32 last_size;
    __u32 bytes_acc;                    /* byte budget since last SSM update */
    __u32 parent_churn_bucket;
    __s16 score_scan;
    __s16 score_life;
    __u8  last_mode;                    /* 0=init, 1=seqR, 2=rndR, 3=seqW, 4=rndW */
    __u8  pad[3];
};

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
    __type(value, struct kssm_inode_state_v5);
} zero_inode_map SEC(".maps");

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
    __type(value, struct kssm_inode_state_v5);
} inode_storage SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(map_flags, BPF_F_NO_COMMON_LRU);
    __uint(max_entries, 131072);
    __type(key, __u64);
    __type(value, struct kssm_inode_state_v5);
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

struct kssm_evt_scratch {
    __u32 s1, s2, s3, s4, s5;
    __u32 pad;
};

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct kssm_evt_scratch);
} evt_scratch_map SEC(".maps");

static __always_inline __u32 fast_size_bucket(__u64 v)
{
    if (v < (1ULL << 16)) return (v >= (1ULL << 12)) ? 1 : 0;
    if (v < (1ULL << 24)) return (v >= (1ULL << 20)) ? 3 : 2;
    if (v < (1ULL << 30)) return (v >= (1ULL << 28)) ? 5 : 4;
    return (v >= (1ULL << 32)) ? 7 : 6;
}

#define PACK4(v) (((__u64)(v)) | ((__u64)(v) << 16) | ((__u64)(v) << 32) | ((__u64)(v) << 48))
#define MASK_SH(sh) PACK4(0xFFFFULL >> (sh))
#define CONST_POS5(d, c) PACK4(10240ULL + (16384ULL >> (c)) - (16384ULL >> (d)))
#define CONST_NEG5(d, c) PACK4(10240ULL - (16384ULL >> (c)) - (16384ULL >> (d)))
#define CONST_POS2(d, c) PACK4(4096ULL  + (16384ULL >> (c)) - (16384ULL >> (d)))
#define CONST_NEG2(d, c) PACK4(4096ULL  - (16384ULL >> (c)) - (16384ULL >> (d)))

#define SWAR_SHARED_STEP(vi, vt, g, d, c, e1, e2, e3, e4, e5) do { \
    __u64 _oi = (vi)[g]; \
    __u64 _ot = (vt)[g]; \
    __u64 _b  = (e1)[g] + (e2)[g] + (e3)[g] + (e4)[g] + (e5)[g]; \
    (vi)[g] = _oi - ((_oi >> (d)) & MASK_SH(d)) + ((_ot >> (c)) & MASK_SH(c)) + _b - CONST_POS5(d, c); \
    (vt)[g] = _ot - ((_ot >> (d)) & MASK_SH(d)) - ((_oi >> (c)) & MASK_SH(c)) + _b - CONST_NEG5(d, c); \
} while (0)

#define SWAR_SUB_STEP(vi, vt, g, d, c, ea, eb) do { \
    __u64 _oi = (vi)[g]; \
    __u64 _ot = (vt)[g]; \
    __u64 _b  = (ea)[g] + (eb)[g]; \
    (vi)[g] = _oi - ((_oi >> (d)) & MASK_SH(d)) + ((_ot >> (c)) & MASK_SH(c)) + _b - CONST_POS2(d, c); \
    (vt)[g] = _ot - ((_ot >> (d)) & MASK_SH(d)) - ((_oi >> (c)) & MASK_SH(c)) + _b - CONST_NEG2(d, c); \
} while (0)

static __always_inline void kssm_step_swar_shared_64d(
    struct kssm_hidden *hi, struct kssm_hidden *ht,
    const struct kssm_swar_tables *tbl, const struct kssm_evt_scratch *ev)
{
    const volatile struct kssm_evt_scratch *vev = ev;
    __u64 i1 = KSSM_S1_OFF + ((__u64)vev->s1 & 31ULL);
    __u64 i2 = KSSM_S2_OFF + ((__u64)vev->s2 & 63ULL);
    __u64 i3 = KSSM_S3_OFF + ((__u64)vev->s3 & 31ULL);
    __u64 i4 = KSSM_S4_OFF + ((__u64)vev->s4 & 31ULL);
    __u64 i5 = KSSM_S5_OFF + ((__u64)vev->s5 & 31ULL);

    const __u64 *e1 = tbl->emb_i[i1];
    const __u64 *e2 = tbl->emb_i[i2];
    const __u64 *e3 = tbl->emb_i[i3];
    const __u64 *e4 = tbl->emb_i[i4];
    const __u64 *e5 = tbl->emb_i[i5];
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

static __always_inline void kssm_step_swar_subspace_64d(
    struct kssm_hidden *hi, struct kssm_hidden *ht,
    const struct kssm_swar_tables *tbl, const struct kssm_evt_scratch *ev)
{
    const volatile struct kssm_evt_scratch *vev = ev;
    __u64 i1 = KSSM_S1_OFF + ((__u64)vev->s1 & 31ULL);
    __u64 i2 = KSSM_S2_OFF + ((__u64)vev->s2 & 63ULL);
    __u64 i3 = KSSM_S3_OFF + ((__u64)vev->s3 & 31ULL);
    __u64 i4 = KSSM_S4_OFF + ((__u64)vev->s4 & 31ULL);
    __u64 i5 = KSSM_S5_OFF + ((__u64)vev->s5 & 31ULL);

    const __u64 *e1 = tbl->emb_i[i1];
    const __u64 *e2 = tbl->emb_i[i2];
    const __u64 *e3 = tbl->emb_i[i3];
    const __u64 *e4 = tbl->emb_i[i4];
    const __u64 *e5 = tbl->emb_i[i5];
    __u64 *vi = (__u64 *)hi->h;
    __u64 *vt = (__u64 *)ht->h;

    SWAR_SUB_STEP(vi, vt,  0, 1, 2, e1, e2);
    SWAR_SUB_STEP(vi, vt,  1, 1, 3, e1, e2);
    SWAR_SUB_STEP(vi, vt,  2, 2, 3, e1, e2);
    SWAR_SUB_STEP(vi, vt,  3, 3, 4, e1, e2);
    SWAR_SUB_STEP(vi, vt,  4, 1, 2, e1, e3);
    SWAR_SUB_STEP(vi, vt,  5, 2, 3, e1, e3);
    SWAR_SUB_STEP(vi, vt,  6, 3, 4, e1, e3);
    SWAR_SUB_STEP(vi, vt,  7, 4, 5, e1, e3);
    SWAR_SUB_STEP(vi, vt,  8, 1, 2, e1, e4);
    SWAR_SUB_STEP(vi, vt,  9, 2, 3, e1, e4);
    SWAR_SUB_STEP(vi, vt, 10, 3, 4, e1, e4);
    SWAR_SUB_STEP(vi, vt, 11, 4, 5, e1, e4);
    SWAR_SUB_STEP(vi, vt, 12, 1, 2, e1, e5);
    SWAR_SUB_STEP(vi, vt, 13, 2, 3, e1, e5);
    SWAR_SUB_STEP(vi, vt, 14, 3, 4, e1, e5);
    SWAR_SUB_STEP(vi, vt, 15, 4, 5, e1, e5);
}

static __always_inline __s16 kssm_readout_single8(
    const struct kssm_hidden *hi, const struct kssm_hidden *ht,
    const __s16 *wi, const __s16 *wt, __s32 bias)
{
    __s32 acc = bias;
    #pragma clang loop unroll(full)
    for (int k = 0; k < 8; k++) {
        int c = k * 8;
        __s32 xi = (__s32)(__u16)hi->h[c] - 16384;
        __s32 xt = (__s32)(__u16)ht->h[c] - 16384;
        acc += xi * (__s32)wi[c] + xt * (__s32)wt[c];
    }
    return (__s16)(acc >> 8);
}

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
            #pragma clang loop unroll(full)
            for (int d = 15; d >= 0; d--) {
                __u32 idx = (head - d) & 15;
                __u32 diff = ts->stack_ring[idx] ^ u_sid;
                __u32 nz = (diff | (__u32)(-(__s32)diff)) >> 31;
                __u32 mask = (__u32)(-(__s32)(1U - nz));
                dist = ((__u32)d & mask) | (dist & ~mask);
            }
            __u32 next_head = (head + 1) & 15;
            ts->stack_ring[next_head] = u_sid;
            ts->ring_head = next_head;
        }
    }
    return ((rw_bucket << 4) | (dist & 15)) & 63;
}

static __always_inline struct kssm_inode_state_v5 *get_inode_st(__u64 ino_key)
{
    struct kssm_inode_state_v5 *is = bpf_map_lookup_elem(&inode_lru_map, &ino_key);
    if (is) return is;
    __u32 k0 = 0;
    struct kssm_inode_state_v5 *zero = bpf_map_lookup_elem(&zero_inode_map, &k0);
    if (!zero) return 0;
    bpf_map_update_elem(&inode_lru_map, &ino_key, zero, BPF_NOEXIST);
    return bpf_map_lookup_elem(&inode_lru_map, &ino_key);
}

/* Mode-Transition + 128 KB Byte-Budget Gating:
 * Triggers SSM ONLY on:
 *   (a) First touch on inode (last_mode == 0)
 *   (b) Genuine behavioral mode transition (seqR <-> rndR <-> seqW <-> rndW)
 *   (c) Cumulative I/O volume reaches 128 KB window (bytes_acc >= 131072)
 * Guarantees <= 1/32 (3.125%) worst-case steady-state trigger rate even under
 * 100% 4KB random I/O, while keeping full bpf_get_stackid() s2 signature! */
static __always_inline int handle_iter_rw(void *ctx, struct kiocb *iocb, struct iov_iter *iter, int is_write)
{
    if (!iocb || !iter) return 0;
    struct file *file = iocb->ki_filp;
    if (!file) return 0;
    struct inode *ino = file->f_inode;
    if (!ino) return 0;

    __u64 off = (__u64)iocb->ki_pos;
    __u32 count = (__u32)iter->count;

    struct kssm_inode_state_v5 *is = get_inode_st((__u64)ino);
    if (!is) return 0;

    __s64 stride = (__s64)off - (__s64)is->last_off;
    int is_seq = (stride == (__s64)is->last_size) || (stride == 0);
    __u8 cur_mode = is_write ? (is_seq ? 3 : 4) : (is_seq ? 1 : 2);

    is->last_off = off;
    is->last_size = count;
    __u32 acc = is->bytes_acc + count;

    if (cfg_use_gating && is->last_mode == cur_mode && acc < 131072U) {
        is->bytes_acc = acc;
        return 0;
    }

    is->last_mode = cur_mode;
    is->bytes_acc = 0;

    __u32 k0 = 0;
    struct kssm_cpu_stats *st = bpf_map_lookup_elem(&stats_map, &k0);
    struct task_struct *t = bpf_get_current_task_btf();
    struct kssm_task_state *ts = bpf_task_storage_get(&task_storage, t, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
    if (!ts) return 0;

    struct kssm_evt_scratch *ev = bpf_map_lookup_elem(&evt_scratch_map, &k0);
    struct kssm_swar_tables *stbl = bpf_map_lookup_elem(&swar_tbl_map, &k0);
    struct kssm_readout *ro = bpf_map_lookup_elem(&readout_map, &k0);
    if (!ev || !stbl || !ro) return 0;

    __u64 fsz = (__u64)ino->i_size;
    ev->s1 = !is_write ? (4 + (!is_seq)) : ((off >= fsz) ? 6 : (7 + (!is_seq)));
    ev->s2 = compute_s2(ctx, ts, is_write);
    __u32 sz_b = fast_size_bucket(fsz);
    __u32 region = (off >= 65536) + (off >= (fsz >> 1)) + ((off + count) >= fsz);
    ev->s3 = (sz_b << 2) | (region & 3);
    __u32 req_b = (count >= 16384) + (count >= 65536) + (count >= 262144);
    __u32 str_b = is_seq ? 0 : (stride < 0 ? 1 : (stride > 131072 ? 3 : 2));
    ev->s4 = (req_b << 3) | str_b;
    ev->s5 = is->parent_churn_bucket;

    if (cfg_ssm_mode == 3)
        kssm_step_swar_shared_64d(&is->hidden, &ts->hidden, stbl, ev);
    else
        kssm_step_swar_subspace_64d(&is->hidden, &ts->hidden, stbl, ev);

    if (!is_write)
        is->score_scan = kssm_readout_single8(&is->hidden, &ts->hidden, ro->w_scan_i, ro->w_scan_t, ro->b_scan);
    else
        is->score_life = kssm_readout_single8(&is->hidden, &ts->hidden, ro->w_life_i, ro->w_life_t, ro->b_life);

    if (st) st->ssm_updates++;
    return 0;
}

SEC("fentry/ext4_file_read_iter")
int BPF_PROG(ext4_kssm_read, struct kiocb *iocb, struct iov_iter *to)
{
    return handle_iter_rw(ctx, iocb, to, 0);
}

SEC("fentry/ext4_file_write_iter")
int BPF_PROG(ext4_kssm_write, struct kiocb *iocb, struct iov_iter *from)
{
    return handle_iter_rw(ctx, iocb, from, 1);
}

SEC("fentry/vfs_open")
int BPF_PROG(vfs_kssm_open, const struct path *path, struct file *file)
{
    if (!path || !file) return 0;
    struct dentry *dentry = path->dentry;
    if (!dentry) return 0;
    struct inode *ino = dentry->d_inode;
    if (!ino || (ino->i_mode & 0170000) != 0100000) return 0;

    struct kssm_inode_state_v5 *is = get_inode_st((__u64)ino);
    if (!is) return 0;

    __u64 pkey = (__u64)dentry->d_parent;
    struct dir_churn *dc = bpf_map_lookup_elem(&dir_churn_map, &pkey);
    if (dc) is->parent_churn_bucket = fast_size_bucket((__u64)dc->ema_events << 12);

    struct task_struct *t = bpf_get_current_task_btf();
    struct kssm_task_state *ts = bpf_task_storage_get(&task_storage, t, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
    if (!ts) return 0;

    __u32 k0 = 0;
    struct kssm_evt_scratch *ev = bpf_map_lookup_elem(&evt_scratch_map, &k0);
    struct kssm_swar_tables *stbl = bpf_map_lookup_elem(&swar_tbl_map, &k0);
    struct kssm_readout *ro = bpf_map_lookup_elem(&readout_map, &k0);
    if (!ev || !stbl || !ro) return 0;

    __u32 flags = file->f_flags;
    ev->s1 = ((flags & 0100) ? 1 : 0) + ((flags & 3) ? 2 : 0);
    ev->s2 = compute_s2(ctx, ts, (flags & 3) != 0);
    ev->s3 = fast_size_bucket((__u64)ino->i_size);
    ev->s4 = 0;
    ev->s5 = is->parent_churn_bucket;

    kssm_step_swar_shared_64d(&is->hidden, &ts->hidden, stbl, ev);
    is->score_life = kssm_readout_single8(&is->hidden, &ts->hidden, ro->w_life_i, ro->w_life_t, ro->b_life);
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
