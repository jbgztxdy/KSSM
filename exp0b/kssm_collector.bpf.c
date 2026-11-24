#include "../vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "trace_event.h"

char LICENSE[] SEC("license") = "GPL";

volatile const __u32 target_sdev = 0;
volatile const __u32 filter_pid = 0;

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 32 * 1024 * 1024);
} rb SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_STACK_TRACE);
    __uint(key_size, sizeof(__u32));
    __uint(value_size, 16 * sizeof(__u64));
    __uint(max_entries, 32768);
} stack_map SEC(".maps");

struct ino_track {
    __u64 last_off;
    __u32 last_size;
    __u32 bytes_acc;
    __u32 win_pages;
    __u32 win_reused;
    __u8  last_mode;
    __u8  pad[7];
};

struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(map_flags, BPF_F_NO_COMMON_LRU);
    __uint(max_entries, 131072);
    __type(key, __u64);
    __type(value, struct ino_track);
} ino_map SEC(".maps");

struct page_key {
    __u64 ino;
    __u64 pgoff;
};

struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(map_flags, BPF_F_NO_COMMON_LRU);
    __uint(max_entries, 524288);
    __type(key, struct page_key);
    __type(value, __u32);
} page_seen_map SEC(".maps");

struct dir_stat {
    __u64 last_ts;
    __u32 ema;
    __u32 pad;
};

struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(map_flags, BPF_F_NO_COMMON_LRU);
    __uint(max_entries, 32768);
    __type(key, __u64);
    __type(value, struct dir_stat);
} dir_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 2);
    __type(key, __u32);
    __type(value, __u64);
} cnt_map SEC(".maps");

static __always_inline void inc_cnt(__u32 idx)
{
    __u64 *v = bpf_map_lookup_elem(&cnt_map, &idx);
    if (v) (*v)++;
}

static __always_inline int check_dev(struct inode *ino)
{
    if (!ino) return 0;
    if ((ino->i_mode & 0170000) != 0100000) return 0;
    if (target_sdev) {
        struct super_block *sb = ino->i_sb;
        if (!sb || sb->s_dev != target_sdev) return 0;
    }
    if (filter_pid) {
        __u32 pid = bpf_get_current_pid_tgid() >> 32;
        if (pid != filter_pid) return 0;
    }
    return 1;
}

static __always_inline __u32 get_dir_churn(__u64 parent_ino, int bump)
{
    if (!parent_ino) return 0;
    __u64 now = bpf_ktime_get_ns();
    struct dir_stat *ds = bpf_map_lookup_elem(&dir_map, &parent_ino);
    if (ds) {
        if (now > ds->last_ts && (now - ds->last_ts) > 5000000000ULL)
            ds->ema >>= 1;
        if (bump) {
            ds->ema += 1;
            ds->last_ts = now;
        }
        return ds->ema;
    }
    if (bump) {
        struct dir_stat init = { .last_ts = now, .ema = 1 };
        bpf_map_update_elem(&dir_map, &parent_ino, &init, BPF_NOEXIST);
        return 1;
    }
    return 0;
}

static __always_inline void emit_ev(
    void *ctx, __u32 op, struct inode *ino, __u64 parent_ino,
    __u64 off, __u64 cnt, __s64 stride, __u32 flags, __u32 churn,
    __u32 win_pages, __u32 win_reused, __u8 is_seq, __u8 mode_changed)
{
    struct kssm_event *e = bpf_ringbuf_reserve(&rb, sizeof(*e), 0);
    if (!e) {
        inc_cnt(1);
        return;
    }
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    e->ts_ns = bpf_ktime_get_ns();
    e->ino = ino->i_ino;
    e->parent_ino = parent_ino;
    e->offset = off;
    e->count = cnt;
    e->file_size = (__u64)ino->i_size;
    e->stride = stride;
    e->pid = pid_tgid >> 32;
    e->tid = (__u32)pid_tgid;
    e->flags = flags;
    e->op = op;
    e->u_stack_id = (__s32)bpf_get_stackid(ctx, &stack_map, BPF_F_USER_STACK | BPF_F_FAST_STACK_CMP);
    e->k_stack_id = (__s32)bpf_get_stackid(ctx, &stack_map, BPF_F_FAST_STACK_CMP);
    e->dir_churn = churn;
    e->win_pages = win_pages;
    e->win_reused = win_reused;
    e->is_seq = is_seq;
    e->mode_changed = mode_changed;
    bpf_get_current_comm(e->comm, sizeof(e->comm));
    bpf_ringbuf_submit(e, 0);
    inc_cnt(0);
}

static __always_inline void check_page_reuse(__u64 ino_num, __u64 off, struct ino_track *tr)
{
    struct page_key pk = { .ino = ino_num, .pgoff = off >> 12 };
    __u32 *seen = bpf_map_lookup_elem(&page_seen_map, &pk);
    tr->win_pages += 1;
    if (seen) {
        (*seen)++;
        tr->win_reused += 1;
    } else {
        __u32 one = 1;
        bpf_map_update_elem(&page_seen_map, &pk, &one, BPF_NOEXIST);
    }
}

static __always_inline int on_rw(void *ctx, struct kiocb *iocb, struct iov_iter *iter, int is_write)
{
    if (!iocb || !iter) return 0;
    struct file *file = iocb->ki_filp;
    if (!file) return 0;
    struct inode *ino = file->f_inode;
    if (!check_dev(ino)) return 0;

    __u64 off = (__u64)iocb->ki_pos;
    __u64 cnt = (__u64)iter->count;
    __u64 ino_num = ino->i_ino;

    struct ino_track *tr = bpf_map_lookup_elem(&ino_map, &ino_num);
    if (!tr) {
        struct ino_track init = {};
        bpf_map_update_elem(&ino_map, &ino_num, &init, BPF_NOEXIST);
        tr = bpf_map_lookup_elem(&ino_map, &ino_num);
        if (!tr) return 0;
    }

    if (!is_write)
        check_page_reuse(ino_num, off, tr);

    __s64 stride = (__s64)off - (__s64)tr->last_off;
    __u8 is_seq = (stride == (__s64)tr->last_size) || (stride == 0);
    __u8 cur_mode = is_write ? (is_seq ? 3 : 4) : (is_seq ? 1 : 2);
    __u8 mode_changed = (tr->last_mode != cur_mode);

    tr->last_off = off;
    tr->last_size = (__u32)cnt;
    __u32 acc = tr->bytes_acc + (__u32)cnt;

    if (!mode_changed && acc < 131072U) {
        tr->bytes_acc = acc;
        return 0;
    }

    __u32 wp = tr->win_pages;
    __u32 wr = tr->win_reused;
    tr->last_mode = cur_mode;
    tr->bytes_acc = 0;
    tr->win_pages = 0;
    tr->win_reused = 0;

    __u64 pino = 0;
    struct dentry *d = file->f_path.dentry;
    if (d && d->d_parent && d->d_parent->d_inode)
        pino = d->d_parent->d_inode->i_ino;
    __u32 churn = get_dir_churn(pino, 0);

    emit_ev(ctx, is_write ? EV_WRITE : EV_READ, ino, pino, off, cnt, stride,
            file->f_flags, churn, wp, wr, is_seq, mode_changed);
    return 0;
}

SEC("fentry/ext4_file_read_iter")
int BPF_PROG(on_ext4_read, struct kiocb *iocb, struct iov_iter *to) { return on_rw(ctx, iocb, to, 0); }

SEC("fentry/ext4_file_write_iter")
int BPF_PROG(on_ext4_write, struct kiocb *iocb, struct iov_iter *from) { return on_rw(ctx, iocb, from, 1); }

SEC("fentry/vfs_open")
int BPF_PROG(on_vfs_open, const struct path *path, struct file *file)
{
    if (!path || !file) return 0;
    struct dentry *d = path->dentry;
    if (!d) return 0;
    struct inode *ino = d->d_inode;
    if (!check_dev(ino)) return 0;

    __u64 pino = 0;
    if (d->d_parent && d->d_parent->d_inode)
        pino = d->d_parent->d_inode->i_ino;

    __u32 fl = file->f_flags;
    int is_creat = (fl & 0100) != 0;
    int is_trunc = (fl & 01000) != 0;
    int is_wr    = (fl & 3) != 0;

    __u32 op = EV_OPEN_RO;
    if (is_creat && ino->i_size == 0) op = EV_OPEN_CREAT;
    else if (is_trunc) op = EV_TRUNC;
    else if (is_wr) op = EV_OPEN_RW;

    __u32 churn = get_dir_churn(pino, (op == EV_OPEN_CREAT));
    emit_ev(ctx, op, ino, pino, 0, 0, 0, fl, churn, 0, 0, 1, 1);
    return 0;
}

SEC("fentry/vfs_unlink")
int BPF_PROG(on_vfs_unlink, struct user_namespace *mnt_userns, struct inode *dir, struct dentry *dentry)
{
    if (!dir || !dentry) return 0;
    struct inode *ino = dentry->d_inode;
    if (!check_dev(ino)) return 0;

    __u64 pino = dir->i_ino;
    __u32 churn = get_dir_churn(pino, 1);
    emit_ev(ctx, EV_UNLINK, ino, pino, 0, 0, 0, 0, churn, 0, 0, 0, 1);

    __u64 ino_num = ino->i_ino;
    bpf_map_delete_elem(&ino_map, &ino_num);
    return 0;
}

SEC("fentry/vfs_rename")
int BPF_PROG(on_vfs_rename, struct renamedata *rd)
{
    if (!rd) return 0;
    struct dentry *od = rd->old_dentry;
    if (!od) return 0;
    struct inode *ino = od->d_inode;
    if (!check_dev(ino)) return 0;

    __u64 pino = rd->new_dir ? rd->new_dir->i_ino : 0;
    __u32 churn = get_dir_churn(pino, 1);
    emit_ev(ctx, EV_RENAME, ino, pino, 0, 0, 0, rd->flags, churn, 0, 0, 0, 1);

    struct dentry *nd = rd->new_dentry;
    if (nd && nd->d_inode && check_dev(nd->d_inode)) {
        emit_ev(ctx, EV_UNLINK, nd->d_inode, pino, 0, 0, 0, 1, churn, 0, 0, 0, 1);
    }
    return 0;
}

SEC("fentry/vfs_fsync_range")
int BPF_PROG(on_vfs_fsync, struct file *file, loff_t start, loff_t end, int datasync)
{
    if (!file) return 0;
    struct inode *ino = file->f_inode;
    if (!check_dev(ino)) return 0;

    __u64 pino = 0;
    struct dentry *d = file->f_path.dentry;
    if (d && d->d_parent && d->d_parent->d_inode)
        pino = d->d_parent->d_inode->i_ino;
    __u32 churn = get_dir_churn(pino, 0);
    emit_ev(ctx, EV_FSYNC, ino, pino, (__u64)start, 0, 0, (__u32)datasync, churn, 0, 0, 0, 1);
    return 0;
}

SEC("fentry/filemap_fault")
int BPF_PROG(on_filemap_fault, struct vm_fault *vmf)
{
    if (!vmf) return 0;
    struct vm_area_struct *vma = vmf->vma;
    if (!vma) return 0;
    struct file *file = vma->vm_file;
    if (!file) return 0;
    struct inode *ino = file->f_inode;
    if (!check_dev(ino)) return 0;

    __u64 off = ((__u64)vmf->pgoff) << 12;
    __u64 ino_num = ino->i_ino;
    struct ino_track *tr = bpf_map_lookup_elem(&ino_map, &ino_num);
    if (!tr) {
        struct ino_track init = {};
        bpf_map_update_elem(&ino_map, &ino_num, &init, BPF_NOEXIST);
        tr = bpf_map_lookup_elem(&ino_map, &ino_num);
        if (!tr) return 0;
    }

    check_page_reuse(ino_num, off, tr);
    __s64 stride = (__s64)off - (__s64)tr->last_off;
    __u8 is_seq = (stride >= 0 && stride <= 131072);
    __u8 mode_changed = (tr->last_mode != 5);
    tr->last_off = off;
    tr->last_size = 4096;
    __u32 acc = tr->bytes_acc + 4096;

    if (!mode_changed && acc < 131072U) {
        tr->bytes_acc = acc;
        return 0;
    }

    __u32 wp = tr->win_pages;
    __u32 wr = tr->win_reused;
    tr->last_mode = 5;
    tr->bytes_acc = 0;
    tr->win_pages = 0;
    tr->win_reused = 0;

    __u64 pino = 0;
    struct dentry *d = file->f_path.dentry;
    if (d && d->d_parent && d->d_parent->d_inode)
        pino = d->d_parent->d_inode->i_ino;
    __u32 churn = get_dir_churn(pino, 0);

    emit_ev(ctx, EV_MMAP_FAULT, ino, pino, off, 4096, stride, 0, churn, wp, wr, is_seq, mode_changed);
    return 0;
}
