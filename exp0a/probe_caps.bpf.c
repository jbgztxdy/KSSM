#include "../vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

char LICENSE[] SEC("license") = "GPL";

struct state_val {
    __s16 h[64];
};

struct {
    __uint(type, BPF_MAP_TYPE_TASK_STORAGE);
    __uint(map_flags, BPF_F_NO_PREALLOC);
    __type(key, int);
    __type(value, struct state_val);
} task_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_INODE_STORAGE);
    __uint(map_flags, BPF_F_NO_PREALLOC);
    __type(key, int);
    __type(value, struct state_val);
} inode_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_STACK_TRACE);
    __uint(key_size, sizeof(__u32));
    __uint(value_size, 16 * sizeof(__u64));
    __uint(max_entries, 4096);
} stack_map SEC(".maps");

// Test 1: fentry with task_storage ONLY
SEC("fentry/vfs_read")
int BPF_PROG(test_fentry_task, struct file *file, char *buf, size_t count, loff_t *pos)
{
    struct task_struct *t = bpf_get_current_task_btf();
    struct state_val *ts = bpf_task_storage_get(&task_map, t, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
    if (ts)
        ts->h[0] += 1;
    return 0;
}

// Test 2: fentry with task_storage + inode_storage
SEC("fentry/vfs_read")
int BPF_PROG(test_fentry_both, struct file *file, char *buf, size_t count, loff_t *pos)
{
    struct task_struct *t = bpf_get_current_task_btf();
    struct state_val *ts = bpf_task_storage_get(&task_map, t, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
    struct inode *ino = file->f_inode;
    struct state_val *is = bpf_inode_storage_get(&inode_map, ino, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
    if (ts && is)
        is->h[0] += ts->h[0];
    return 0;
}

// Test 3: lsm/file_open with task_storage + inode_storage
SEC("lsm/file_open")
int BPF_PROG(test_lsm_both, struct file *file)
{
    struct task_struct *t = bpf_get_current_task_btf();
    struct state_val *ts = bpf_task_storage_get(&task_map, t, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
    struct inode *ino = file->f_inode;
    struct state_val *is = bpf_inode_storage_get(&inode_map, ino, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
    if (ts && is)
        is->h[0] += ts->h[0];
    return 0;
}

// Test 4: fentry with bpf_get_stackid
SEC("fentry/vfs_read")
int BPF_PROG(test_fentry_stack, struct file *file, char *buf, size_t count, loff_t *pos)
{
    long sid = bpf_get_stackid(ctx, &stack_map, BPF_F_FAST_STACK_CMP);
    (void)sid;
    return 0;
}
