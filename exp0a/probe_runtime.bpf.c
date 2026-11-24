#include "../vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

char LICENSE[] SEC("license") = "GPL";

__u64 lsm_hits = 0;
__u64 fentry_hits = 0;
__s64 last_kstack = -999;
__s64 last_ustack = -999;
__u32 target_pid = 0;

struct {
    __uint(type, BPF_MAP_TYPE_STACK_TRACE);
    __uint(key_size, sizeof(__u32));
    __uint(value_size, 16 * sizeof(__u64));
    __uint(max_entries, 4096);
} stack_map SEC(".maps");

SEC("lsm/file_open")
int BPF_PROG(on_lsm_open, struct file *file)
{
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid == target_pid)
        __sync_fetch_and_add(&lsm_hits, 1);
    return 0;
}

SEC("fentry/vfs_read")
int BPF_PROG(on_fentry_read, struct file *file, char *buf, size_t count, loff_t *pos)
{
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid == target_pid) {
        __sync_fetch_and_add(&fentry_hits, 1);
        last_kstack = bpf_get_stackid(ctx, &stack_map, BPF_F_FAST_STACK_CMP);
        last_ustack = bpf_get_stackid(ctx, &stack_map, BPF_F_USER_STACK | BPF_F_FAST_STACK_CMP);
    }
    return 0;
}
