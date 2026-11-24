#ifndef __KSSM_TRACE_EVENT_H
#define __KSSM_TRACE_EVENT_H

#define EV_OPEN_RO     1
#define EV_OPEN_CREAT  2
#define EV_OPEN_RW     3
#define EV_READ        4
#define EV_WRITE       5
#define EV_FSYNC       6
#define EV_UNLINK      7
#define EV_RENAME      8
#define EV_MMAP_FAULT  9
#define EV_TRUNC       10

struct kssm_event {
    __u64 ts_ns;
    __u64 ino;
    __u64 parent_ino;
    __u64 offset;
    __u64 count;
    __u64 file_size;
    __s64 stride;
    __u32 pid;
    __u32 tid;
    __u32 flags;
    __u32 op;
    __s32 u_stack_id;
    __s32 k_stack_id;
    __u32 dir_churn;
    __u32 win_pages;
    __u32 win_reused;
    __u8  is_seq;
    __u8  mode_changed;
    __u8  pad[2];
    char  comm[16];
};

#endif
