#ifndef __KSSM_ABI_H
#define __KSSM_ABI_H

#define KSSM_DIM         64
#define KSSM_TOTAL_SLOTS 192
#define KSSM_S1_OFF      0
#define KSSM_S2_OFF      32
#define KSSM_S3_OFF      96
#define KSSM_S4_OFF      128
#define KSSM_S5_OFF      160

/* Frozen L1d-resident base model: 24,704 Bytes (24 KB emb + 128 B s1 selective gates) */
struct kssm_base_model {
    __u64 emb[KSSM_TOTAL_SLOTS][16]; /* 24,576 B: 4x uint16 Excess-2048 packed per u64 */
    __u8  reg_s1[32][4];             /*    128 B: 4-band selective regime per s1 (0=HOLD,1=SLOW,2=FAST,3=RESET) */
};

/* Hot-swappable dual-task linear readout head: 1,040 Bytes (~1 KB) updated by zero-BPTT slow loop */
struct kssm_dual_head {
    __s16 w_scan_i[KSSM_DIM];        /* 128 B: Q7.8 */
    __s16 w_scan_t[KSSM_DIM];        /* 128 B: Q7.8 */
    __s16 w_life_i[3][KSSM_DIM];     /* 384 B: Q7.8 */
    __s16 w_life_t[3][KSSM_DIM];     /* 384 B: Q7.8 */
    __s32 b_scan;                    /*   4 B: Q15.16 */
    __s32 b_life[3];                 /*  12 B: Q15.16 */
};

#define KSSM_MSG_DROP_HINT      1
#define KSSM_MSG_BIRTH_HINT     2
#define KSSM_MSG_DEATH_FEEDBACK 3

struct kssm_ctrl_msg {
    __u64 ino;
    __u64 arg0;       /* drop_start (DROP) | birth_ts_ns (BIRTH) | lifetime_ns (DEATH) */
    __u64 arg1;       /* drop_len   (DROP) | file_size   (BIRTH) | unused */
    __u32 pid;
    __s32 score_q8;
    __u8  msg_type;
    __u8  pred_class; /* 0=Short, 1=Med, 2=Long */
    __u8  true_class; /* valid for DEATH */
    __u8  is_write;   /* valid for DROP */
    __s16 snap_hi[KSSM_DIM]; /* 128 B birth snapshot for zero-BPTT slow loop */
    __s16 snap_ht[KSSM_DIM]; /* 128 B birth snapshot for zero-BPTT slow loop */
};

struct kssm_cpu_metrics {
    __u64 total_hooks;
    __u64 fast_skips;
    __u64 ssm_updates;
    __u64 scan_votes;
    __u64 hot_votes;
    __u64 births_pred;
    __u64 deaths_obs;
    __u64 msgs_sent;
    __u64 rb_drops;
};

/* Concurrency backend & gating modes for 1-64 core scalability evaluation */
#define KSSM_GATING_TL_PERCPU    0 /* Thread-local fast-path gating in TASK_STORAGE + Per-CPU LRU_HASH */
#define KSSM_GATING_SHARED_INODE 1 /* Fast-path gating in shared inode_map + Per-CPU LRU_HASH */
#define KSSM_GATING_COMMON_LRU   2 /* Shared global LRU_HASH (common LRU lock contention baseline) */

#endif
