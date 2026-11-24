#ifndef __KSSM_COMMON_H
#define __KSSM_COMMON_H

#define KSSM_DIM 64
#define KSSM_DIM_32 32

/* 5-tuple discrete vocabulary sizes: 32 + 64 + 32 + 32 + 32 = 192 slots */
#define KSSM_S1_SIZE 32
#define KSSM_S2_SIZE 64
#define KSSM_S3_SIZE 32
#define KSSM_S4_SIZE 32
#define KSSM_S5_SIZE 32

#define KSSM_S1_OFF 0
#define KSSM_S2_OFF 32
#define KSSM_S3_OFF 96
#define KSSM_S4_OFF 128
#define KSSM_S5_OFF 160
#define KSSM_TOTAL_SLOTS 192

#define KSSM_CLUSTER_SHIFT 17 /* 128 KB */
#define KSSM_STACK_RING_SZ 16

/* 128 Bytes exact hidden state: 64 x int16_t (Q7.8 fixed-point) */
struct kssm_hidden {
    __s16 h[KSSM_DIM];
};

struct kssm_hidden32 {
    __s16 h[KSSM_DIM_32];
};

/* Per-task state stored in BPF_MAP_TYPE_TASK_STORAGE */
struct kssm_task_state {
    struct kssm_hidden hidden;          /* 128 B */
    __u32 stack_ring[KSSM_STACK_RING_SZ]; /* 64 B: recent 16 stack IDs */
    __u16 ring_head;
    __u16 recent_reads;
    __u16 recent_writes;
    __u16 pad;
};

/* Per-inode state stored in BPF_MAP_TYPE_INODE_STORAGE (LSM) or LRU_HASH (fentry) */
struct kssm_inode_state {
    struct kssm_hidden hidden;          /* 128 B */
    __u64 last_off;
    __u64 last_cluster;
    __s64 last_stride;
    __u32 last_size;
    __u32 parent_churn_bucket;
    __s16 score_scan;
    __s16 score_life;
    __u8  last_op;
    __u8  initialized;
    __u16 pad;
};

/* Pre-baked read-only lookup tables: 64 KB total (fits in L2, hot rows in L1d) */
struct kssm_tables {
    /* 192 * 64 * 2B = 24 KB: inode input bias embeddings */
    __s16 emb_inode[KSSM_TOTAL_SLOTS][KSSM_DIM];
    /* 192 * 64 * 2B = 24 KB: task input bias embeddings */
    __s16 emb_task[KSSM_TOTAL_SLOTS][KSSM_DIM];
    /* 32 * 4 * 64 * 2B = 16 KB: per-s1 diagonal decay & coupling gates (Q7.8) */
    __s16 alpha_inode[KSSM_S1_SIZE][KSSM_DIM];
    __s16 gamma_t2i[KSSM_S1_SIZE][KSSM_DIM];
    __s16 alpha_task[KSSM_S1_SIZE][KSSM_DIM];
    __s16 eta_i2t[KSSM_S1_SIZE][KSSM_DIM];
    /* Shift-gate exponents for shift-only variant (1 KB) */
    __u8  sh_decay_i[KSSM_S1_SIZE][KSSM_DIM]; /* alpha_i = 1 - 2^{-k} */
    __u8  sh_coup_i[KSSM_S1_SIZE][KSSM_DIM];  /* gamma   = +/- 2^{-k} (bit7=sign) */
    __u8  sh_decay_t[KSSM_S1_SIZE][KSSM_DIM];
    __u8  sh_coup_t[KSSM_S1_SIZE][KSSM_DIM];
};

/* Linear readout heads (< 1 KB): updated periodically by user-space slow loop */
struct kssm_readout {
    __s16 w_scan_i[KSSM_DIM];   /* 128 B */
    __s16 w_scan_t[KSSM_DIM];   /* 128 B */
    __s16 w_life_i[KSSM_DIM];   /* 128 B */
    __s16 w_life_t[KSSM_DIM];   /* 128 B */
    __s32 b_scan;
    __s32 b_life;
};

/* Per-CPU runtime counters */
struct kssm_cpu_stats {
    __u64 total_hooks;
    __u64 ssm_updates;
    __u64 fast_skips;
    __u64 scan_preds;
    __u64 short_life_preds;
};

#endif
