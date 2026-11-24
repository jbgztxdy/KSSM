#ifndef __KSSM_EXP0C_SHARED_H
#define __KSSM_EXP0C_SHARED_H

#define KSSM_DIM 64
#define KSSM_TOTAL_SLOTS 192
#define KSSM_S1_OFF 0
#define KSSM_S2_OFF 32
#define KSSM_S3_OFF 96
#define KSSM_S4_OFF 128
#define KSSM_S5_OFF 160

#define MODE_KSSM           0
#define MODE_HEURISTIC_SIZE 1

/* Exact layout matching kssm_weights.bin (24,836 Bytes -> fits in 48KB L1d!) */
struct kssm_live_weights {
    __u64 emb[KSSM_TOTAL_SLOTS][16]; /* 24,576 B: 4x uint16 Excess-2048 packed per u64 */
    __s16 w_scan_i[KSSM_DIM];        /* 128 B: Q7.8 */
    __s16 w_scan_t[KSSM_DIM];        /* 128 B: Q7.8 */
    __s32 b_scan;                    /* 4 B: Q15.16 */
};

struct kssm_drop_hint {
    __u64 ino;
    __u64 drop_start;
    __u64 drop_len;
    __u64 file_size;
    __u32 pid;
    __s32 score_q8;   /* scan score * 256 */
    __u8  is_write;
    __u8  pad[7];
};

struct kssm_exp0c_stats {
    __u64 total_rw_hooks;
    __u64 ssm_updates;
    __u64 scan_pos_votes;
    __u64 hot_neg_votes;
    __u64 hints_emitted;
    __u64 hints_dropped;
};

#endif
