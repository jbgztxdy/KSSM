#ifndef __KSSM_EXP4_SHARED_H
#define __KSSM_EXP4_SHARED_H

#define KSSM_DIM 64
#define KSSM_TOTAL_SLOTS 192
#define KSSM_S1_OFF 0
#define KSSM_S2_OFF 32
#define KSSM_S3_OFF 96
#define KSSM_S4_OFF 128
#define KSSM_S5_OFF 160

#define EXP4_MODE_KSSM     0
#define EXP4_MODE_SIZE     1
#define EXP4_MODE_SEQDROP  2

struct kssm_live_weights {
    __u64 emb[KSSM_TOTAL_SLOTS][16];
    __s16 w_scan_i[KSSM_DIM];
    __s16 w_scan_t[KSSM_DIM];
    __s32 b_scan;
};

struct exp4_drop_hint {
    __u64 ino;
    __u64 drop_start;
    __u64 drop_len;
    __u32 pid;
    __s32 score_q8;
    __u8  is_write;
    __u8  pad[7];
};

struct exp4_stats {
    __u64 rw_hooks;
    __u64 ssm_updates;
    __u64 reuse_hits;
    __u64 scan_votes;
    __u64 hot_votes;
    __u64 hints_sent;
    __u64 rb_drops;
};
#endif
