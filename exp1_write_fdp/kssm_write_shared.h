#ifndef __KSSM_WRITE_SHARED_H
#define __KSSM_WRITE_SHARED_H

#define KSSM_DIM 64
#define KSSM_TOTAL_SLOTS 192
#define KSSM_S1_OFF 0
#define KSSM_S2_OFF 32
#define KSSM_S3_OFF 96
#define KSSM_S4_OFF 128
#define KSSM_S5_OFF 160

/* Frozen 24 KB Guard-SWAR lookup base */
struct kssm_life_base {
    __u64 emb[KSSM_TOTAL_SLOTS][16];
};

/* Hot-swappable < 1 KB (784 Bytes) linear lifetime readout head updated by slow loop! */
struct kssm_life_head {
    __s16 w_i[3][KSSM_DIM]; /* 384 B: Q7.8 */
    __s16 w_t[3][KSSM_DIM]; /* 384 B: Q7.8 */
    __s32 b[3];             /*  12 B: Q15.16 */
    __s32 pad;              /*   4 B */
};

#define MSG_BIRTH_HINT  1
#define MSG_DEATH_LABEL 2

/* Ringbuf message: either 0th-second stream assignment hint OR delayed ground-truth death feedback */
struct kssm_write_msg {
    __u64 ino;
    __u64 ts_ns;
    __u64 lifetime_ns;       /* valid for MSG_DEATH_LABEL */
    __u32 pid;
    __u8  msg_type;          /* MSG_BIRTH_HINT or MSG_DEATH_LABEL */
    __u8  pred_class;        /* 0=Short, 1=Med, 2=Long -> RWH_WRITE_LIFE_SHORT/MEDIUM/LONG */
    __u8  true_class;        /* valid for MSG_DEATH_LABEL */
    __u8  pad;
    /* Birth hidden-state snapshot (compressed to 64+64 int16 = 256 B) for zero-BPTT slow-loop calibration! */
    __s16 snap_hi[KSSM_DIM];
    __s16 snap_ht[KSSM_DIM];
};

struct kssm_write_stats {
    __u64 total_hooks;
    __u64 ssm_updates;
    __u64 births_predicted;
    __u64 deaths_observed;
    __u64 rb_drops;
};

#endif
