#ifndef __KSSM_SELECTIVE_H
#define __KSSM_SELECTIVE_H

#define KSSM_DIM 64
#define KSSM_S1_SIZE 32
#define KSSM_TOTAL_SLOTS 192
#define KSSM_S1_OFF 0
#define KSSM_S2_OFF 32
#define KSSM_S3_OFF 96
#define KSSM_S4_OFF 128
#define KSSM_S5_OFF 160

/* Input-dependent Selective Gate descriptor per (s1, band) - 64 Bytes aligned!
 * Branchlessly encodes ALL 3 selective regimes:
 *   - HOLD (alpha=1, noise-immune): keep_mask=~0ULL, mask_d=0, mask_c=0
 *   - DECAY (alpha=1-2^-d):         keep_mask=~0ULL, mask_d=PACK4(0xFFFF>>d)
 *   - RESET (alpha=0, phase-cut):   keep_mask=0ULL,  mask_d=0, mask_c=PACK4(0xFFFF>>1)
 */
struct sel_band_gate {
    __u64 keep_i;
    __u64 mask_di;
    __u64 mask_ci;
    __u64 const_i;
    __u64 keep_t;
    __u64 mask_dt;
    __u64 mask_ct;
    __u64 const_t;
    __u32 d_i;
    __u32 c_i;
    __u32 d_t;
    __u32 c_t;
};

/* Total table size:
 *   emb:   192 * 16 * 8B = 24,576 B (24 KB)
 *   gates:  32 *  4 * 80B = 10,240 B (10 KB)
 *   TOTAL = 34.8 KB (100% fits inside 48 KB L1d cache!)
 */
struct kssm_selective_tables {
    __u64 emb[KSSM_TOTAL_SLOTS][16];
    struct sel_band_gate gate[KSSM_S1_SIZE][4];
};

struct bench_sel_pair {
    __u64 vi[16];
    __u64 vt[16];
    __u32 s1, s2, s3, s4, s5;
    __u32 pad;
};
#endif
