import os, math, pickle
import numpy as np
from collections import Counter, defaultdict

TR_DIR = "/home/yjw/workspace/research/KSSM/exp2_scale_distill/traces"
OUT_PKL = "/home/yjw/workspace/research/KSSM/exp2_scale_distill/dataset_10app.pkl"
APPS = ["rocksdb", "sqlite", "pytorch", "llm", "etl",
        "compiler", "git_vcs", "log_segment", "zstd_arch", "vllm_kv"]

# Allowed child process comms for multi-process build/vcs/archive workloads
CHILD_COMMS = {
    "compiler": {"python", "python3", "g++", "gcc", "cc1plus", "cc1", "as", "ld", "collect2"},
    "git_vcs":  {"python", "python3", "git", "git-pack-object", "git-repack", "git-gc"},
    "zstd_arch":{"python", "python3", "zstd", "tar"},
}

DTYPE = np.dtype([
    ("ts_ns", "<u8"), ("ino", "<u8"), ("parent_ino", "<u8"),
    ("offset", "<u8"), ("count", "<u8"), ("file_size", "<u8"),
    ("stride", "<i8"), ("pid", "<u4"), ("tid", "<u4"),
    ("flags", "<u4"), ("op", "<u4"), ("u_stack_id", "<i4"),
    ("k_stack_id", "<i4"), ("dir_churn", "<u4"), ("win_pages", "<u4"),
    ("win_reused", "<u4"), ("is_seq", "u1"), ("mode_changed", "u1"),
    ("pad", "2u1"), ("comm", "S16")
])

def fast_size_bucket(v):
    if v < (1 << 16): return 1 if v >= (1 << 12) else 0
    if v < (1 << 24): return 3 if v >= (1 << 20) else 2
    if v < (1 << 30): return 5 if v >= (1 << 28) else 4
    return 7 if v >= (1 << 32) else 6

def process_app(app_idx, app_name):
    pid = int(open(f"{TR_DIR}/{app_name}.pid").read().strip())
    arr = np.fromfile(f"{TR_DIR}/{app_name}.bin", dtype=DTYPE)
    if app_name in CHILD_COMMS:
        comms = np.array([c.split(b"\x00", 1)[0].decode("utf-8", "ignore") for c in arr["comm"]])
        allowed = CHILD_COMMS[app_name]
        mask = (arr["pid"] == pid) | np.isin(comms, list(allowed))
        sub = arr[mask]
    else:
        sub = arr[arr["pid"] == pid]

    sub = sub[np.argsort(sub["ts_ns"], kind="mergesort")]
    N = len(sub)
    t0, t1 = int(sub["ts_ns"][0]), int(sub["ts_ns"][-1])

    ops = sub["op"]; inos = sub["ino"]; tids = sub["tid"]; tss = sub["ts_ns"]
    offs = sub["offset"]; cnts = sub["count"]; fszs = sub["file_size"]
    strides = sub["stride"]; flags = sub["flags"]
    ustks = sub["u_stack_id"]; kstks = sub["k_stack_id"]
    churns = sub["dir_churn"]; is_seqs = sub["is_seq"]; mode_chgs = sub["mode_changed"]

    # Pass 1: (ino, gen) tracking across ext4 inode reuse
    ino_gen = defaultdict(int)
    inst_id_of_ev = np.zeros(N, dtype=np.int32)
    inst_map = {}; next_inst = 1
    birth_ev_idx, birth_ts, death_ts = {}, {}, {}

    for i in range(N):
        op = int(ops[i]); ino = int(inos[i]); ts = int(tss[i])
        key = (ino, ino_gen[ino])
        if op == 2:
            if key in inst_map and inst_map[key] in birth_ts:
                ino_gen[ino] += 1; key = (ino, ino_gen[ino])
            if key not in inst_map:
                inst_map[key] = next_inst; next_inst += 1
            iid = inst_map[key]
            birth_ev_idx[iid] = i; birth_ts[iid] = ts
            inst_id_of_ev[i] = iid
        elif op in (7, 10):
            if key not in inst_map:
                inst_map[key] = next_inst; next_inst += 1
            iid = inst_map[key]; inst_id_of_ev[i] = iid
            if iid in birth_ts and iid not in death_ts:
                death_ts[iid] = ts
            ino_gen[ino] += 1
        else:
            if key not in inst_map:
                inst_map[key] = next_inst; next_inst += 1
            inst_id_of_ev[i] = inst_map[key]

    uniq_tids = {}
    tid_id_of_ev = np.zeros(N, dtype=np.int32)
    for i in range(N):
        t = int(tids[i])
        if t not in uniq_tids: uniq_tids[t] = len(uniq_tids)
        tid_id_of_ev[i] = uniq_tids[t]

    life_ev_indices, life_labels, life_dt, life_b_s, life_d_s = [], [], [], [], []
    for iid, ev_i in birth_ev_idx.items():
        tb = birth_ts[iid]
        if iid in death_ts:
            dt = (death_ts[iid] - tb) / 1e9
            cls = 0 if dt < 10.0 else (1 if dt < 50.0 else 2)
            life_ev_indices.append(ev_i); life_labels.append(cls)
            life_dt.append(dt); life_b_s.append((tb - t0)/1e9); life_d_s.append((death_ts[iid] - t0)/1e9)
        else:
            if (t1 - tb) / 1e9 >= 22.0:
                life_ev_indices.append(ev_i); life_labels.append(2)
                life_dt.append((t1 - tb)/1e9); life_b_s.append((tb - t0)/1e9); life_d_s.append(-1.0)

    lo = np.argsort(life_ev_indices)
    life_ev_indices = np.array(life_ev_indices, dtype=np.int32)[lo]
    life_labels = np.array(life_labels, dtype=np.int64)[lo]
    life_dt = np.array(life_dt, dtype=np.float32)[lo]
    life_b_s = np.array(life_b_s, dtype=np.float32)[lo]
    life_d_s = np.array(life_d_s, dtype=np.float32)[lo]

    # Pass 2: Forward byte-reuse-distance ground truth for One-Shot Scan vs Hot Reusable
    rd_idx = np.where((ops == 4) | (ops == 9))[0]
    n_rd = len(rd_idx)
    cum_b = np.zeros(n_rd + 1, dtype=np.int64)
    rd_keys = [None] * n_rd
    for j in range(n_rd):
        idx = int(rd_idx[j])
        cum_b[j + 1] = cum_b[j] + max(4096, int(cnts[idx]))
        rd_keys[j] = (int(inst_id_of_ev[idx]), int(offs[idx]) >> 17)

    next_seen = {}
    scan_ev_indices, scan_labels = [], []
    tail_cut = int(n_rd * 0.85)
    for j in range(n_rd - 1, -1, -1):
        k = rd_keys[j]; lbl = -1
        if k in next_seen:
            dist = cum_b[next_seen[k]] - cum_b[j]
            if dist < 16 * 1024 * 1024: lbl = 0
            elif dist >= 64 * 1024 * 1024: lbl = 1
        else:
            if j < tail_cut: lbl = 1
        next_seen[k] = j
        if lbl != -1:
            scan_ev_indices.append(int(rd_idx[j]))
            scan_labels.append(lbl)
    scan_ev_indices = np.array(scan_ev_indices[::-1], dtype=np.int32)
    scan_labels = np.array(scan_labels[::-1], dtype=np.int64)

    # Pass 3: Strictly causal 5-tuple grammar (per-thread stride + scale-invariant relative s5)
    s_gram = np.zeros((N, 5), dtype=np.uint8)
    scalars = np.zeros((N, 16), dtype=np.float32)
    pc_sigs = np.zeros(N, dtype=np.int64)
    basic_tok = np.zeros(N, dtype=np.int32)

    num_t = len(uniq_tids)
    stack_rings = np.full((num_t, 16), -1, dtype=np.int64)
    ring_heads = np.zeros(num_t, dtype=np.int32)
    t_reads = np.zeros(num_t, dtype=np.int32)
    t_writes = np.zeros(num_t, dtype=np.int32)
    t_last_inst = np.full(num_t, -1, dtype=np.int32)
    t_last_off  = np.zeros(num_t, dtype=np.int64)
    t_last_cnt  = np.zeros(num_t, dtype=np.int64)
    t_seq_run   = np.zeros(num_t, dtype=np.int32)
    proc_churn_ema = 2.0

    for i in range(N):
        op = int(ops[i]); tid = int(tid_id_of_ev[i]); iid = int(inst_id_of_ev[i])
        fl = int(flags[i]); off = int(offs[i]); cnt = int(cnts[i]); fsz = int(fszs[i])
        ch = int(churns[i]); mchg = int(mode_chgs[i])

        if op in (4, 5, 9):
            if t_last_inst[tid] == iid:
                st = off - int(t_last_off[tid])
                seq = 1 if (st > 0 and st <= max(131072, int(t_last_cnt[tid]) * 2)) else (1 if (st == 0 and int(is_seqs[i])) else 0)
            else:
                st = int(strides[i]); seq = int(is_seqs[i])
            t_last_inst[tid] = iid; t_last_off[tid] = off; t_last_cnt[tid] = cnt
            t_seq_run[tid] = min(31, t_seq_run[tid] + 1) if seq else 0
        else:
            st = 0; seq = int(is_seqs[i])

        if op == 1: s1 = 0
        elif op == 2: s1 = 1 + (1 if (fl & 0o2000) else 0) + (2 if (fl & 0o1000) else 0)
        elif op == 3: s1 = 5 + (1 if (fl & 0o2000) else 0)
        elif op == 10: s1 = 7
        elif op == 4: s1 = 8 if seq else 9
        elif op == 5: s1 = 10 if (off >= fsz or (fl & 0o2000)) else (11 if seq else 12)
        elif op == 6: s1 = 13
        elif op == 7: s1 = 14
        elif op == 8: s1 = 15
        elif op == 9: s1 = 16 if seq else 17
        else: s1 = 18

        is_w = op in (2, 3, 5, 6, 7, 8, 10)
        if is_w: t_writes[tid] += 1
        else: t_reads[tid] += 1
        if t_reads[tid] + t_writes[tid] > 256:
            t_reads[tid] >>= 1; t_writes[tid] >>= 1
        tot_rw = t_reads[tid] + t_writes[tid] + 1
        rw_q = min(3, (t_writes[tid] * 4) // tot_rw)

        raw_stk = int(ustks[i]) if int(ustks[i]) >= 0 else (int(kstks[i]) + 100000)
        hd = int(ring_heads[tid]); rg = stack_rings[tid]; dist = 15
        for d in range(16):
            if rg[(hd - d) & 15] == raw_stk: dist = d; break
        nh = (hd + 1) & 15; rg[nh] = raw_stk; ring_heads[tid] = nh
        s2 = ((rw_q << 4) | dist) & 63

        sz_b = fast_size_bucket(fsz)
        reg = 0 if off < 65536 else (3 if off + cnt >= fsz else (1 if off < (fsz >> 1) else 2))
        s3 = ((sz_b << 2) | reg) & 31

        req_b = (1 if cnt >= 16384 else 0) + (1 if cnt >= 65536 else 0) + (1 if cnt >= 262144 else 0)
        if seq: str_b = 0 if t_seq_run[tid] >= 3 else 4
        elif st < 0: str_b = 1 if st > -131072 else 2
        elif st == 0: str_b = 3
        elif st <= 65536: str_b = 5
        elif st <= 524288: str_b = 6
        else: str_b = 7
        s4 = ((req_b << 3) | str_b) & 31

        ch_log = math.log2(ch + 1.0)
        if op in (2, 7, 8): proc_churn_ema = 0.95 * proc_churn_ema + 0.05 * ch_log
        rel_delta = ch_log - proc_churn_ema
        rel_b = max(0, min(15, int(round((rel_delta + 4.0) * 1.875))))
        s5 = (((0 if ch <= 4 else 1) << 4) | rel_b) & 31

        s_gram[i] = (s1, s2, s3, s4, s5)

        scalars[i] = (
            op / 10.0, float(seq), float(mchg), math.log2(fsz + 1) / 35.0,
            math.log2(off + 1) / 35.0, min(1.0, off / (fsz + 1.0)), math.log2(cnt + 1) / 22.0,
            1.0 if st > 0 else (-1.0 if st < 0 else 0.0), math.log2(abs(st) + 1) / 35.0,
            rel_delta / 4.0, float((fl & 0o100) != 0), float((fl & 0o1000) != 0),
            float((fl & 0o2000) != 0), float((fl & 3) != 0), float(t_writes[tid]) / float(tot_rw), float(reg) / 3.0
        )
        pc_sigs[i] = (app_idx << 32) | (raw_stk & 0xFFFFFFFF)
        basic_tok[i] = (s1 << 5) | s4

    s0 = float(np.mean(scalars[scan_ev_indices[scan_labels == 0], 1])) if np.any(scan_labels == 0) else 0
    s1_m = float(np.mean(scalars[scan_ev_indices[scan_labels == 1], 1])) if np.any(scan_labels == 1) else 0
    print(f"[{app_name:11s}] N={N:6d} | tids={num_t:4d} | life_0s={len(life_labels):4d} {dict(Counter(life_labels.tolist()))} | "
          f"scan={len(scan_labels):6d} {dict(Counter(scan_labels.tolist()))} (seq: Hot={s0:.2f}, Scan={s1_m:.2f})")

    return {
        "name": app_name, "N": N, "dur_s": (t1 - t0) / 1e9,
        "ts_rel_s": ((tss - t0) / 1e9).astype(np.float32),
        "tid": tid_id_of_ev, "inst": inst_id_of_ev,
        "num_tids": num_t, "num_insts": next_inst,
        "s_gram": s_gram, "scalars": scalars, "pc_sigs": pc_sigs, "basic_tok": basic_tok,
        "op": ops.astype(np.uint8),
        "life_ev": life_ev_indices, "life_y": life_labels, "life_dt": life_dt,
        "life_birth_s": life_b_s, "life_death_s": life_d_s,
        "scan_ev": scan_ev_indices, "scan_y": scan_labels,
    }

data = {app: process_app(idx, app) for idx, app in enumerate(APPS)}
with open(OUT_PKL, "wb") as f:
    pickle.dump(data, f, protocol=4)
print("Saved 10-app dataset to", OUT_PKL)
