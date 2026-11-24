import os, math, pickle
import numpy as np
from collections import Counter, defaultdict

TR_DIR = "/home/yjw/workspace/research/KSSM/exp0b/traces"
OUT_PKL = "/home/yjw/workspace/research/KSSM/exp0b/dataset_exp0b.pkl"
APPS = ["rocksdb", "sqlite", "pytorch", "llm", "etl"]

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

def log2_bucket(v, max_b=31):
    if v <= 0: return 0
    return min(max_b, int(math.log2(v + 1)))

def process_app(app_idx, app_name):
    pid = int(open(f"{TR_DIR}/{app_name}.pid").read().strip())
    arr = np.fromfile(f"{TR_DIR}/{app_name}.bin", dtype=DTYPE)
    sub = arr[arr["pid"] == pid]
    order = np.argsort(sub["ts_ns"], kind="mergesort")
    sub = sub[order]

    # To keep RocksDB (~299k events) tractable while preserving 100% of ALL metadata/write/birth/death
    # events AND full continuous read context around every birth & sampled read window:
    # Actually 299k events is fast enough in numpy/python to process in 3 seconds! No dropping needed!
    N = len(sub)
    t0 = int(sub["ts_ns"][0])
    t1 = int(sub["ts_ns"][-1])

    ops = sub["op"]
    inos = sub["ino"]
    tids = sub["tid"]
    tss = sub["ts_ns"]
    offs = sub["offset"]
    cnts = sub["count"]
    fszs = sub["file_size"]
    strides = sub["stride"]
    flags = sub["flags"]
    ustks = sub["u_stack_id"]
    kstks = sub["k_stack_id"]
    churns = sub["dir_churn"]
    is_seqs = sub["is_seq"]
    mode_chgs = sub["mode_changed"]

    # Pass 1: Assign exact (ino, gen) file instance IDs and compute ground-truth lifetimes
    ino_gen = defaultdict(int)
    inst_id_of_ev = np.zeros(N, dtype=np.int32)
    inst_map = {} # (ino, gen) -> int inst_id
    next_inst = 1

    birth_ev_idx = {}   # inst_id -> event index of birth (EV_OPEN_CREAT)
    birth_ts = {}       # inst_id -> ts_ns
    death_ts = {}       # inst_id -> ts_ns

    for i in range(N):
        op = int(ops[i])
        ino = int(inos[i])
        ts = int(tss[i])
        key = (ino, ino_gen[ino])

        if op == 2: # OPEN_CREAT
            if key in inst_map and inst_map[key] in birth_ts:
                ino_gen[ino] += 1
                key = (ino, ino_gen[ino])
            if key not in inst_map:
                inst_map[key] = next_inst
                next_inst += 1
            iid = inst_map[key]
            birth_ev_idx[iid] = i
            birth_ts[iid] = ts
            inst_id_of_ev[i] = iid
        elif op in (7, 10): # UNLINK or TRUNC
            if key not in inst_map:
                inst_map[key] = next_inst
                next_inst += 1
            iid = inst_map[key]
            inst_id_of_ev[i] = iid
            if iid in birth_ts and iid not in death_ts:
                death_ts[iid] = ts
            ino_gen[ino] += 1
        else:
            if key not in inst_map:
                inst_map[key] = next_inst
                next_inst += 1
            inst_id_of_ev[i] = inst_map[key]

    # Map tid -> compact 0..num_tids-1
    uniq_tids = {}
    tid_id_of_ev = np.zeros(N, dtype=np.int32)
    for i in range(N):
        t = int(tids[i])
        if t not in uniq_tids:
            uniq_tids[t] = len(uniq_tids)
        tid_id_of_ev[i] = uniq_tids[t]

    # Build strict ground-truth lifetime labels at birth event index (0th second!)
    life_ev_indices = []
    life_labels = []
    life_dt_sec = []
    life_birth_rel_sec = []
    life_death_rel_sec = []

    for iid, ev_i in birth_ev_idx.items():
        tb = birth_ts[iid]
        if iid in death_ts:
            dt = (death_ts[iid] - tb) / 1e9
            cls = 0 if dt < 10.0 else (1 if dt < 50.0 else 2)
            life_ev_indices.append(ev_i)
            life_labels.append(cls)
            life_dt_sec.append(dt)
            life_birth_rel_sec.append((tb - t0) / 1e9)
            life_death_rel_sec.append((death_ts[iid] - t0) / 1e9)
        else:
            # Strict right-censoring guard: must survive >= 22s before trace end
            if (t1 - tb) / 1e9 >= 22.0:
                life_ev_indices.append(ev_i)
                life_labels.append(2)
                life_dt_sec.append((t1 - tb) / 1e9)
                life_birth_rel_sec.append((tb - t0) / 1e9)
                life_death_rel_sec.append(-1.0) # still alive

    # Sort lifetime targets by event index
    l_ord = np.argsort(life_ev_indices)
    life_ev_indices = np.array(life_ev_indices, dtype=np.int32)[l_ord]
    life_labels = np.array(life_labels, dtype=np.int64)[l_ord]
    life_dt_sec = np.array(life_dt_sec, dtype=np.float32)[l_ord]
    life_birth_rel_sec = np.array(life_birth_rel_sec, dtype=np.float32)[l_ord]
    life_death_rel_sec = np.array(life_death_rel_sec, dtype=np.float32)[l_ord]

    # Pass 2: Exact forward byte-reuse-distance ground truth for One-Shot Scan vs Hot Reusable
    rd_idx = np.where((ops == 4) | (ops == 9))[0]
    n_rd = len(rd_idx)
    cum_b = np.zeros(n_rd + 1, dtype=np.int64)
    rd_keys = [None] * n_rd
    for j in range(n_rd):
        idx = int(rd_idx[j])
        cum_b[j + 1] = cum_b[j] + max(4096, int(cnts[idx]))
        rd_keys[j] = (int(inst_id_of_ev[idx]), int(offs[idx]) >> 17)

    next_seen = {}
    scan_ev_indices = []
    scan_labels = []
    tail_cut = int(n_rd * 0.85)
    for j in range(n_rd - 1, -1, -1):
        k = rd_keys[j]
        lbl = -1
        if k in next_seen:
            dist = cum_b[next_seen[k]] - cum_b[j]
            if dist < 16 * 1024 * 1024:
                lbl = 0 # Hot Reusable
            elif dist >= 64 * 1024 * 1024:
                lbl = 1 # One-Shot Scan
        else:
            if j < tail_cut:
                lbl = 1 # Never reused -> One-Shot Scan
        next_seen[k] = j
        if lbl != -1:
            scan_ev_indices.append(int(rd_idx[j]))
            scan_labels.append(lbl)

    scan_ev_indices = np.array(scan_ev_indices[::-1], dtype=np.int32)
    scan_labels = np.array(scan_labels[::-1], dtype=np.int64)

    # Pass 3: Strictly causal online feature extraction (exact match with kernel eBPF!)
    # 5-tuple grammar u_t = (s1, s2, s3, s4, s5) + Heimdall stateless scalars + PCStream raw PC + BASIC bucket
    s_gram = np.zeros((N, 5), dtype=np.uint8)
    scalars = np.zeros((N, 16), dtype=np.float32)
    pc_sigs = np.zeros(N, dtype=np.int64)
    basic_tok = np.zeros(N, dtype=np.int32)

    # Per-thread state for s2 (16-slot call-stack ring + recent RW counts)
    num_t = len(uniq_tids)
    stack_rings = np.full((num_t, 16), -1, dtype=np.int64)
    ring_heads = np.zeros(num_t, dtype=np.int32)
    t_reads = np.zeros(num_t, dtype=np.int32)
    t_writes = np.zeros(num_t, dtype=np.int32)

    for i in range(N):
        op = int(ops[i])
        tid = int(tid_id_of_ev[i])
        fl = int(flags[i])
        off = int(offs[i])
        cnt = int(cnts[i])
        fsz = int(fszs[i])
        st = int(strides[i])
        ch = int(churns[i])
        seq = int(is_seqs[i])
        mchg = int(mode_chgs[i])

        # s1 in [0..31]: VFS primitive + access pattern + key open flags
        if op == 1: s1 = 0                         # OPEN_RO
        elif op == 2:                              # OPEN_CREAT
            s1 = 1 + (1 if (fl & 0o2000) else 0) + (2 if (fl & 0o1000) else 0) # +O_APPEND, +O_TRUNC
        elif op == 3: s1 = 5 + (1 if (fl & 0o2000) else 0) # OPEN_RW
        elif op == 10: s1 = 7                      # TRUNC
        elif op == 4: s1 = 8 if seq else 9         # READ seq vs rnd
        elif op == 5:                              # WRITE append vs seq_ow vs rnd_ow
            s1 = 10 if (off >= fsz) else (11 if seq else 12)
        elif op == 6: s1 = 13                      # FSYNC
        elif op == 7: s1 = 14                      # UNLINK
        elif op == 8: s1 = 15                      # RENAME
        elif op == 9: s1 = 16 if seq else 17       # MMAP seq vs rnd
        else: s1 = 18

        # s2 in [0..63]: call-stack recurrence distance (0..15) + thread recent RW ratio (0..3)
        is_w = op in (2, 3, 5, 6, 7, 8, 10)
        if is_w: t_writes[tid] += 1
        else: t_reads[tid] += 1
        if t_reads[tid] + t_writes[tid] > 256:
            t_reads[tid] >>= 1
            t_writes[tid] >>= 1
        tot_rw = t_reads[tid] + t_writes[tid] + 1
        rw_q = min(3, (t_writes[tid] * 4) // tot_rw)

        raw_stk = int(ustks[i]) if int(ustks[i]) >= 0 else (int(kstks[i]) + 100000)
        head = int(ring_heads[tid])
        ring = stack_rings[tid]
        dist = 15
        for d in range(16):
            if ring[(head - d) & 15] == raw_stk:
                dist = d
                break
        nhead = (head + 1) & 15
        ring[nhead] = raw_stk
        ring_heads[tid] = nhead
        s2 = ((rw_q << 4) | dist) & 63

        # s3 in [0..31]: normalized geometric position (log2(file_size) x region)
        sz_b = fast_size_bucket(fsz)
        if off < 65536: reg = 0
        elif off + cnt >= fsz: reg = 3
        elif off < (fsz >> 1): reg = 1
        else: reg = 2
        s3 = ((sz_b << 2) | reg) & 31

        # s4 in [0..31]: request size bucket (0..3) x stride bucket (0..7)
        req_b = (1 if cnt >= 16384 else 0) + (1 if cnt >= 65536 else 0) + (1 if cnt >= 262144 else 0)
        if seq: str_b = 0
        elif st < 0: str_b = 1 if st > -131072 else 2
        elif st == 0: str_b = 3
        elif st <= 65536: str_b = 4
        elif st <= 524288: str_b = 5
        elif st <= 4194304: str_b = 6
        else: str_b = 7
        s4 = ((req_b << 3) | str_b) & 31

        # s5 in [0..31]: parent directory churn bucket
        s5 = min(31, log2_bucket(ch, 31))

        s_gram[i, 0] = s1
        s_gram[i, 1] = s2
        s_gram[i, 2] = s3
        s_gram[i, 3] = s4
        s_gram[i, 4] = s5

        # Heimdall stateless isolated-event scalar vector (16 dims, NO sequence memory)
        scalars[i, 0] = op / 10.0
        scalars[i, 1] = float(seq)
        scalars[i, 2] = float(mchg)
        scalars[i, 3] = math.log2(fsz + 1) / 35.0
        scalars[i, 4] = math.log2(off + 1) / 35.0
        scalars[i, 5] = min(1.0, off / (fsz + 1.0))
        scalars[i, 6] = math.log2(cnt + 1) / 22.0
        scalars[i, 7] = 1.0 if st > 0 else (-1.0 if st < 0 else 0.0)
        scalars[i, 8] = math.log2(abs(st) + 1) / 35.0
        scalars[i, 9] = math.log2(ch + 1) / 16.0
        scalars[i, 10] = float((fl & 0o100) != 0)
        scalars[i, 11] = float((fl & 0o1000) != 0)
        scalars[i, 12] = float((fl & 0o2000) != 0)
        scalars[i, 13] = float((fl & 3) != 0)
        scalars[i, 14] = float(t_writes[tid]) / float(tot_rw)
        scalars[i, 15] = float(reg) / 3.0

        # PCStream: cross-binary raw PC hash (distinct per binary!)
        pc_sigs[i] = (app_idx << 32) | (raw_stk & 0xFFFFFFFF)

        # BASIC-Prefetcher: I/O size bucket x op/dir state (for 1st-order Markov chain)
        basic_tok[i] = (s1 << 5) | s4

    print(f"[{app_name:8s}] N={N:6d} | tids={num_t:2d} | insts={next_inst:5d} | "
          f"life_0s={len(life_labels):4d} {dict(Counter(life_labels.tolist()))} | "
          f"scan={len(scan_labels):6d} {dict(Counter(scan_labels.tolist()))}")

    return {
        "name": app_name,
        "N": N,
        "dur_s": (t1 - t0) / 1e9,
        "ts_rel_s": ((tss - t0) / 1e9).astype(np.float32),
        "tid": tid_id_of_ev,
        "inst": inst_id_of_ev,
        "num_tids": num_t,
        "num_insts": next_inst,
        "s_gram": s_gram,
        "scalars": scalars,
        "pc_sigs": pc_sigs,
        "basic_tok": basic_tok,
        "op": ops.astype(np.uint8),
        "life_ev": life_ev_indices,
        "life_y": life_labels,
        "life_dt": life_dt_sec,
        "life_birth_s": life_birth_rel_sec,
        "life_death_s": life_death_rel_sec,
        "scan_ev": scan_ev_indices,
        "scan_y": scan_labels,
    }

data = {}
for idx, app in enumerate(APPS):
    data[app] = process_app(idx, app)

with open(OUT_PKL, "wb") as f:
    pickle.dump(data, f, protocol=4)
print("Saved preprocessed dataset to", OUT_PKL)
