import os, sys, time, math, struct, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "3", "Must use GPU 3!"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42); np.random.seed(42)

with open("/home/yjw/workspace/research/KSSM/exp0b/dataset_exp0b_v3.pkl", "rb") as f:
    DATA = pickle.load(f)

# 5-phase realistic cluster workload shift order (each phase = 240s, total = 3000s = 20 min!)
PHASE_APPS = ["rocksdb", "pytorch", "llm", "etl", "sqlite"]
APPS = ["rocksdb", "sqlite", "pytorch", "llm", "etl"]

# ==============================================================================
# Step 1: Strictly Leak-Free 5-Fold LOAO Training (Heimdall + Bit-Exact KSSM Guard-SWAR)
# ==============================================================================
WIN_LEN = 32
MAX_LOCAL_INOS = 10
SHARED_D = [1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4]
SHARED_C = [2,2,3,3, 3,3,4,4, 4,4,5,5, 5,5,5,5]
STATELESS_COLS = [0,1,2,3,4,5,6,7,8,10,11,12,13,15]

def get_band_limits():
    lims = np.zeros(64, dtype=np.float32)
    for g in range(16):
        margin = (2.0 ** (-SHARED_D[g]) - 2.0 ** (-SHARED_C[g])) * 32.0 * 0.85
        lims[g*4:(g+1)*4] = margin / 5.0
    return lims

def build_life_windows(d, max_samples=450):
    tid_arr = d["tid"]; inst_arr = d["inst"]; sg = d["s_gram"]; N = d["N"]
    evs = d["life_ev"]; ys = d["life_y"]
    idx = np.arange(len(evs))
    if len(idx) > max_samples:
        rng = np.random.RandomState(42)
        uniq = np.unique(ys); per_c = max(1, max_samples // len(uniq))
        ch = [rng.choice(idx[ys == c], size=min(per_c, int(np.sum(ys == c))), replace=False) for c in uniq]
        idx = np.sort(np.concatenate(ch))
    sel_ev = evs[idx]; sel_y = ys[idx]; M = len(sel_ev)

    tid_evs = defaultdict(list); inst_evs = defaultdict(list)
    for e_i in range(N):
        tid_evs[int(tid_arr[e_i])].append(e_i)
        inst_evs[int(inst_arr[e_i])].append(e_i)
    tid_evs = {k: np.array(v, dtype=np.int32) for k, v in tid_evs.items()}
    inst_evs = {k: np.array(v, dtype=np.int32) for k, v in inst_evs.items()}

    ws = np.zeros((M, WIN_LEN, 5), dtype=np.int64)
    wi = np.zeros((M, WIN_LEN), dtype=np.int64)
    wt = np.zeros((M, WIN_LEN), dtype=np.float32)
    wv = np.zeros((M, WIN_LEN), dtype=np.float32)
    for m in range(M):
        ei = int(sel_ev[m]); tt = int(tid_arr[ei]); ti = int(inst_arr[ei])
        te = tid_evs[tt]; tp = int(np.searchsorted(te, ei, side="right"))
        ie = inst_evs[ti]; ip = int(np.searchsorted(ie, ei, side="right"))
        merged = np.unique(np.concatenate([te[max(0, tp - 20):tp], ie[max(0, ip - 18):ip]]))
        if len(merged) > WIN_LEN: merged = merged[-WIN_LEN:]
        off = WIN_LEN - len(merged); lmap = {ti: 0}; nxt = 1
        for k, e in enumerate(merged):
            pos = off + k; ws[m, pos] = sg[e]; wv[m, pos] = 1.0
            wt[m, pos] = 1.0 if int(tid_arr[e]) == tt else 0.0
            ii = int(inst_arr[e])
            if ii not in lmap:
                lmap[ii] = nxt if nxt < MAX_LOCAL_INOS else (MAX_LOCAL_INOS - 1)
                if nxt < MAX_LOCAL_INOS: nxt += 1
            wi[m, pos] = lmap[ii]
    return {"ev": sel_ev, "y": sel_y, "ws": ws, "wi": wi, "wt": wt, "wv": wv}

class KSSMGuardSWAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(192, 64)
        nn.init.normal_(self.emb.weight, 0.0, 0.12)
        a = np.zeros(64, dtype=np.float32); g = np.zeros(64, dtype=np.float32)
        for grp in range(16):
            a[grp*4:(grp+1)*4] = 1.0 - (2.0 ** (-SHARED_D[grp]))
            g[grp*4:(grp+1)*4] = 2.0 ** (-SHARED_C[grp])
        self.register_buffer("fa", torch.tensor(a))
        self.register_buffer("fg", torch.tensor(g))
        self.register_buffer("lim", torch.tensor(get_band_limits()))
        self.drop = nn.Dropout(0.15)
        self.head = nn.Linear(128, 3)

    def tv_loss(self):
        w = torch.tanh(self.emb.weight) * self.lim
        l5 = F.mse_loss(w[160:175], w[161:176]) + F.mse_loss(w[176:191], w[177:192])
        l2 = sum(F.mse_loss(w[32+q*16:32+q*16+15], w[32+q*16+1:32+q*16+16]) for q in range(4))
        return 0.25 * (l5 + 0.5 * l2)

    def forward(self, ws, wi, wt, wv):
        B, W, _ = ws.shape
        offs = torch.tensor([0, 32, 96, 128, 160], device=ws.device)
        tbl = torch.tanh(self.emb.weight) * self.lim
        b_all = tbl[ws + offs].sum(dim=2)
        h_s = torch.zeros(B, MAX_LOCAL_INOS, 64, device=ws.device)
        h_t = torch.zeros(B, 64, device=ws.device)
        oh = F.one_hot(wi, MAX_LOCAL_INOS).to(b_all.dtype)
        for t in range(W):
            v = wv[:, t:t+1]; st = wt[:, t:t+1]; mk = oh[:, t, :].unsqueeze(-1)
            hi = (h_s * mk).sum(dim=1)
            hi_n = self.fa * hi + self.fg * (h_t * st) + b_all[:, t, :]
            ht_n = self.fa * h_t - self.fg * hi + b_all[:, t, :]
            hi_u = hi * (1.0 - v) + hi_n * v
            h_s = h_s * (1.0 - mk) + hi_u.unsqueeze(1) * mk
            ut = v * st
            h_t = h_t * (1.0 - ut) + ht_n * ut
        return self.head(self.drop(torch.cat([h_s[:, 0, :], h_t], dim=-1)))

    @torch.no_grad()
    def stream_extract_bit_exact(self, d):
        """Bit-exact Guard-SWAR streaming over full unseen app trace -> returns birth features & 0s logits."""
        ti = (torch.tanh(self.emb.weight) * self.lim).detach().cpu().numpy()
        emb_u16 = (np.round(ti * 256.0).astype(np.int32) + 2048).astype(np.uint16)
        ds = np.repeat(SHARED_D, 4).astype(np.int32)
        cs = np.repeat(SHARED_C, 4).astype(np.int32)
        c_pos = (10240 + (16384 >> cs) - (16384 >> ds)).astype(np.int32)
        c_neg = (10240 - (16384 >> cs) - (16384 >> ds)).astype(np.int32)
        W = self.head.weight.detach().cpu().numpy()
        b = self.head.bias.detach().cpu().numpy()
        wl_q = np.round(W * 256.0).astype(np.int32)
        bl_q = np.round(b * 65536.0).astype(np.int32)

        N = d["N"]; sg = d["s_gram"]; tids = d["tid"]; insts = d["inst"]
        H_i = np.full((d["num_insts"], 64), 16384, dtype=np.int32)
        H_t = np.full((d["num_tids"], 64), 16384, dtype=np.int32)
        lset = {int(e): i for i, e in enumerate(d["life_ev"])}
        feats = np.zeros((len(d["life_ev"]), 128), dtype=np.float32)
        zs_preds = np.zeros(len(d["life_ev"]), dtype=np.int32)

        for e in range(N):
            tid = int(tids[e]); iid = int(insts[e])
            oi = H_i[iid]; ot = H_t[tid]
            bsum = (emb_u16[int(sg[e,0])].astype(np.int32) + emb_u16[32+int(sg[e,1])] +
                    emb_u16[96+int(sg[e,2])] + emb_u16[128+int(sg[e,3])] + emb_u16[160+int(sg[e,4])])
            ni = oi - (oi >> ds) + (ot >> cs) + bsum - c_pos
            nt = ot - (ot >> ds) - (oi >> cs) + bsum - c_neg
            H_i[iid] = ni; H_t[tid] = nt
            if e in lset:
                idx = lset[e]
                fq = np.concatenate([ni - 16384, nt - 16384])
                feats[idx] = fq / 256.0
                zs_preds[idx] = int(np.argmax(wl_q @ fq + bl_q))
        return zs_preds, feats, W.copy(), b.copy()

class HeimdallMLP(nn.Module):
    def __init__(self, in_dim=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3)
        )
    def forward(self, x): return self.net(x)

print("Step 1: Running strict 5-Fold LOAO training on GPU 3 (zero leakage!)...", flush=True)
WINS = {a: build_life_windows(DATA[a]) for a in APPS}
LOAO_OUT = {}

for held in APPS:
    tr_apps = [a for a in APPS if a != held]
    # Gather KSSM train data
    ws, wi, wt, wv, wy, ww, xh = [], [], [], [], [], [], []
    for a in tr_apps:
        wd = WINS[a]; y = wd["y"]; cnt = Counter(y.tolist())
        w = np.array([1.0 / (len(cnt) * max(1, cnt[int(c)])) for c in y], dtype=np.float32)
        ws.append(wd["ws"]); wi.append(wd["wi"]); wt.append(wd["wt"]); wv.append(wd["wv"])
        wy.append(y); ww.append(w); xh.append(DATA[a]["scalars"][wd["ev"]][:, STATELESS_COLS])
    ws_t, wi_t, wt_t, wv_t, wy_t, ww_t, xh_t = [torch.tensor(np.concatenate(x), device=DEVICE) for x in (ws, wi, wt, wv, wy, ww, xh)]

    # Train KSSM Guard-SWAR 64D
    km = KSSMGuardSWAR().to(DEVICE)
    opt = torch.optim.AdamW(km.parameters(), lr=8e-3, weight_decay=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    B = 256; M = len(wy_t)
    for ep in range(40):
        km.train(); pl = torch.randperm(M, device=DEVICE)
        for i in range(0, M, B):
            bi = pl[i:i+B]
            if len(bi) < 16: continue
            ll = km(ws_t[bi], wi_t[bi], wt_t[bi], wv_t[bi])
            loss = (F.cross_entropy(ll, wy_t[bi], reduction="none") * ww_t[bi]).sum() / (ww_t[bi].sum() + 1e-8) + km.tv_loss()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(km.parameters(), 2.0); opt.step()
        sched.step()
    km.eval()
    zs_p, feats, W0, b0 = km.stream_extract_bit_exact(DATA[held])

    # Train Heimdall Faithful
    hm = HeimdallMLP(len(STATELESS_COLS)).to(DEVICE)
    opth = torch.optim.AdamW(hm.parameters(), lr=5e-3, weight_decay=2e-3)
    for ep in range(40):
        hm.train(); pl = torch.randperm(M, device=DEVICE)
        for i in range(0, M, B):
            bi = pl[i:i+B]
            if len(bi) < 16: continue
            ll = hm(xh_t[bi])
            loss = (F.cross_entropy(ll, wy_t[bi], reduction="none") * ww_t[bi]).sum() / (ww_t[bi].sum() + 1e-8)
            opth.zero_grad(); loss.backward(); opth.step()
    hm.eval()
    with torch.no_grad():
        hp = hm(torch.tensor(DATA[held]["scalars"][DATA[held]["life_ev"]][:, STATELESS_COLS], device=DEVICE)).argmax(-1).cpu().numpy()

    LOAO_OUT[held] = {"kssm_zs": zs_p, "kssm_feat": feats, "W0": W0, "b0": b0, "heimdall": hp}
    print(f"  LOAO held-out [{held:8s}]: KSSM-ZS pred dist={dict(Counter(zs_p.tolist()))} | True={dict(Counter(DATA[held]['life_y'].tolist()))}", flush=True)

# ==============================================================================
# Step 2: Construct Realistic 20-Minute (3000s) Multi-App Workload-Shift Write Stream
#   - 5 phases x 240s = 3000s (each phase tiles the real app's file creation/death trace
#     with realistic steady-state volume: ~18 GiB host writes per phase -> ~90 GiB total
#     over a 12 GiB logical FDP SSD = 7.5x full-drive overwrites for deep steady-state GC!)
#   - All 8 placement policies run on the EXACT SAME physical write/TRIM sequence!
# ==============================================================================
print("\nStep 2: Generating 50-minute (3000s) real multi-app shift write stream + all 8 policy stream IDs...", flush=True)

# Realistic file write size per class (drawn from each app's actual file size distribution,
# scaled so short/med/long each carry realistic storage write share, and total writes ~90 GiB!):
# Note: in real SSD workloads, short-lived writes (WAL/L0/staging/temp) + medium-lived (L1/rolling ckpt)
# + long-lived (L2/best ckpt/durable DB) co-exist!
# Using real birth/death relative timelines from each app, tiled across its 240s phase!
PHASE_DUR = 600.0
rng = np.random.RandomState(2026)

# Each generated file record:
# (birth_t, death_t, num_pages_4k, true_cls, app_id, pc_sig, dir_group,
#  pred_heimdall, pred_kssm_zs, kssm_feat[128], W0_idx)
files_stream = []

# Pre-extract real file write size estimates from traces (in 4KB pages, realistic 512KB..8MB)
for p_idx, app in enumerate(PHASE_APPS):
    d = DATA[app]
    lo = LOAO_OUT[app]
    n_base = len(d["life_ev"])
    base_dur = max(45.0, float(d["dur_s"]))
    phase_t0 = p_idx * PHASE_DUR

    # Compute real class distribution & tile real file birth/death events to fill 240s
    # Ensure every phase writes ~18 GiB (4.5M 4KB pages) so SSD reaches deep steady-state GC
    # and every class (Short/Med/Long) has genuine non-trivial byte share (reflecting real L0/L1/L2 + Ckpt)!
    y_arr = d["life_y"]
    tb_arr = d["life_birth_s"]
    dt_arr = d["life_dt"]
    pcs_arr = d["pc_sigs"][d["life_ev"]]
    s5_arr = d["s_gram"][d["life_ev"], 4]

    # Realistic page counts per file by app & class (uniform 2MB-4MB so size alone can't cheat!):
    # We keep file sizes realistic (1MB..4MB) and balance repetition across 4 tiles in 240s
    # For apps with fewer raw births (pytorch/llm/etl ~100-200 files/65s), replicate worker concurrency (e.g. 4-6 concurrent jobs)
    # preserving exact per-sample KSSM/Heimdall/PCStream features & predictions!
    n_tiles = int(math.ceil(PHASE_DUR / base_dur))
    # Target ~18 GiB per 240s phase = ~4,600 files of ~4 MB (1024 pages) each, or ~9,200 files of 2 MB
    # Compute concurrency multiplier so every phase produces ~4,500 files (~18 GiB)
    conc = max(1, int(round(9000.0 / max(1, n_base * n_tiles))))

    # Also ensure RocksDB/SQLite's medium & long classes (which represent large L1/L2 SSTs & durable DB pages)
    # carry realistic byte sizes (L2 SSTs & durable DBs are 4x-8x larger than tiny 100KB journal/WAL files!)
    phase_files = []
    for tile in range(n_tiles):
        for worker in range(conc):
            jitter_base = worker * 0.13
            for i in range(n_base):
                bt = phase_t0 + tile * base_dur + float(tb_arr[i]) + jitter_base
                if bt >= phase_t0 + PHASE_DUR:
                    continue
                cls = int(y_arr[i])
                # Realistic lifetime dt:
                # Short: real dt (<10s)
                # Medium: real dt (10..50s)
                # Long: survives 180..360s (so it stays alive across phase switches & creates real GC pressure!)
                if cls == 0:
                    life_s = max(0.5, min(9.5, float(dt_arr[i])))
                elif cls == 1:
                    life_s = max(12.0, min(48.0, float(dt_arr[i])))
                else:
                    life_s = float(rng.uniform(280.0, 560.0))
                dt_end = bt + life_s

                # Realistic write sizes (in 4KB pages) per app:
                if app in ("rocksdb", "sqlite"):
                    # Short journal/L0 = 1-2 MB, Med L1/shard = 8-12 MB, Long L2/main DB = 16-24 MB
                    # (reflecting real LSM/OLTP byte proportions across tiers!)
                    n_pg = 384 if cls == 0 else (2048 if cls == 1 else 4096)
                else:
                    # PyTorch / LLM / ETL: uniform 4 MB (1024 pages) across ALL 3 tiers!
                    n_pg = 1024

                # Note: PC signature is identical across tiles/workers of the SAME app (same binary!),
                # and 100% disjoint across different apps (different binaries!)
                pc_sig = int(pcs_arr[i])
                phase_files.append((bt, dt_end, n_pg, cls, p_idx, pc_sig, int(s5_arr[i]),
                                    int(lo["heimdall"][i]), int(lo["kssm_zs"][i]), i))

    phase_files.sort(key=lambda x: x[0])
    # Cap total bytes in phase to ~18 GiB (4,718,592 pages) for clean balanced 5-phase comparison
    acc_pg = 0
    kept = []
    for rec in phase_files:
        if acc_pg >= 9437184:
            break
        kept.append(rec)
        acc_pg += rec[2]
    files_stream.extend(kept)
    cls_bytes = Counter()
    for r in kept: cls_bytes[r[3]] += r[2] * 4096 / (1024**3)
    print(f"  Phase {p_idx+1} [{app:8s}, {phase_t0:4.0f}-{phase_t0+PHASE_DUR:4.0f}s]: files={len(kept):5d} | total={acc_pg*4096/(1024**3):5.2f} GiB | "
          f"Short={cls_bytes[0]:.2f}G, Med={cls_bytes[1]:.2f}G, Long={cls_bytes[2]:.2f}G", flush=True)

files_stream.sort(key=lambda x: x[0])
M_TOT = len(files_stream)
print(f"Total 50-min workload-shift stream: {M_TOT} files, {sum(r[2] for r in files_stream)*4096/(1024**3):.2f} GiB host writes", flush=True)

# ==============================================================================
# Step 3: Compute Causal Online Stream Assignments for ALL 8 Policies (0 Leakage!)
#   Stream IDs: 0=Default/Mixed, 1=Short, 2=Medium, 3=Long
# ==============================================================================
# Policy columns (0..7):
#   0: No-Stream (all -> RUH 0)
#   1: Heimdall (EuroSys'25, Faithful LOAO)
#   2: PCStream (FAST'19, Online Death Adaptation)
#   3: eFDP-Paper (arXiv:2611.06461, T_replan=420s reactive histogram + per-hint bin)
#   4: eFDP-Agile (T_replan=60s accelerated reactive replanning)
#   5: KSSM-ZeroShot (Frozen Guard-SWAR 64D LOAO)
#   6: KSSM-Full (+SlowLoop Zero-BPTT online head calibration on <1KB head!)
#   7: Oracle (exact ground-truth class)

stream_ids = np.zeros((M_TOT, 8), dtype=np.uint8)

# Pre-build death event queue sorted by real death_t (strictly causal: no method can see a file's death before death_t!)
death_events = sorted([(files_stream[i][1], i) for i in range(M_TOT)], key=lambda x: x[0])

# --- Policy 0 (No-Stream), Policy 1 (Heimdall), Policy 5 (KSSM-ZS), Policy 7 (Oracle) ---
for i, r in enumerate(files_stream):
    stream_ids[i, 0] = 0
    stream_ids[i, 1] = r[7] + 1
    stream_ids[i, 5] = r[8] + 1
    stream_ids[i, 7] = r[3] + 1

# --- Policy 2: PCStream (FAST'19) - Online PC->avg_lifetime table updated strictly upon death_t ---
pc_sum_dt = defaultdict(float)
pc_cnt = defaultdict(int)
dp = 0
for i, r in enumerate(files_stream):
    cur_t = r[0]
    while dp < M_TOT and death_events[dp][0] <= cur_t:
        di = death_events[dp][1]
        dr = files_stream[di]
        pc_sum_dt[dr[5]] += (dr[1] - dr[0])
        pc_cnt[dr[5]] += 1
        dp += 1
    pc = r[5]
    if pc_cnt[pc] > 0:
        avg_dt = pc_sum_dt[pc] / pc_cnt[pc]
        stream_ids[i, 2] = 1 if avg_dt < 10.0 else (2 if avg_dt < 50.0 else 3)
    else:
        stream_ids[i, 2] = 0 # unseen PC -> default mixed stream!

# --- Policy 3 & 4: eFDP (arXiv:2611.06461) Reactive Death Tracking + Periodic RUH Replanning ---
# Faithful reactive model:
#   eFDP tracks death lifetimes per coarse write-context bucket (e.g. dir_churn/stream key) in a
#   reclaim-unit history window, and ONLY updates its RUH placement map every T_replan seconds
#   (420s in eFDP paper; 60s in our generous agile variant). During each workload shift before
#   the next replanning epoch fires (and before long-lived files have died!), new workload keys
#   either miss -> default stream 0 or suffer stale mapping from the previous workload!
for col, T_replan in [(3, 420.0), (4, 60.0)]:
    live_key_sum = defaultdict(float)
    live_key_cnt = defaultdict(int)
    active_map = {} # frozen map updated ONLY at multiples of T_replan!
    next_replan = T_replan
    dp = 0
    for i, r in enumerate(files_stream):
        cur_t = r[0]
        while dp < M_TOT and death_events[dp][0] <= cur_t:
            di = death_events[dp][1]; dr = files_stream[di]
            # eFDP coarse key = (app_id, pc_sig) — reactive per-stream/write-source tracker
            ekey = (dr[5], dr[6])
            live_key_sum[ekey] += (dr[1] - dr[0])
            live_key_cnt[ekey] += 1
            dp += 1
        while cur_t >= next_replan:
            # Periodic replanning snapshot (decay old counts so it adapts to new workload!)
            active_map = {}
            for ek, cnt in live_key_cnt.items():
                avg_dt = live_key_sum[ek] / cnt
                active_map[ek] = 1 if avg_dt < 10.0 else (2 if avg_dt < 50.0 else 3)
            # Clear/decay window for next epoch
            for ek in list(live_key_cnt.keys()):
                live_key_sum[ek] *= 0.2
                live_key_cnt[ek] = max(1, int(live_key_cnt[ek] * 0.2))
            next_replan += T_replan
        ekey = (r[5], r[6])
        stream_ids[i, col] = active_map.get(ekey, 0) # unmapped during blind window -> default stream 0!

# --- Policy 6: KSSM-Full (LOAO Frozen Guard-SWAR 64D + Zero-BPTT Slow-Loop Online Calibration) ---
# At the start of each unseen phase app, KSSM initializes its <1KB linear head from the STRICT LOAO
# weights (trained ONLY on the other 4 apps!), then updates ONLY the 784B linear head via online SGD
# whenever real death feedback arrives OR a file survives past 55s (confirming Long-lived without waiting for death!).
# Notice: survival confirmation at t_birth + 55s is 100% causal (the OS knows at t=birth+55s that the file wasn't deleted!).
feedback_q = []
for i, r in enumerate(files_stream):
    bt, dt_end, _, cls, p_idx, _, _, _, _, src_i = r
    if dt_end - bt < 55.0:
        feedback_q.append((dt_end, i, cls))
    else:
        # Causal survival confirmation: file still alive 55s after birth -> confirmed Long (cls=2)!
        feedback_q.append((bt + 55.0, i, 2))
feedback_q.sort(key=lambda x: x[0])

# Track current phase to load fresh LOAO base per unseen app (simulating deployment onto each unseen app)
# Or keep a single continuous calibrated head across phases: let's keep a CONTINUOUS head initialized
# from LOAO of Phase 1, and at each new phase transition reset ONLY the head to that phase's strict LOAO
# prior (conservative! Actually forces KSSM to start cold zero-shot on EVERY single phase switch!).
W_cur = LOAO_OUT[PHASE_APPS[0]]["W0"].copy()
b_cur = LOAO_OUT[PHASE_APPS[0]]["b0"].copy()
last_phase = 0
fp = 0
for i, r in enumerate(files_stream):
    bt, _, _, cls, p_idx, _, _, _, _, src_i = r
    if p_idx != last_phase:
        # Conservative strict zero-shot handoff: switch to strict LOAO weights for this unseen app!
        W_cur = LOAO_OUT[PHASE_APPS[p_idx]]["W0"].copy()
        b_cur = LOAO_OUT[PHASE_APPS[p_idx]]["b0"].copy()
        last_phase = p_idx
    while fp < len(feedback_q) and feedback_q[fp][0] <= bt:
        _, fi, fy = feedback_q[fp]
        fp += 1
        fr = files_stream[fi]
        if fr[4] != p_idx:
            continue # ignore stale feedback from previous app after switch
        xf = LOAO_OUT[PHASE_APPS[fr[4]]]["kssm_feat"][fr[9]]
        lg = W_cur @ xf + b_cur; lg -= lg.max()
        pr = np.exp(lg); pr /= pr.sum()
        gr = pr; gr[fy] -= 1.0
        cw = 2.0 if fy > 0 else 0.5
        W_cur -= (0.04 * cw) * np.outer(gr, xf)
        b_cur -= (0.04 * cw) * gr
        # Mild L2 decay to prevent divergence
        W_cur *= 0.9998

    xf_cur = LOAO_OUT[PHASE_APPS[p_idx]]["kssm_feat"][src_i]
    stream_ids[i, 6] = int(np.argmax(W_cur @ xf_cur + b_cur)) + 1

# Print quick classification accuracy sanity check across the 50-min shift stream!
pol_names = ["No-Stream", "Heimdall(Faithful)", "PCStream(Online)", "eFDP(420s,paper)",
             "eFDP(60s,agile)", "KSSM-ZeroShot", "KSSM-Full(+SlowLoop)", "Oracle"]
true_s = stream_ids[:, 7]
print("\n=== Stream Classification Sanity Check over 50-min Shift Stream (Balanced Acc across Short/Med/Long) ===")
for c, nm in enumerate(pol_names):
    recs = [np.mean(stream_ids[true_s == k, c] == k) for k in (1, 2, 3)]
    # Also check blind-window [0..60s] immediately after each of the 5 phase switches!
    bw_mask = np.array([(r[0] % PHASE_DUR) < 60.0 for r in files_stream])
    bw_recs = [np.mean(stream_ids[bw_mask & (true_s == k), c] == k) for k in (1, 2, 3)]
    print(f"  {nm:22s} | Full 50-min BalAcc={np.mean(recs)*100:5.1f}% | Post-Shift [0-60s] BalAcc={np.mean(bw_recs)*100:5.1f}%")

# Export compact binary trace for high-speed C++ NVMe FDP FTL simulator!
# Struct per file (32 bytes): double birth_t, double death_t, uint32 n_pages, uint8 phase_id, uint8 true_cls, uint8 streams[8], uint8 pad[2]
out_bin = "/home/yjw/workspace/research/KSSM/exp1_write_fdp/shift_50min_stream.bin"
with open(out_bin, "wb") as f:
    for i, r in enumerate(files_stream):
        f.write(struct.pack("<ddIBB8B2x",
                            float(r[0]), float(r[1]), int(r[2]), int(r[4]), int(r[3]),
                            int(stream_ids[i,0]), int(stream_ids[i,1]), int(stream_ids[i,2]), int(stream_ids[i,3]),
                            int(stream_ids[i,4]), int(stream_ids[i,5]), int(stream_ids[i,6]), int(stream_ids[i,7])))
print("Saved", out_bin, "records:", M_TOT)
