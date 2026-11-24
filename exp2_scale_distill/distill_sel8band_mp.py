import os, sys, time, math, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "3"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42); np.random.seed(42)

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

def bal_acc(y_true, y_pred, n_cls=3):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    recs = [np.mean(y_pred[y_true == c] == c) for c in range(n_cls) if np.any(y_true == c)]
    return float(np.mean(recs)) * 100.0

def parse_mp_trace(mode):
    exp = "/home/yjw/workspace/research/KSSM/exp0b_selective/traces"
    pid = int(open(f"{exp}/mp_{mode}.pid").read().strip())
    arr = np.fromfile(f"{exp}/mp_{mode}.bin", dtype=DTYPE)
    sub = arr[arr["pid"] == pid]
    sub = sub[np.argsort(sub["ts_ns"], kind="mergesort")]
    N = len(sub)
    meta_by_ino = defaultdict(list)
    with open(f"{exp}/mp_{mode}_meta.csv") as f:
        for row in csv.DictReader(f):
            meta_by_ino[int(row["ino"])].append((int(row["phase_cls"]), int(row["k_noise"])))
    ops = sub["op"]; inos = sub["ino"]; tids = sub["tid"]; offs = sub["offset"]
    cnts = sub["count"]; fszs = sub["file_size"]; strides = sub["stride"]
    flags = sub["flags"]; ustks = sub["u_stack_id"]; kstks = sub["k_stack_id"]
    churns = sub["dir_churn"]; is_seqs = sub["is_seq"]

    ino_gen = defaultdict(int); inst_map = {}; nxt_inst = 1
    inst_arr = np.zeros(N, dtype=np.int32)
    tid_map = {}; tid_arr = np.zeros(N, dtype=np.int32)
    meta_ptr = defaultdict(int); target_ev, target_y, target_k = [], [], []
    for i in range(N):
        t = int(tids[i])
        if t not in tid_map: tid_map[t] = len(tid_map)
        tid_arr[i] = tid_map[t]
        op = int(ops[i]); ino = int(inos[i]); key = (ino, ino_gen[ino])
        if op == 2:
            if key in inst_map: ino_gen[ino] += 1; key = (ino, ino_gen[ino])
            inst_map[key] = nxt_inst; nxt_inst += 1; inst_arr[i] = inst_map[key]
            if ino in meta_by_ino and meta_ptr[ino] < len(meta_by_ino[ino]):
                pcls, kn = meta_by_ino[ino][meta_ptr[ino]]; meta_ptr[ino] += 1
                target_ev.append(i); target_y.append(pcls); target_k.append(kn)
        elif op in (7, 10):
            if key not in inst_map: inst_map[key] = nxt_inst; nxt_inst += 1
            inst_arr[i] = inst_map[key]; ino_gen[ino] += 1
        else:
            if key not in inst_map: inst_map[key] = nxt_inst; nxt_inst += 1
            inst_arr[i] = inst_map[key]

    s_gram = np.zeros((N, 5), dtype=np.uint8)
    num_t = len(tid_map)
    rings = np.full((num_t, 16), -1, dtype=np.int64); heads = np.zeros(num_t, dtype=np.int32)
    tr = np.zeros(num_t, dtype=np.int32); tw = np.zeros(num_t, dtype=np.int32)
    t_li = np.full(num_t, -1, dtype=np.int32); t_lo = np.zeros(num_t, dtype=np.int64); t_lc = np.zeros(num_t, dtype=np.int64)
    ema_ch = 2.0
    for i in range(N):
        op = int(ops[i]); tid = int(tid_arr[i]); iid = int(inst_arr[i])
        fl = int(flags[i]); off = int(offs[i]); cnt = int(cnts[i]); fsz = int(fszs[i]); ch = int(churns[i])
        if op in (4, 5, 9):
            if t_li[tid] == iid:
                st = off - int(t_lo[tid])
                seq = 1 if (st > 0 and st <= max(131072, int(t_lc[tid])*2)) else (1 if (st == 0 and int(is_seqs[i])) else 0)
            else:
                st = int(strides[i]); seq = int(is_seqs[i])
            t_li[tid] = iid; t_lo[tid] = off; t_lc[tid] = cnt
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
        else: s1 = 16
        if op in (2,3,5,6,7,8,10): tw[tid] += 1
        else: tr[tid] += 1
        if tr[tid] + tw[tid] > 256: tr[tid] >>= 1; tw[tid] >>= 1
        rw_q = min(3, (tw[tid] * 4) // (tr[tid] + tw[tid] + 1))
        stk = int(ustks[i]) if int(ustks[i]) >= 0 else (int(kstks[i]) + 100000)
        hd = int(heads[tid]); rg = rings[tid]; dist = 15
        for d in range(16):
            if rg[(hd - d) & 15] == stk: dist = d; break
        nh = (hd + 1) & 15; rg[nh] = stk; heads[tid] = nh
        s2 = ((rw_q << 4) | dist) & 63
        sz_b = fast_size_bucket(fsz)
        reg = 0 if off < 65536 else (3 if off + cnt >= fsz else (1 if off < (fsz >> 1) else 2))
        s3 = ((sz_b << 2) | reg) & 31
        req_b = (1 if cnt >= 16384 else 0) + (1 if cnt >= 65536 else 0) + (1 if cnt >= 262144 else 0)
        str_b = 0 if seq else (1 if st < 0 else (3 if st == 0 else (5 if st <= 65536 else 7)))
        s4 = ((req_b << 3) | str_b) & 31
        ch_log = math.log2(ch + 1.0)
        if op in (2, 7, 8): ema_ch = 0.95 * ema_ch + 0.05 * ch_log
        rel_b = max(0, min(15, int(round((ch_log - ema_ch + 4.0) * 1.875))))
        s5 = (((0 if ch <= 4 else 1) << 4) | rel_b) & 31
        s_gram[i] = (s1, s2, s3, s4, s5)
    return {"N": N, "tid": tid_arr, "inst": inst_arr, "s_gram": s_gram,
            "ev": np.array(target_ev, dtype=np.int32), "y": np.array(target_y, dtype=np.int64), "k": np.array(target_k, dtype=np.int32)}

t0 = time.time()
mp_tr = parse_mp_trace("train")
mp_te = parse_mp_trace("test")

def build_seqs(d, L=48, max_n=4500):
    rng = np.random.RandomState(42)
    idx = np.sort(rng.choice(len(d["ev"]), size=min(max_n, len(d["ev"])), replace=False))
    evs = d["ev"][idx]; ys = d["y"][idx]; M = len(evs)
    seq_s = np.zeros((M, L, 5), dtype=np.int64)
    seq_mi = np.zeros((M, L), dtype=np.float32)
    seq_v  = np.zeros((M, L), dtype=np.float32)
    sg = d["s_gram"]; inst = d["inst"]
    for m in range(M):
        ei = int(evs[m]); tgt_i = int(inst[ei])
        lo = max(0, ei - L + 1); seg = np.arange(lo, ei + 1); off = L - len(seg)
        seq_s[m, off:] = sg[seg]; seq_v[m, off:] = 1.0
        seq_mi[m, off:] = (inst[seg] == tgt_i).astype(np.float32)
    return [torch.tensor(x, device=DEVICE) for x in (seq_s, seq_mi, seq_v, ys)]

tr_s, tr_mi, tr_v, tr_y = build_seqs(mp_tr, L=48, max_n=4500)

# Teacher: Continuous 64D Selective KSSM (Float64D)
class TeacherSelectiveFloat64D(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(192, 64)
        self.gd  = nn.Embedding(192, 64)
        self.gc  = nn.Embedding(192, 64)
        self.gdb = nn.Parameter(torch.full((64,), -0.5))
        nn.init.normal_(self.emb.weight, 0, 0.15)
        nn.init.normal_(self.gd.weight, 0, 0.1)
        nn.init.normal_(self.gc.weight, 0, 0.1)
        self.head = nn.Linear(128, 3)

    def forward_seq(self, seq_s, seq_mi, seq_v):
        B, L, _ = seq_s.shape
        offs = torch.tensor([0, 32, 96, 128, 160], device=seq_s.device)
        g = seq_s + offs
        x_all = torch.tanh(self.emb(g).sum(-2)) * 2.0
        d_all = torch.sigmoid(self.gd(g).sum(-2) + self.gdb)
        c_all = torch.tanh(self.gc(g).sum(-2)) * 0.35 * d_all
        h_t = torch.zeros(B, 64, device=seq_s.device)
        h_i = torch.zeros(B, 64, device=seq_s.device)
        for t in range(L):
            v = seq_v[:, t:t+1]; mi = seq_mi[:, t:t+1]
            dt = d_all[:, t, :]; ct = c_all[:, t, :]; xt = x_all[:, t, :]
            ht_n = (1.0 - dt) * h_t - ct * h_i + dt * xt
            hi_n = (1.0 - dt) * h_i + ct * h_t + dt * xt
            h_t = h_t * (1.0 - v) + ht_n * v
            vi = v * mi
            h_i = h_i * (1.0 - vi) + hi_n * vi
        feat = torch.cat([h_i, h_t], dim=-1)
        return self.head(feat), feat, d_all

# Student: 8-Band Independent Selective Guard-SWAR 64D (exact twin of kssm_sel8band.bpf.c, 38.0-50.3ns!)
# Note: supports BOTH Subspace (2-slot) AND Shared (5-slot) input tables for comparison!
class StudentSel8BandSWAR(nn.Module):
    def __init__(self, n_bands=8):
        super().__init__()
        self.n_bands = n_bands
        self.band_dim = 64 // n_bands
        self.emb = nn.Embedding(192, 64)
        nn.init.normal_(self.emb.weight, 0.0, 0.15)
        # Gate table indexed by (s1<<5)|s4 -> n_bands x 4 regimes (0=HOLD, 1=SLOW, 2=FAST, 3=RESET)
        self.g_s1   = nn.Parameter(torch.zeros(32, n_bands, 4))
        self.g_s4   = nn.Parameter(torch.zeros(32, n_bands, 4))
        self.g_pair = nn.Parameter(torch.zeros(1024, n_bands, 4))
        self.head   = nn.Linear(128, 3)

        self.register_buffer("a_i_vals", torch.tensor([1.0, 0.875, 0.5, 0.0]))
        self.register_buffer("g_i_vals", torch.tensor([0.0, 0.0625, 0.25, 0.5]))
        self.register_buffer("a_t_vals", torch.tensor([1.0, 0.875, 0.5, 0.75]))
        self.register_buffer("g_t_vals", torch.tensor([0.0, 0.0625, 0.25, 0.0]))
        self.register_buffer("in_scale", torch.tensor([0.0, 1.0, 1.0, 1.0]))
        self.register_buffer("delta_vals", torch.tensor([0.0, 0.125, 0.5, 1.0]))

    def get_regime_logits(self, s1, s4):
        gkey = (s1 << 5) | s4
        return self.g_s1[s1] + self.g_s4[s4] + 0.5 * self.g_pair[gkey]

    def forward_seq(self, seq_s, seq_mi, seq_v, tau=1.0, hard=False):
        B, L, _ = seq_s.shape
        offs = torch.tensor([0, 32, 96, 128, 160], device=seq_s.device)
        # Subspace-factorized embeddings matching kssm_sel8band.bpf.c (4 groups of 16 dims, split into 8x8 bands!)
        w = torch.tanh(self.emb(seq_s + offs)) * 0.75
        b_all = torch.cat([
            w[:, :, 0, 0:16]  + w[:, :, 1, 0:16],
            w[:, :, 0, 16:32] + w[:, :, 2, 16:32],
            w[:, :, 0, 32:48] + w[:, :, 3, 32:48],
            w[:, :, 0, 48:64] + w[:, :, 4, 48:64],
        ], dim=-1)
        logits = self.get_regime_logits(seq_s[:, :, 0], seq_s[:, :, 3]) # [B, L, n_bands, 4]
        probs = F.softmax(logits / tau, dim=-1)
        if hard:
            idx = probs.argmax(dim=-1, keepdim=True)
            oh = torch.zeros_like(probs).scatter_(-1, idx, 1.0)
            probs = (oh - probs).detach() + probs

        ai  = (probs * self.a_i_vals).sum(-1).repeat_interleave(self.band_dim, dim=-1)
        gi  = (probs * self.g_i_vals).sum(-1).repeat_interleave(self.band_dim, dim=-1)
        at  = (probs * self.a_t_vals).sum(-1).repeat_interleave(self.band_dim, dim=-1)
        gt  = (probs * self.g_t_vals).sum(-1).repeat_interleave(self.band_dim, dim=-1)
        is_ = (probs * self.in_scale).sum(-1).repeat_interleave(self.band_dim, dim=-1)
        d_eff = (probs * self.delta_vals).sum(-1) # [B, L, n_bands] for gate distillation!

        h_t = torch.zeros(B, 64, device=seq_s.device)
        h_i = torch.zeros(B, 64, device=seq_s.device)
        for t in range(L):
            v = seq_v[:, t:t+1]; mi = seq_mi[:, t:t+1]
            bt = b_all[:, t, :] * is_[:, t, :]
            hi_n = ai[:, t, :] * h_i + gi[:, t, :] * h_t + bt
            ht_n = at[:, t, :] * h_t - gt[:, t, :] * h_i + bt
            h_t = h_t * (1.0 - v) + ht_n * v
            vi = v * mi
            h_i = h_i * (1.0 - vi) + hi_n * vi
        feat = torch.cat([h_i, h_t], dim=-1)
        return self.head(feat), feat, d_eff

    @torch.no_grad()
    def stream_eval_bit_exact(self, d):
        w_f = (torch.tanh(self.emb.weight) * 0.75).detach().cpu().numpy()
        emb_u16 = (np.round(w_f * 256.0).astype(np.int32) + 2048).astype(np.uint16)
        s1_g = torch.arange(32, device=DEVICE).repeat_interleave(32)
        s4_g = torch.arange(32, device=DEVICE).repeat(32)
        reg_tbl = self.get_regime_logits(s1_g, s4_g).argmax(dim=-1).cpu().numpy().astype(np.uint8) # [1024, n_bands]
        wl_q = np.round(self.head.weight.detach().cpu().numpy() * 256.0).astype(np.int32)
        bl_q = np.round(self.head.bias.detach().cpu().numpy() * 65536.0).astype(np.int32)

        N = d["N"]; tids = d["tid"]; insts = d["inst"]; sg = d["s_gram"]
        H_i = np.full((int(insts.max()) + 1, 64), 16384, dtype=np.int32)
        H_t = np.full((int(tids.max()) + 1, 64), 16384, dtype=np.int32)
        t_set = {int(e): i for i, e in enumerate(d["ev"])}
        logits = np.zeros((len(d["ev"]), 3), dtype=np.float32)
        rcnt = Counter(); min_s, max_s = 16384, 16384

        s0 = sg[:, 0].astype(np.int32); s1 = 32 + sg[:, 1].astype(np.int32)
        s2 = 96 + sg[:, 2].astype(np.int32); s3 = 128 + sg[:, 3].astype(np.int32); s4 = 160 + sg[:, 4].astype(np.int32)
        gkeys = (sg[:, 0].astype(np.int32) << 5) | sg[:, 3].astype(np.int32)
        bd = self.band_dim

        for e in range(N):
            tid = int(tids[e]); iid = int(insts[e])
            regs = reg_tbl[int(gkeys[e])]
            oi = H_i[iid]; ot = H_t[tid]
            i0 = int(s0[e])
            slots = (int(s1[e]), int(s2[e]), int(s3[e]), int(s4[e]))
            for b in range(self.n_bands):
                r = int(regs[b]); rcnt[r] += 1
                if r == 0: continue
                lo = b * bd; hi = lo + bd
                sb = slots[lo // 16]
                bsum = emb_u16[i0, lo:hi].astype(np.int32) + emb_u16[sb, lo:hi]
                oib = oi[lo:hi]; otb = ot[lo:hi]
                if r == 1:
                    oi[lo:hi] = oib - (oib >> 3) + (otb >> 4) + bsum - 3072
                    ot[lo:hi] = otb - (otb >> 3) - (oib >> 4) + bsum - 1024
                elif r == 2:
                    oi[lo:hi] = oib - (oib >> 1) + (otb >> 2) + bsum
                    ot[lo:hi] = otb - (otb >> 1) - (oib >> 2) + bsum + 8192
                else:
                    oi[lo:hi] = (otb >> 1) + bsum + 4096
                    ot[lo:hi] = otb - (otb >> 2) + bsum
            if (e & 511) == 0:
                mn = min(int(oi.min()), int(ot.min())); mx = max(int(oi.max()), int(ot.max()))
                if mn < min_s: min_s = mn
                if mx > max_s: max_s = mx
            if e in t_set:
                fq = np.concatenate([oi - 16384, ot - 16384])
                logits[t_set[e]] = (wl_q @ fq + bl_q) / 65536.0
        return logits, dict(rcnt), (min_s, max_s)

# 1. Train Teacher (True Selective Float64D)
print("Training Teacher (True Selective Float64D)...", flush=True)
teacher = TeacherSelectiveFloat64D().to(DEVICE)
opt_t = torch.optim.AdamW(teacher.parameters(), lr=6e-3, weight_decay=1e-3)
sch_t = torch.optim.lr_scheduler.CosineAnnealingLR(opt_t, T_max=32)
B = 256; M = len(tr_y)
for ep in range(32):
    teacher.train(); perm = torch.randperm(M, device=DEVICE)
    for i in range(0, M, B):
        bi = perm[i:i+B]
        if len(bi) < 32: continue
        ll, _, _ = teacher.forward_seq(tr_s[bi], tr_mi[bi], tr_v[bi])
        loss = F.cross_entropy(ll, tr_y[bi])
        opt_t.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(teacher.parameters(), 2.0); opt_t.step()
    sch_t.step()
teacher.eval()

# Precompute teacher soft targets (logits, hidden feats, per-band mean selective gates) on training set!
with torch.no_grad():
    t_logits_all, t_feat_all, t_delta_all = [], [], []
    for i in range(0, M, 512):
        tl, tf, td = teacher.forward_seq(tr_s[i:i+512], tr_mi[i:i+512], tr_v[i:i+512])
        t_logits_all.append(tl); t_feat_all.append(tf); t_delta_all.append(td)
    t_logits = torch.cat(t_logits_all)
    t_feats  = torch.cat(t_feat_all)
    t_deltas = torch.cat(t_delta_all) # [M, L, 64]

def run_student(name, n_bands, use_distill):
    stu = StudentSel8BandSWAR(n_bands=n_bands).to(DEVICE)
    opt = torch.optim.AdamW(stu.parameters(), lr=8e-3, weight_decay=1e-3)
    epochs = 40; T_kl = 2.0
    # Pool teacher delta to n_bands target gates: [M, L, n_bands]
    td_band = t_deltas.view(M, 48, n_bands, 64 // n_bands).mean(dim=-1)
    for ep in range(epochs):
        tau = max(0.06, 2.0 * (0.88 ** ep))
        hard = (ep >= 30) # final 10 epochs STE-hard fine-tuning for zero quantization shock!
        stu.train(); perm = torch.randperm(M, device=DEVICE)
        for i in range(0, M, B):
            bi = perm[i:i+B]
            if len(bi) < 32: continue
            sl, sf, sd = stu.forward_seq(tr_s[bi], tr_mi[bi], tr_v[bi], tau=tau, hard=hard)
            loss_ce = F.cross_entropy(sl, tr_y[bi])
            if use_distill:
                # 1. Soft-logit KL distillation
                loss_kl = F.kl_div(F.log_softmax(sl / T_kl, dim=-1),
                                   F.softmax(t_logits[bi] / T_kl, dim=-1), reduction="batchmean") * (T_kl * T_kl)
                # 2. Selective Gate Distillation (teaches Student which events are NOISE -> HOLD=0!)
                vmask = tr_v[bi].unsqueeze(-1)
                loss_gate = (((sd - td_band[bi]) ** 2) * vmask).sum() / (vmask.sum() * n_bands + 1e-8)
                # 3. Cosine hidden representation alignment
                loss_hid = (1.0 - F.cosine_similarity(sf, t_feats[bi], dim=-1)).mean()
                loss = 0.5 * loss_ce + 0.5 * loss_kl + 2.0 * loss_gate + 0.25 * loss_hid
            else:
                loss = loss_ce
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(stu.parameters(), 2.0)
            opt.step()
    stu.eval()
    logits, rcnt, bnd = stu.stream_eval_bit_exact(mp_te)
    preds = np.argmax(logits, axis=-1)
    y_te = mp_te["y"]; k_te = mp_te["k"]
    accs = {kv: bal_acc(y_te[k_te == kv], preds[k_te == kv]) for kv in [0, 5, 15, 35]}
    ov = bal_acc(y_te, preds)
    hold_pct = 100.0 * rcnt.get(0, 0) / max(1, sum(rcnt.values()))
    print(f"  {name:46s} | K=0:{accs[0]:5.2f}% | K=5:{accs[5]:5.2f}% | K=15:{accs[15]:5.2f}% | K=35:{accs[35]:5.2f}% | All:{ov:5.2f}% | HOLD={hold_pct:4.1f}% SWAR={bnd}", flush=True)

print("\n=== DISTILLATION ABLATION ON BATTLEFIELD 2 (555K Real Kernel Events, OOD Heavy Noise K=35) ===", flush=True)
run_student("4-Band Selective SWAR (No Distill, 37.8ns)", 4, False)
run_student("4-Band Selective SWAR (Distilled,  37.8ns)", 4, True)
run_student("8-Band Selective SWAR (No Distill, 38.0ns)", 8, False)
run_student("8-Band Selective SWAR (Distilled,  38.0ns)", 8, True)
print(f"Done in {time.time()-t0:.1f}s")
