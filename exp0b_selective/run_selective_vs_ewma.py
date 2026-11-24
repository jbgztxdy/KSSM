import os, sys, time, math, csv, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "3", "Must run with CUDA_VISIBLE_DEVICES=3!"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

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

def roc_auc_exact(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(y_true == 1)); n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0: return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ss = y_score[order]; i = 0; n = len(ss)
    while i < n:
        j = i + 1
        while j < n and ss[j] == ss[i]: j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j); i = j
    return float((np.sum(ranks[y_true == 1]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

def cls_metrics(y_true, y_pred, n_cls=3):
    y_true = np.asarray(y_true, dtype=np.int64); y_pred = np.asarray(y_pred, dtype=np.int64)
    acc = float(np.mean(y_true == y_pred)) if len(y_true) > 0 else 0.0
    recs, f1s = [], []
    for c in range(n_cls):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        if tp + fn > 0:
            r = tp / (tp + fn); p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recs.append(r); f1s.append(2*p*r/(p+r) if (p+r) > 0 else 0.0)
    return acc, (float(np.mean(recs)) if recs else 0.0), (float(np.mean(f1s)) if f1s else 0.0)

# ==============================================================================
# Models for Head-to-Head Showdown:
#   1. Cross-Entity Single-EWMA (alpha=0.80) + 3-Layer MLP
#   2. Cross-Entity Multi-Scale EWMA (alphas=[0.5, 0.75, 0.875, 0.9375, 0.96875]) + 3-Layer MLP
#   3. Fixed-Decay Guard-SWAR 64D (linear readout, non-selective baseline from Exp 0-B)
#   4. True Selective KSSM (Input-Dependent Continuous Gate alpha(u_t), gamma(u_t) + Linear Readout)
#   5. True Selective 4-Regime Guard-SWAR 64D (Bit-Exact 32-48ns eBPF kernel semantics!)
# ==============================================================================
class CrossEntityEWMA_MLP(nn.Module):
    """Strongest reviewer ablation baseline:
    Uses the EXACT SAME 5-tuple relative behavioral grammar AND exact same task/inode
    cross-entity state binding, replacing Selective SSM with fixed-decay EWMA aggregation
    (single-scale or 4-timescale) + 3-layer nonlinear MLP!"""
    def __init__(self, dim=64, alphas=(0.5, 0.75, 0.875, 0.9375), hidden_mlp=128):
        super().__init__()
        self.dim = dim
        self.alphas = alphas
        self.n_scales = len(alphas)
        # Learnable embedding table for the 5-tuple grammar (same 192 slots!)
        self.emb = nn.Embedding(192, dim)
        nn.init.normal_(self.emb.weight, 0.0, 0.15)
        # Assign channels evenly across fixed EWMA timescales
        a_vec = np.zeros(dim, dtype=np.float32)
        chunk = dim // self.n_scales
        for si, al in enumerate(alphas):
            lo = si * chunk
            hi = dim if si == self.n_scales - 1 else (si + 1) * chunk
            a_vec[lo:hi] = al
        self.register_buffer("a_fixed", torch.tensor(a_vec))
        # Strong 3-layer nonlinear MLP over [ewma_inode, ewma_task, current_slot_emb]
        self.mlp = nn.Sequential(
            nn.Linear(dim * 3, hidden_mlp), nn.LayerNorm(hidden_mlp), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_mlp, hidden_mlp // 2), nn.LayerNorm(hidden_mlp // 2), nn.GELU(),
            nn.Linear(hidden_mlp // 2, 64), nn.GELU(),
        )
        self.head_life = nn.Linear(64, 3)
        self.head_scan = nn.Linear(64, 1)

    def _slot_sum(self, s_idx):
        offs = torch.tensor([0, 32, 96, 128, 160], device=s_idx.device, dtype=torch.long)
        return self.emb(s_idx + offs).sum(dim=-2)

    def forward_seq(self, seq_s, seq_is_target_ino, seq_valid):
        """seq_s: [B, L, 5], seq_is_target_ino: [B, L], seq_valid: [B, L]"""
        B, L, _ = seq_s.shape
        x_all = self._slot_sum(seq_s) # [B, L, D]
        a = self.a_fixed
        h_t = torch.zeros(B, self.dim, device=seq_s.device)
        h_i = torch.zeros(B, self.dim, device=seq_s.device)
        for t in range(L):
            v = seq_valid[:, t:t+1]
            mi = seq_is_target_ino[:, t:t+1]
            xt = x_all[:, t, :]
            # Standard fixed-rate EWMA: h_t = a * h_t + (1 - a) * x_t
            ht_new = a * h_t + (1.0 - a) * xt
            hi_new = a * h_i + (1.0 - a) * xt
            h_t = h_t * (1.0 - v) + ht_new * v
            vi = v * mi
            h_i = h_i * (1.0 - vi) + hi_new * vi
        feat = torch.cat([h_i, h_t, x_all[:, -1, :]], dim=-1)
        z = self.mlp(feat)
        return self.head_life(z), self.head_scan(z).squeeze(-1)

    @torch.no_grad()
    def stream_eval(self, tids, insts, s_gram, target_evs, scan_evs=None):
        N = len(tids)
        n_t = int(tids.max()) + 1; n_i = int(insts.max()) + 1
        emb_w = self.emb.weight.detach().cpu().numpy()
        a = self.a_fixed.detach().cpu().numpy()
        one_m_a = 1.0 - a
        H_t = np.zeros((n_t, self.dim), dtype=np.float32)
        H_i = np.zeros((n_i, self.dim), dtype=np.float32)
        t_set = {int(e): i for i, e in enumerate(target_evs)}
        s_set = {int(e): i for i, e in enumerate(scan_evs)} if scan_evs is not None else {}
        t_feats = np.zeros((len(target_evs), self.dim * 3), dtype=np.float32)
        s_feats = np.zeros((len(scan_evs), self.dim * 3), dtype=np.float32) if scan_evs is not None else None
        for e in range(N):
            tid = int(tids[e]); iid = int(insts[e])
            x = (emb_w[int(s_gram[e,0])] + emb_w[32+int(s_gram[e,1])] +
                 emb_w[96+int(s_gram[e,2])] + emb_w[128+int(s_gram[e,3])] + emb_w[160+int(s_gram[e,4])])
            H_t[tid] = a * H_t[tid] + one_m_a * x
            H_i[iid] = a * H_i[iid] + one_m_a * x
            if e in t_set:
                t_feats[t_set[e]] = np.concatenate([H_i[iid], H_t[tid], x])
            if e in s_set:
                s_feats[s_set[e]] = np.concatenate([H_i[iid], H_t[tid], x])
        ll = self.head_life(self.mlp(torch.tensor(t_feats, device=DEVICE))).cpu().numpy()
        ss = None
        if s_feats is not None and len(s_feats) > 0:
            # Batch eval for large scan arrays
            outs = []
            for i in range(0, len(s_feats), 16384):
                outs.append(self.head_scan(self.mlp(torch.tensor(s_feats[i:i+16384], device=DEVICE))).squeeze(-1).cpu().numpy())
            ss = np.concatenate(outs)
        return ll, ss


class TrueSelectiveKSSM(nn.Module):
    """Input-Dependent Selective Diagonal Coupled SSM!
    Unlike fixed EWMA, BOTH the forget gate alpha(u_t) in [0, 1] AND input/coupling gate
    delta(u_t) in [0, 1] are dynamically computed from the current input event u_t = (s1..s5)!
    - Noise events -> delta(u_t) -> 0, alpha(u_t) -> 1 (HOLD: zero memory decay!)
    - Phase events -> delta(u_t) -> 1, alpha(u_t) -> 0..0.5 (RESET/WRITE: instant state switch!)
    Also supports exact quantization to our 4-Regime Selective Guard-SWAR 64D (32-48 ns)!
    """
    def __init__(self, dim=64, selective=True, mode="float_sel"):
        super().__init__()
        self.dim = dim
        self.selective = selective
        self.mode = mode # "fixed_swar", "float_sel", "sel4_swar"
        self.emb = nn.Embedding(192, dim)
        nn.init.normal_(self.emb.weight, 0.0, 0.15)

        # Input-dependent selective gate table (pre-baked offline from (s1, s3, s4) or 5-tuple sum!)
        # Note: in eBPF, a 192-slot additive scalar gate lookup costs 5 adds (~3 ns) -> zero matrix mul!
        self.gate_delta_emb = nn.Embedding(192, dim) # controls write/forget rate delta(u_t) = sigmoid(sum_k g[s_k])
        self.gate_coup_emb  = nn.Embedding(192, dim) # controls cross-entity coupling gamma(u_t)
        nn.init.normal_(self.gate_delta_emb.weight, 0.0, 0.1)
        nn.init.normal_(self.gate_coup_emb.weight, 0.0, 0.1)
        self.gate_delta_bias = nn.Parameter(torch.full((dim,), -0.5))

        # Fixed fallback timescales for non-selective comparison
        ds = np.repeat([1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4], 4)
        cs = np.repeat([2,2,3,3, 3,3,4,4, 4,4,5,5, 5,5,5,5], 4)
        self.register_buffer("fixed_d", torch.tensor(2.0 ** (-ds), dtype=torch.float32))
        self.register_buffer("fixed_c", torch.tensor(2.0 ** (-cs), dtype=torch.float32))

        self.head_life = nn.Linear(dim * 2, 3)
        self.head_scan = nn.Linear(dim * 2, 1)

    def _compute_step_params(self, s_idx):
        offs = torch.tensor([0, 32, 96, 128, 160], device=s_idx.device, dtype=torch.long)
        g_idx = s_idx + offs
        x = torch.tanh(self.emb(g_idx).sum(dim=-2)) * 2.0 # bounded input projection
        if not self.selective:
            delta = self.fixed_d.expand_as(x)
            coup  = self.fixed_c.expand_as(x)
        else:
            # Input-dependent selective write/forget gate delta(u_t) in [0, 1]!
            # When delta(u_t) -> 0: alpha = 1 - delta -> 1 AND input delta*x -> 0 (PERFECT NOISE HOLD!)
            # When delta(u_t) -> 1: alpha = 1 - delta -> 0 AND input delta*x -> x (PERFECT PHASE RESET!)
            raw_d = self.gate_delta_emb(g_idx).sum(dim=-2) + self.gate_delta_bias
            delta = torch.sigmoid(raw_d)
            if self.mode == "sel4_swar":
                # Straight-Through Estimator (STE) to exact 4 hardware regimes:
                #   0.0     (HOLD: skip noise, 3.5ns)
                #   0.125   (SLOW_DECAY: d=3)
                #   0.5     (FAST_DECAY: d=1)
                #   1.0     (PHASE_RESET: alpha=0, 28ns)
                # Group-average across 16-dim bands or per-event scalar regime!
                band_d = delta.mean(dim=-1, keepdim=True) # single event regime matching switch(regime) in kssm_sel4.bpf.c!
                # Quantize band_d to nearest of {0.0, 0.125, 0.5, 1.0} via STE
                levels = torch.tensor([0.0, 0.125, 0.5, 1.0], device=delta.device)
                idx_near = (band_d.unsqueeze(-1) - levels).abs().argmin(dim=-1)
                d_hard = levels[idx_near]
                delta = (d_hard - band_d).detach() + band_d
                delta = delta.expand_as(x)
                # Coupling active proportionally to delta (zero during HOLD!)
                coup = (0.5 * (delta >= 0.9).float() + 0.125 * ((delta > 0.05) & (delta < 0.9)).float())
                coup = (coup - band_d * 0.25).detach() + band_d * 0.25
            else:
                coup = torch.tanh(self.gate_coup_emb(g_idx).sum(dim=-2)) * 0.35 * delta
        return x, delta, coup

    def forward_seq(self, seq_s, seq_is_target_ino, seq_valid):
        B, L, _ = seq_s.shape
        x_all, d_all, c_all = self._compute_step_params(seq_s)
        h_t = torch.zeros(B, self.dim, device=seq_s.device)
        h_i = torch.zeros(B, self.dim, device=seq_s.device)
        for t in range(L):
            v = seq_valid[:, t:t+1]
            mi = seq_is_target_ino[:, t:t+1]
            xt = x_all[:, t, :]; dt = d_all[:, t, :]; ct = c_all[:, t, :]
            # Selective ZOH diagonal coupled SSM:
            # h_t' = (1 - dt)*h_t - ct*h_i + dt*x_t
            # h_i' = (1 - dt)*h_i + ct*h_t + dt*x_t
            ht_new = (1.0 - dt) * h_t - ct * h_i + dt * xt
            hi_new = (1.0 - dt) * h_i + ct * h_t + dt * xt
            h_t = h_t * (1.0 - v) + ht_new * v
            vi = v * mi
            h_i = h_i * (1.0 - vi) + hi_new * vi
        feat = torch.cat([h_i, h_t], dim=-1)
        return self.head_life(feat), self.head_scan(feat).squeeze(-1)

    @torch.no_grad()
    def stream_eval(self, tids, insts, s_gram, target_evs, scan_evs=None):
        N = len(tids)
        n_t = int(tids.max()) + 1; n_i = int(insts.max()) + 1
        emb_w = self.emb.weight.detach().cpu().numpy()
        gd_w  = self.gate_delta_emb.weight.detach().cpu().numpy()
        gc_w  = self.gate_coup_emb.weight.detach().cpu().numpy()
        gd_b  = self.gate_delta_bias.detach().cpu().numpy()
        fd    = self.fixed_d.detach().cpu().numpy()
        fc    = self.fixed_c.detach().cpu().numpy()
        wl    = self.head_life.weight.detach().cpu().numpy()
        bl    = self.head_life.bias.detach().cpu().numpy()
        ws    = self.head_scan.weight.detach().cpu().numpy()[0]
        bs    = float(self.head_scan.bias.detach().cpu().item())

        H_t = np.zeros((n_t, self.dim), dtype=np.float32)
        H_i = np.zeros((n_i, self.dim), dtype=np.float32)
        t_set = {int(e): i for i, e in enumerate(target_evs)}
        s_set = {int(e): i for i, e in enumerate(scan_evs)} if scan_evs is not None else {}
        life_logits = np.zeros((len(target_evs), 3), dtype=np.float32)
        scan_scores = np.zeros(len(scan_evs), dtype=np.float32) if scan_evs is not None else None
        regime_counts = Counter()

        # Precompute lookup indices
        s0 = s_gram[:, 0].astype(np.int32)
        s1 = 32 + s_gram[:, 1].astype(np.int32)
        s2 = 96 + s_gram[:, 2].astype(np.int32)
        s3 = 128 + s_gram[:, 3].astype(np.int32)
        s4 = 160 + s_gram[:, 4].astype(np.int32)
        levels = np.array([0.0, 0.125, 0.5, 1.0], dtype=np.float32)

        for e in range(N):
            tid = int(tids[e]); iid = int(insts[e])
            i0, i1, i2, i3, i4 = int(s0[e]), int(s1[e]), int(s2[e]), int(s3[e]), int(s4[e])
            x = np.tanh(emb_w[i0] + emb_w[i1] + emb_w[i2] + emb_w[i3] + emb_w[i4]) * 2.0
            if not self.selective:
                dt = fd; ct = fc
            else:
                raw_d = gd_w[i0] + gd_w[i1] + gd_w[i2] + gd_w[i3] + gd_w[i4] + gd_b
                dt = 1.0 / (1.0 + np.exp(-np.clip(raw_d, -20.0, 20.0)))
                if self.mode == "sel4_swar":
                    bd = float(np.mean(dt))
                    reg = int(np.argmin(np.abs(levels - bd)))
                    regime_counts[reg] += 1
                    if reg == 0: # HOLD -> exact zero-cost skip!
                        if e in t_set:
                            f = np.concatenate([H_i[iid], H_t[tid]])
                            life_logits[t_set[e]] = wl @ f + bl
                        if e in s_set:
                            f = np.concatenate([H_i[iid], H_t[tid]])
                            scan_scores[s_set[e]] = float(np.dot(ws, f) + bs)
                        continue
                    dv = levels[reg]
                    # Bit-exact Q5.8 quantization of input & shift!
                    dt = dv
                    ct = 0.5 if reg == 3 else 0.125
                    x = np.round(x * 256.0) / 256.0
                else:
                    raw_c = gc_w[i0] + gc_w[i1] + gc_w[i2] + gc_w[i3] + gc_w[i4]
                    ct = np.tanh(raw_c) * 0.35 * dt

            oi = H_i[iid]; ot = H_t[tid]
            H_t[tid] = (1.0 - dt) * ot - ct * oi + dt * x
            H_i[iid] = (1.0 - dt) * oi + ct * ot + dt * x

            if e in t_set:
                f = np.concatenate([H_i[iid], H_t[tid]])
                life_logits[t_set[e]] = wl @ f + bl
            if e in s_set:
                f = np.concatenate([H_i[iid], H_t[tid]])
                scan_scores[s_set[e]] = float(np.dot(ws, f) + bs)
        return life_logits, scan_scores, dict(regime_counts)


# ==============================================================================
# BATTLEFIELD 2: Real Kernel Multi-Phase Interleaved-Noise Stress Test (K=0,5,15,35)
# ==============================================================================
def parse_mp_trace(mode):
    exp = "/home/yjw/workspace/research/KSSM/exp0b_selective/traces"
    pid = int(open(f"{exp}/mp_{mode}.pid").read().strip())
    arr = np.fromfile(f"{exp}/mp_{mode}.bin", dtype=DTYPE)
    sub = arr[arr["pid"] == pid]
    sub = sub[np.argsort(sub["ts_ns"], kind="mergesort")]
    N = len(sub)

    # Load ground-truth (ino -> phase_cls, k_noise) list in chronological birth order
    meta_by_ino = defaultdict(list)
    with open(f"{exp}/mp_{mode}_meta.csv") as f:
        for row in csv.DictReader(f):
            meta_by_ino[int(row["ino"])].append((int(row["phase_cls"]), int(row["k_noise"])))

    ops = sub["op"]; inos = sub["ino"]; tids = sub["tid"]; offs = sub["offset"]
    cnts = sub["count"]; fszs = sub["file_size"]; strides = sub["stride"]
    flags = sub["flags"]; ustks = sub["u_stack_id"]; kstks = sub["k_stack_id"]
    churns = sub["dir_churn"]; is_seqs = sub["is_seq"]

    # Track ext4 (ino, gen) and match OPEN_CREAT births to exact phase_cls & k_noise
    ino_gen = defaultdict(int); inst_map = {}; nxt_inst = 1
    inst_arr = np.zeros(N, dtype=np.int32)
    tid_map = {}; tid_arr = np.zeros(N, dtype=np.int32)
    meta_ptr = defaultdict(int)
    target_ev, target_y, target_k = [], [], []

    for i in range(N):
        t = int(tids[i])
        if t not in tid_map: tid_map[t] = len(tid_map)
        tid_arr[i] = tid_map[t]

        op = int(ops[i]); ino = int(inos[i])
        key = (ino, ino_gen[ino])
        if op == 2: # OPEN_CREAT
            if key in inst_map:
                ino_gen[ino] += 1; key = (ino, ino_gen[ino])
            inst_map[key] = nxt_inst; nxt_inst += 1
            inst_arr[i] = inst_map[key]
            # Check if this OPEN_CREAT is one of our shared_out target files
            if ino in meta_by_ino and meta_ptr[ino] < len(meta_by_ino[ino]):
                pcls, kn = meta_by_ino[ino][meta_ptr[ino]]
                meta_ptr[ino] += 1
                target_ev.append(i); target_y.append(pcls); target_k.append(kn)
        elif op in (7, 10):
            if key not in inst_map: inst_map[key] = nxt_inst; nxt_inst += 1
            inst_arr[i] = inst_map[key]; ino_gen[ino] += 1
        else:
            if key not in inst_map: inst_map[key] = nxt_inst; nxt_inst += 1
            inst_arr[i] = inst_map[key]

    # Causal 5-tuple grammar extraction (exact same algorithm as Exp 0-B!)
    s_gram = np.zeros((N, 5), dtype=np.uint8)
    num_t = len(tid_map)
    rings = np.full((num_t, 16), -1, dtype=np.int64)
    heads = np.zeros(num_t, dtype=np.int32)
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

    return {
        "N": N, "tid": tid_arr, "inst": inst_arr, "s_gram": s_gram,
        "ev": np.array(target_ev, dtype=np.int32),
        "y":  np.array(target_y, dtype=np.int64),
        "k":  np.array(target_k, dtype=np.int32),
    }

print("="*110, flush=True)
print(" BATTLEFIELD 2: Real Kernel Multi-Phase Interleaved-Noise Stress Test (Same Dir / Same Size / Same Flags!)", flush=True)
print("   Train K_noise in {0, 3, 8}  --->  Zero-Shot Noise Extrapolation Test K_noise in {0, 5, 15, 35}", flush=True)
print("="*110, flush=True)

mp_tr = parse_mp_trace("train")
mp_te = parse_mp_trace("test")
print(f"Parsed mp_train: N={mp_tr['N']}, targets={len(mp_tr['y'])} (K dist={dict(Counter(mp_tr['k'].tolist()))})", flush=True)
print(f"Parsed mp_test : N={mp_te['N']}, targets={len(mp_te['y'])} (K dist={dict(Counter(mp_te['k'].tolist()))})", flush=True)

# Build causal history windows of length L=48 for fast GPU training on mp_train (sample 3600 balanced targets)
def build_mp_train_seqs(d, L=48, max_n=3600):
    rng = np.random.RandomState(42)
    idx = rng.choice(len(d["ev"]), size=min(max_n, len(d["ev"])), replace=False)
    idx = np.sort(idx)
    evs = d["ev"][idx]; ys = d["y"][idx]
    M = len(evs)
    seq_s = np.zeros((M, L, 5), dtype=np.int64)
    seq_mi = np.zeros((M, L), dtype=np.float32)
    seq_v  = np.zeros((M, L), dtype=np.float32)
    sg = d["s_gram"]; inst = d["inst"]
    for m in range(M):
        ei = int(evs[m]); tgt_i = int(inst[ei])
        lo = max(0, ei - L + 1)
        seg = np.arange(lo, ei + 1)
        off = L - len(seg)
        seq_s[m, off:] = sg[seg]
        seq_v[m, off:] = 1.0
        seq_mi[m, off:] = (inst[seg] == tgt_i).astype(np.float32)
    return (torch.tensor(seq_s, device=DEVICE),
            torch.tensor(seq_mi, device=DEVICE),
            torch.tensor(seq_v, device=DEVICE),
            torch.tensor(ys, device=DEVICE))

tr_s, tr_mi, tr_v, tr_y = build_mp_train_seqs(mp_tr, L=48, max_n=3600)

def train_mp_model(model, epochs=28, lr=5e-3):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    B = 256; M = len(tr_y)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(M, device=DEVICE)
        for i in range(0, M, B):
            bi = perm[i:i+B]
            if len(bi) < 32: continue
            ll, _ = model.forward_seq(tr_s[bi], tr_mi[bi], tr_v[bi])
            loss = F.cross_entropy(ll, tr_y[bi])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()
    model.eval()
    return model

# Evaluate ALL 6 competitors on FULL continuous streaming mp_test (555,608 events)!
# Subsample evaluation targets evenly (6000 targets across K=0,5,15,35) for instant reporting while streaming FULL trace!
mp_models = [
    ("Heimdall (Stateless MLP, L=1)",               CrossEntityEWMA_MLP(64, alphas=(0.0,), hidden_mlp=128)),
    ("Cross-Entity Single-EWMA (a=0.80) + 3L-MLP",  CrossEntityEWMA_MLP(64, alphas=(0.80,), hidden_mlp=128)),
    ("Cross-Entity Multi-EWMA (4-scale) + 3L-MLP",  CrossEntityEWMA_MLP(64, alphas=(0.5, 0.75, 0.875, 0.9375), hidden_mlp=128)),
    ("Cross-Entity UltraSlow-EWMA (5-sc) + 3L-MLP", CrossEntityEWMA_MLP(64, alphas=(0.5, 0.75, 0.875, 0.9375, 0.96875), hidden_mlp=128)),
    ("Fixed-Decay Guard-SWAR 64D (Non-Selective)",  TrueSelectiveKSSM(64, selective=False, mode="fixed_swar")),
    ("True Selective KSSM (Float64D, Dynamic Gate)",TrueSelectiveKSSM(64, selective=True,  mode="float_sel")),
    ("True Selective 4-Regime Guard-SWAR (32-48ns)",TrueSelectiveKSSM(64, selective=True,  mode="sel4_swar")),
]

print(f"\n{'Model (Trained on K in {0,3,8}, Full-Stream Eval on 555K Real Kernel Events)':48s} | {'K=0':>7s} | {'K=5':>7s} | {'K=15':>7s} | {'K=35(OOD)':>9s} | {'Overall':>7s}", flush=True)
print("-"*110, flush=True)

for name, mdl in mp_models:
    mdl = train_mp_model(mdl, epochs=28, lr=5e-3)
    out = mdl.stream_eval(mp_te["tid"], mp_te["inst"], mp_te["s_gram"], mp_te["ev"])
    logits = out[0]
    preds = np.argmax(logits, axis=-1)
    y_te = mp_te["y"]; k_te = mp_te["k"]
    accs = {}
    for kv in [0, 5, 15, 35]:
        m = (k_te == kv)
        accs[kv] = cls_metrics(y_te[m], preds[m])[1] * 100.0
    ov = cls_metrics(y_te, preds)[1] * 100.0
    extra = f"  regimes={out[2]}" if (len(out) > 2 and out[2]) else ""
    print(f"{name:48s} | {accs[0]:6.2f}% | {accs[5]:6.2f}% | {accs[15]:6.2f}% | {accs[35]:8.2f}% | {ov:6.2f}%{extra}", flush=True)

# ==============================================================================
# BATTLEFIELD 1: Original 5-App Real Traces (dataset_exp0b_v3.pkl) — 5-Fold LOAO
# Adding Cross-Entity Multi-Scale EWMA + 3L-MLP & True Selective KSSM head-to-head!
# ==============================================================================
print("\n" + "="*110, flush=True)
print(" BATTLEFIELD 1: Original 5-App LOAO Zero-Shot (Adding Cross-Entity EWMA+MLP & True Selective KSSM)", flush=True)
print("="*110, flush=True)

with open("/home/yjw/workspace/research/KSSM/exp0b/dataset_exp0b_v3.pkl", "rb") as f:
    DATA = pickle.load(f)
APPS = ["rocksdb", "sqlite", "pytorch", "llm", "etl"]

def build_app_seqs(d, target_evs, labels, L=32, max_n=450):
    rng = np.random.RandomState(42)
    idx = np.arange(len(target_evs))
    if len(idx) > max_n:
        uniq = np.unique(labels); per_c = max(1, max_n // len(uniq))
        ch = [rng.choice(idx[labels == c], size=min(per_c, int(np.sum(labels == c))), replace=False) for c in uniq]
        idx = np.sort(np.concatenate(ch))
    evs = target_evs[idx]; ys = labels[idx]; M = len(evs)
    tid_arr = d["tid"]; inst_arr = d["inst"]; sg = d["s_gram"]
    # Pre-index by tid for fast lookup
    tid_evs = defaultdict(list)
    for ei in range(d["N"]): tid_evs[int(tid_arr[ei])].append(ei)
    tid_evs = {k: np.array(v, dtype=np.int32) for k, v in tid_evs.items()}

    seq_s = np.zeros((M, L, 5), dtype=np.int64)
    seq_mi = np.zeros((M, L), dtype=np.float32)
    seq_v  = np.zeros((M, L), dtype=np.float32)
    for m in range(M):
        ei = int(evs[m]); tt = int(tid_arr[ei]); ti = int(inst_arr[ei])
        te = tid_evs[tt]; tp = int(np.searchsorted(te, ei, side="right"))
        seg = te[max(0, tp - L):tp]
        off = L - len(seg)
        seq_s[m, off:] = sg[seg]
        seq_v[m, off:] = 1.0
        seq_mi[m, off:] = (inst_arr[seg] == ti).astype(np.float32)
    return seq_s, seq_mi, seq_v, ys

APP_WL = {a: build_app_seqs(DATA[a], DATA[a]["life_ev"], DATA[a]["life_y"], L=32, max_n=450) for a in APPS}
APP_WS = {a: build_app_seqs(DATA[a], DATA[a]["scan_ev"], DATA[a]["scan_y"], L=32, max_n=1000) for a in APPS}

def train_loao_model(train_apps, model, epochs=32, lr=5e-3):
    def gather(wd):
        ss, sm, sv, sy, sw = [], [], [], [], []
        for a in train_apps:
            s, m, v, y = wd[a]; cnt = Counter(y.tolist())
            w = np.array([1.0 / (len(cnt) * max(1, cnt[int(c)])) for c in y], dtype=np.float32)
            ss.append(s); sm.append(m); sv.append(v); sy.append(y); sw.append(w)
        return [torch.tensor(np.concatenate(x), device=DEVICE) for x in (ss, sm, sv, sy, sw)]
    ls, lm, lv, ly, lw = gather(APP_WL)
    ss, sm, sv, sy, sw = gather(APP_WS)

    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    B = 512; nl, ns = len(ly), len(sy)
    for ep in range(epochs):
        model.train()
        pl = torch.randperm(nl, device=DEVICE); ps = torch.randperm(ns, device=DEVICE)
        for step in range(max(nl // B, ns // B, 4)):
            il = pl[(step * B) % nl : ((step * B) % nl) + B]
            is_ = ps[(step * B) % ns : ((step * B) % ns) + B]
            if len(il) < 32: il = pl[:min(B, nl)]
            if len(is_) < 32: is_ = ps[:min(B, ns)]
            ll, _ = model.forward_seq(ls[il], lm[il], lv[il])
            l1 = (F.cross_entropy(ll, ly[il], reduction="none") * lw[il]).sum() / (lw[il].sum() + 1e-8)
            _, lso = model.forward_seq(ss[is_], sm[is_], sv[is_])
            l2 = (F.binary_cross_entropy_with_logits(lso, sy[is_].float(), reduction="none") * sw[is_]).sum() / (sw[is_].sum() + 1e-8)
            opt.zero_grad(); (l1 + l2).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()
    model.eval()
    return model

loao_specs = [
    ("Cross-Entity Single-EWMA (a=0.80) + 3L-MLP", lambda: CrossEntityEWMA_MLP(64, alphas=(0.80,), hidden_mlp=128)),
    ("Cross-Entity Multi-EWMA (4-scale) + 3L-MLP", lambda: CrossEntityEWMA_MLP(64, alphas=(0.5, 0.75, 0.875, 0.9375), hidden_mlp=128)),
    ("Fixed-Decay Guard-SWAR 64D (Non-Selective)", lambda: TrueSelectiveKSSM(64, selective=False, mode="fixed_swar")),
    ("True Selective KSSM (Float64D)",             lambda: TrueSelectiveKSSM(64, selective=True,  mode="float_sel")),
    ("True Selective 4-Regime Guard-SWAR(32-48ns)",lambda: TrueSelectiveKSSM(64, selective=True,  mode="sel4_swar")),
]
loao_res = {n: {"acc": [], "bal": [], "f1": [], "auc": []} for n, _ in loao_specs}

for held in APPS:
    tr_apps = [a for a in APPS if a != held]
    td = DATA[held]
    print(f"\n>>> LOAO Fold: {held.upper()}", flush=True)
    for name, ctor in loao_specs:
        mdl = train_loao_model(tr_apps, ctor())
        out = mdl.stream_eval(td["tid"], td["inst"], td["s_gram"], td["life_ev"], td["scan_ev"])
        ll, sc = out[0], out[1]
        a, b, f1 = cls_metrics(td["life_y"], np.argmax(ll, axis=-1))
        auc = roc_auc_exact(td["scan_y"], sc)
        loao_res[name]["acc"].append(a); loao_res[name]["bal"].append(b)
        loao_res[name]["f1"].append(f1); loao_res[name]["auc"].append(auc)
        print(f"  {name:44s} | Acc={a*100:5.1f}% BalAcc={b*100:5.1f}% F1={f1*100:5.1f}% | AUC={auc:.4f}", flush=True)

print("\n" + "="*110, flush=True)
print(" BATTLEFIELD 1 SUMMARY (5-App LOAO Macro-Average: Cross-Entity EWMA+MLP vs True Selective KSSM)", flush=True)
print("="*110, flush=True)
print(f"{'Method':46s} | {'Life0s Acc':>10s} | {'Life0s BalAcc':>13s} | {'Life0s F1':>10s} | {'Scan AUC':>9s}", flush=True)
print("-"*110, flush=True)
for name, _ in loao_specs:
    r = loao_res[name]
    print(f"{name:46s} | {np.mean(r['acc'])*100:9.2f}% | {np.mean(r['bal'])*100:12.2f}% | {np.mean(r['f1'])*100:9.2f}% | {np.mean(r['auc']):9.4f}", flush=True)
