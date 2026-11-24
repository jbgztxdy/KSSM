import os, sys, time, math, struct, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "3", "Must use GPU 3!"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42); np.random.seed(42)

with open("/home/yjw/workspace/research/KSSM/exp2_scale_distill/dataset_10app.pkl", "rb") as f:
    DATA = pickle.load(f)

APPS = ["rocksdb", "sqlite", "pytorch", "llm", "etl",
        "compiler", "git_vcs", "log_segment", "zstd_arch", "vllm_kv"]
ORIG5 = ["rocksdb", "sqlite", "pytorch", "llm", "etl"]
STATELESS_COLS = [0,1,2,3,4,5,6,7,8,10,11,12,13,15]

def roc_auc_exact(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int64); y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(y_true == 1)); n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0: return float("nan")
    order = np.argsort(y_score); ranks = np.empty_like(order, dtype=np.float64)
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
# Build Per-Thread + Target-Inode Causal Windows (L=32) for Fast GPU Training
# ==============================================================================
WIN_LEN = 32
def build_seqs(d, target_evs, labels, max_n=400):
    rng = np.random.RandomState(42)
    idx = np.arange(len(target_evs))
    if len(idx) > max_n:
        uniq = np.unique(labels); per_c = max(1, max_n // len(uniq))
        ch = [rng.choice(idx[labels == c], size=min(per_c, int(np.sum(labels == c))), replace=False) for c in uniq]
        idx = np.sort(np.concatenate(ch))
    evs = target_evs[idx]; ys = labels[idx]; M = len(evs)
    tid_arr = d["tid"]; inst_arr = d["inst"]; sg = d["s_gram"]
    tid_evs = defaultdict(list); inst_evs = defaultdict(list)
    for ei in range(d["N"]):
        tid_evs[int(tid_arr[ei])].append(ei)
        inst_evs[int(inst_arr[ei])].append(ei)
    tid_evs = {k: np.array(v, dtype=np.int32) for k, v in tid_evs.items()}
    inst_evs = {k: np.array(v, dtype=np.int32) for k, v in inst_evs.items()}

    seq_s  = np.zeros((M, WIN_LEN, 5), dtype=np.int64)
    seq_mi = np.zeros((M, WIN_LEN), dtype=np.float32) # 1 if target inode
    seq_mt = np.zeros((M, WIN_LEN), dtype=np.float32) # 1 if target thread
    seq_v  = np.zeros((M, WIN_LEN), dtype=np.float32)
    for m in range(M):
        ei = int(evs[m]); tt = int(tid_arr[ei]); ti = int(inst_arr[ei])
        te = tid_evs[tt]; tp = int(np.searchsorted(te, ei, side="right"))
        ie = inst_evs[ti]; ip = int(np.searchsorted(ie, ei, side="right"))
        merged = np.unique(np.concatenate([te[max(0, tp - 24):tp], ie[max(0, ip - 16):ip]]))
        if len(merged) > WIN_LEN: merged = merged[-WIN_LEN:]
        off = WIN_LEN - len(merged)
        seq_s[m, off:]  = sg[merged]
        seq_v[m, off:]  = 1.0
        seq_mi[m, off:] = (inst_arr[merged] == ti).astype(np.float32)
        seq_mt[m, off:] = (tid_arr[merged] == tt).astype(np.float32)
    return {"ev": evs, "y": ys, "s": seq_s, "mi": seq_mi, "mt": seq_mt, "v": seq_v}

print("Pre-building causal training windows across all 10 real apps...", flush=True)
t_w0 = time.time()
WL = {a: build_seqs(DATA[a], DATA[a]["life_ev"], DATA[a]["life_y"], max_n=360) for a in APPS}
WS = {a: build_seqs(DATA[a], DATA[a]["scan_ev"], DATA[a]["scan_y"], max_n=800) for a in APPS}
print(f"Built 10-app windows in {time.time() - t_w0:.1f}s", flush=True)

# ==============================================================================
# Models
# ==============================================================================
class HeimdallMLP(nn.Module):
    def __init__(self, in_dim=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.hl = nn.Linear(32, 3); self.hs = nn.Linear(32, 1)
    def forward(self, x): h = self.net(x); return self.hl(h), self.hs(h).squeeze(-1)

class CrossEntityEWMA_MLP(nn.Module):
    def __init__(self, dim=64, alphas=(0.5, 0.75, 0.875, 0.9375)):
        super().__init__()
        self.dim = dim; self.emb = nn.Embedding(192, dim)
        nn.init.normal_(self.emb.weight, 0, 0.15)
        av = np.repeat(alphas, dim // len(alphas)).astype(np.float32)
        self.register_buffer("a", torch.tensor(av))
        self.mlp = nn.Sequential(
            nn.Linear(dim * 3, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(),
        )
        self.hl = nn.Linear(64, 3); self.hs = nn.Linear(64, 1)

    def forward_seq(self, s, mi, mt, v, **kw):
        B, L, _ = s.shape
        offs = torch.tensor([0, 32, 96, 128, 160], device=s.device)
        x_all = self.emb(s + offs).sum(-2)
        h_t = torch.zeros(B, self.dim, device=s.device)
        h_i = torch.zeros(B, self.dim, device=s.device)
        a = self.a; oa = 1.0 - a
        for t in range(L):
            vt = (v[:, t:t+1] * mt[:, t:t+1]); vi = (v[:, t:t+1] * mi[:, t:t+1])
            xt = x_all[:, t, :]
            h_t = h_t * (1.0 - vt) + (a * h_t + oa * xt) * vt
            h_i = h_i * (1.0 - vi) + (a * h_i + oa * xt) * vi
        z = self.mlp(torch.cat([h_i, h_t, x_all[:, -1, :]], dim=-1))
        return self.hl(z), self.hs(z).squeeze(-1)

    @torch.no_grad()
    def stream_eval(self, d):
        N = d["N"]; tids = d["tid"]; insts = d["inst"]; sg = d["s_gram"]
        ew = self.emb.weight.detach().cpu().numpy(); a = self.a.cpu().numpy(); oa = 1.0 - a
        H_t = np.zeros((d["num_tids"], self.dim), dtype=np.float32)
        H_i = np.zeros((d["num_insts"], self.dim), dtype=np.float32)
        lset = {int(e): i for i, e in enumerate(d["life_ev"])}
        sset = {int(e): i for i, e in enumerate(d["scan_ev"])}
        lf = np.zeros((len(d["life_ev"]), self.dim * 3), dtype=np.float32)
        sf = np.zeros((len(d["scan_ev"]), self.dim * 3), dtype=np.float32)
        for e in range(N):
            tid = int(tids[e]); iid = int(insts[e])
            x = ew[int(sg[e,0])] + ew[32+int(sg[e,1])] + ew[96+int(sg[e,2])] + ew[128+int(sg[e,3])] + ew[160+int(sg[e,4])]
            H_t[tid] = a * H_t[tid] + oa * x; H_i[iid] = a * H_i[iid] + oa * x
            if e in lset: lf[lset[e]] = np.concatenate([H_i[iid], H_t[tid], x])
            if e in sset: sf[sset[e]] = np.concatenate([H_i[iid], H_t[tid], x])
        ll = self.hl(self.mlp(torch.tensor(lf, device=DEVICE))).cpu().numpy()
        scs = []
        for i in range(0, len(sf), 16384):
            scs.append(self.hs(self.mlp(torch.tensor(sf[i:i+16384], device=DEVICE))).squeeze(-1).cpu().numpy())
        return ll, np.concatenate(scs), lf

class KSSMUnified(nn.Module):
    """Supports:
      - inode_only : single-entity ablation (no h_task)
      - uncoupled  : dual-entity independent (gamma=eta=0)
      - fixed_swar : non-selective fixed multi-scale Guard-SWAR (46-58ns)
      - float_sel  : continuous input-dependent Selective Diagonal Coupled SSM
      - sel4_swar  : 4-Band Independent Selective Guard-SWAR 64D + Hard-STE QAT (37.8-50.2ns Bit-Exact!)
    """
    def __init__(self, mode="sel4_swar"):
        super().__init__()
        self.mode = mode
        self.emb = nn.Embedding(192, 64)
        nn.init.normal_(self.emb.weight, 0, 0.15)
        self.drop = nn.Dropout(0.15)

        if mode in ("inode_only", "uncoupled", "float_sel"):
            self.gd = nn.Embedding(192, 64); self.gc = nn.Embedding(192, 64)
            nn.init.normal_(self.gd.weight, 0, 0.1); nn.init.normal_(self.gc.weight, 0, 0.1)
            self.gdb = nn.Parameter(torch.full((64,), -0.5))
        elif mode == "fixed_swar":
            ds = np.repeat([1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4], 4)
            cs = np.repeat([2,2,3,3, 3,3,4,4, 4,4,5,5, 5,5,5,5], 4)
            self.register_buffer("fa", torch.tensor(1.0 - 2.0**(-ds), dtype=torch.float32))
            self.register_buffer("fg", torch.tensor(2.0**(-cs), dtype=torch.float32))
            lims = np.repeat([(2.0**(-ds[g*4]) - 2.0**(-cs[g*4])) * 32.0 * 0.85 / 5.0 for g in range(16)], 4).astype(np.float32)
            self.register_buffer("lim", torch.tensor(lims))
        elif mode == "sel4_swar":
            # Exact twin of kssm_sel4band.bpf.c (37.8-50.2 ns on Linux 5.15!)
            self.g_s1   = nn.Parameter(torch.zeros(32, 4, 4))
            self.g_s4   = nn.Parameter(torch.zeros(32, 4, 4))
            self.g_pair = nn.Parameter(torch.zeros(1024, 4, 4))
            self.register_buffer("a_i_vals", torch.tensor([1.0, 0.875, 0.5, 0.0]))
            self.register_buffer("g_i_vals", torch.tensor([0.0, 0.0625, 0.25, 0.5]))
            self.register_buffer("a_t_vals", torch.tensor([1.0, 0.875, 0.5, 0.75]))
            self.register_buffer("g_t_vals", torch.tensor([0.0, 0.0625, 0.25, 0.0]))
            self.register_buffer("in_scale", torch.tensor([0.0, 1.0, 1.0, 1.0]))

        ro_in = 64 if mode == "inode_only" else 128
        self.hl = nn.Linear(ro_in, 3); self.hs = nn.Linear(ro_in, 1)

    def tv_loss(self):
        w = torch.tanh(self.emb.weight)
        l5 = F.mse_loss(w[160:175], w[161:176]) + F.mse_loss(w[176:191], w[177:192])
        l2 = sum(F.mse_loss(w[32+q*16:32+q*16+15], w[32+q*16+1:32+q*16+16]) for q in range(4))
        return 0.15 * (l5 + 0.5 * l2)

    def get_sel4_logits(self, s1, s4):
        return self.g_s1[s1] + self.g_s4[s4] + 0.5 * self.g_pair[(s1 << 5) | s4]

    def forward_seq(self, s, mi, mt, v, tau=1.0, hard=False):
        B, L, _ = s.shape
        offs = torch.tensor([0, 32, 96, 128, 160], device=s.device)
        g = s + offs
        h_t = torch.zeros(B, 64, device=s.device)
        h_i = torch.zeros(B, 64, device=s.device)

        if self.mode == "sel4_swar":
            w = torch.tanh(self.emb(g)) * 0.75
            b_all = torch.cat([
                w[:, :, 0, 0:16]  + w[:, :, 1, 0:16],
                w[:, :, 0, 16:32] + w[:, :, 2, 16:32],
                w[:, :, 0, 32:48] + w[:, :, 3, 32:48],
                w[:, :, 0, 48:64] + w[:, :, 4, 48:64],
            ], dim=-1)
            probs = F.softmax(self.get_sel4_logits(s[:, :, 0], s[:, :, 3]) / tau, dim=-1)
            if hard:
                idx = probs.argmax(dim=-1, keepdim=True)
                oh = torch.zeros_like(probs).scatter_(-1, idx, 1.0)
                probs = (oh - probs).detach() + probs
            ai  = (probs * self.a_i_vals).sum(-1).repeat_interleave(16, dim=-1)
            gi  = (probs * self.g_i_vals).sum(-1).repeat_interleave(16, dim=-1)
            at  = (probs * self.a_t_vals).sum(-1).repeat_interleave(16, dim=-1)
            gt  = (probs * self.g_t_vals).sum(-1).repeat_interleave(16, dim=-1)
            is_ = (probs * self.in_scale).sum(-1).repeat_interleave(16, dim=-1)
            for t in range(L):
                vt = v[:, t:t+1] * mt[:, t:t+1]; vi = v[:, t:t+1] * mi[:, t:t+1]
                bt = b_all[:, t, :] * is_[:, t, :]
                hi_n = ai[:, t, :] * h_i + gi[:, t, :] * (h_t * mt[:, t:t+1]) + bt
                ht_n = at[:, t, :] * h_t - gt[:, t, :] * (h_i * mi[:, t:t+1]) + bt
                h_i = h_i * (1.0 - vi) + hi_n * vi
                h_t = h_t * (1.0 - vt) + ht_n * vt
        elif self.mode == "fixed_swar":
            b_all = (torch.tanh(self.emb(g)) * self.lim).sum(-2)
            for t in range(L):
                vt = v[:, t:t+1] * mt[:, t:t+1]; vi = v[:, t:t+1] * mi[:, t:t+1]
                hi_n = self.fa * h_i + self.fg * (h_t * mt[:, t:t+1]) + b_all[:, t, :]
                ht_n = self.fa * h_t - self.fg * (h_i * mi[:, t:t+1]) + b_all[:, t, :]
                h_i = h_i * (1.0 - vi) + hi_n * vi
                h_t = h_t * (1.0 - vt) + ht_n * vt
        else:
            x_all = torch.tanh(self.emb(g).sum(-2)) * 2.0
            d_all = torch.sigmoid(self.gd(g).sum(-2) + self.gdb)
            if self.mode == "float_sel":
                c_all = torch.tanh(self.gc(g).sum(-2)) * 0.35 * d_all
            else:
                c_all = torch.zeros_like(d_all)
            for t in range(L):
                vt = v[:, t:t+1] * mt[:, t:t+1]; vi = v[:, t:t+1] * mi[:, t:t+1]
                dt = d_all[:, t, :]; ct = c_all[:, t, :]; xt = x_all[:, t, :]
                hi_n = (1.0 - dt) * h_i + ct * (h_t * mt[:, t:t+1]) + dt * xt
                ht_n = (1.0 - dt) * h_t - ct * (h_i * mi[:, t:t+1]) + dt * xt
                h_i = h_i * (1.0 - vi) + hi_n * vi
                h_t = h_t * (1.0 - vt) + ht_n * vt

        feat = self.drop(h_i if self.mode == "inode_only" else torch.cat([h_i, h_t], dim=-1))
        return self.hl(feat), self.hs(feat).squeeze(-1)

    @torch.no_grad()
    def stream_eval(self, d):
        N = d["N"]; tids = d["tid"]; insts = d["inst"]; sg = d["s_gram"]
        lset = {int(e): i for i, e in enumerate(d["life_ev"])}
        sset = {int(e): i for i, e in enumerate(d["scan_ev"])}
        ro_d = 64 if self.mode == "inode_only" else 128
        ll = np.zeros((len(d["life_ev"]), 3), dtype=np.float32)
        sc = np.zeros(len(d["scan_ev"]), dtype=np.float32)
        lf = np.zeros((len(d["life_ev"]), ro_d), dtype=np.float32)
        wl = self.hl.weight.detach().cpu().numpy(); bl = self.hl.bias.detach().cpu().numpy()
        ws = self.hs.weight.detach().cpu().numpy()[0]; bs = float(self.hs.bias.detach().cpu().item())

        if self.mode == "sel4_swar":
            # 100% Bit-Exact uint16 Excess-16384 Guard-SWAR simulation of kssm_sel4band.bpf.c!
            w_f = (torch.tanh(self.emb.weight) * 0.75).detach().cpu().numpy()
            emb_u16 = (np.round(w_f * 256.0).astype(np.int32) + 2048).astype(np.uint16)
            s1_g = torch.arange(32, device=DEVICE).repeat_interleave(32)
            s4_g = torch.arange(32, device=DEVICE).repeat(32)
            reg_tbl = self.get_sel4_logits(s1_g, s4_g).argmax(dim=-1).cpu().numpy().astype(np.uint8)
            wl_q = np.round(wl * 256.0).astype(np.int32); bl_q = np.round(bl * 65536.0).astype(np.int32)
            ws_q = np.round(ws * 256.0).astype(np.int32); bs_q = int(round(bs * 65536.0))

            H_i = np.full((d["num_insts"], 64), 16384, dtype=np.int32)
            H_t = np.full((d["num_tids"], 64), 16384, dtype=np.int32)
            s0 = sg[:, 0].astype(np.int32); s1 = 32 + sg[:, 1].astype(np.int32)
            s2 = 96 + sg[:, 2].astype(np.int32); s3 = 128 + sg[:, 3].astype(np.int32); s4 = 160 + sg[:, 4].astype(np.int32)
            gkeys = (sg[:, 0].astype(np.int32) << 5) | sg[:, 3].astype(np.int32)
            min_s, max_s = 16384, 16384

            for e in range(N):
                tid = int(tids[e]); iid = int(insts[e])
                regs = reg_tbl[int(gkeys[e])]
                oi = H_i[iid]; ot = H_t[tid]
                i0 = int(s0[e]); sb = (int(s1[e]), int(s2[e]), int(s3[e]), int(s4[e]))
                for b in range(4):
                    r = int(regs[b])
                    if r == 0: continue
                    lo = b * 16; hi = lo + 16
                    bsum = emb_u16[i0, lo:hi].astype(np.int32) + emb_u16[sb[b], lo:hi]
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
                if e in lset:
                    idx = lset[e]; fq = np.concatenate([oi - 16384, ot - 16384])
                    ll[idx] = (wl_q @ fq + bl_q) / 65536.0; lf[idx] = fq / 256.0
                if e in sset:
                    fq = np.concatenate([oi - 16384, ot - 16384])
                    sc[sset[e]] = float(np.dot(ws_q, fq) + bs_q) / 65536.0
            return ll, sc, lf, (min_s, max_s)

        elif self.mode == "fixed_swar":
            ti = (torch.tanh(self.emb.weight) * self.lim).detach().cpu().numpy()
            emb_u16 = (np.round(ti * 256.0).astype(np.int32) + 2048).astype(np.uint16)
            ds = np.repeat([1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4], 4).astype(np.int32)
            cs = np.repeat([2,2,3,3, 3,3,4,4, 4,4,5,5, 5,5,5,5], 4).astype(np.int32)
            c_pos = (10240 + (16384 >> cs) - (16384 >> ds)).astype(np.int32)
            c_neg = (10240 - (16384 >> cs) - (16384 >> ds)).astype(np.int32)
            wl_q = np.round(wl * 256.0).astype(np.int32); bl_q = np.round(bl * 65536.0).astype(np.int32)
            ws_q = np.round(ws * 256.0).astype(np.int32); bs_q = int(round(bs * 65536.0))
            H_i = np.full((d["num_insts"], 64), 16384, dtype=np.int32)
            H_t = np.full((d["num_tids"], 64), 16384, dtype=np.int32)
            for e in range(N):
                tid = int(tids[e]); iid = int(insts[e]); oi = H_i[iid]; ot = H_t[tid]
                bsum = (emb_u16[int(sg[e,0])].astype(np.int32) + emb_u16[32+int(sg[e,1])] +
                        emb_u16[96+int(sg[e,2])] + emb_u16[128+int(sg[e,3])] + emb_u16[160+int(sg[e,4])])
                ni = oi - (oi >> ds) + (ot >> cs) + bsum - c_pos
                nt = ot - (ot >> ds) - (oi >> cs) + bsum - c_neg
                H_i[iid] = ni; H_t[tid] = nt
                if e in lset:
                    idx = lset[e]; fq = np.concatenate([ni - 16384, nt - 16384])
                    ll[idx] = (wl_q @ fq + bl_q) / 65536.0; lf[idx] = fq / 256.0
                if e in sset:
                    fq = np.concatenate([ni - 16384, nt - 16384])
                    sc[sset[e]] = float(np.dot(ws_q, fq) + bs_q) / 65536.0
            return ll, sc, lf, (0, 0)
        else:
            ew = self.emb.weight.detach().cpu().numpy()
            gdw = self.gd.weight.detach().cpu().numpy(); gcw = self.gc.weight.detach().cpu().numpy()
            gdb = self.gdb.detach().cpu().numpy()
            H_i = np.zeros((d["num_insts"], 64), dtype=np.float32)
            H_t = np.zeros((d["num_tids"], 64), dtype=np.float32)
            for e in range(N):
                tid = int(tids[e]); iid = int(insts[e])
                i0, i1, i2, i3, i4 = int(sg[e,0]), 32+int(sg[e,1]), 96+int(sg[e,2]), 128+int(sg[e,3]), 160+int(sg[e,4])
                x = np.tanh(ew[i0]+ew[i1]+ew[i2]+ew[i3]+ew[i4]) * 2.0
                rd = gdw[i0]+gdw[i1]+gdw[i2]+gdw[i3]+gdw[i4]+gdb
                dt = 1.0 / (1.0 + np.exp(-np.clip(rd, -20.0, 20.0)))
                ct = (np.tanh(gcw[i0]+gcw[i1]+gcw[i2]+gcw[i3]+gcw[i4]) * 0.35 * dt) if self.mode == "float_sel" else 0.0
                oi = H_i[iid]; ot = H_t[tid]
                H_i[iid] = (1.0 - dt) * oi + ct * ot + dt * x
                H_t[tid] = (1.0 - dt) * ot - ct * oi + dt * x
                if e in lset:
                    idx = lset[e]; f = H_i[iid] if self.mode == "inode_only" else np.concatenate([H_i[iid], H_t[tid]])
                    ll[idx] = wl @ f + bl; lf[idx] = f
                if e in sset:
                    f = H_i[iid] if self.mode == "inode_only" else np.concatenate([H_i[iid], H_t[tid]])
                    sc[sset[e]] = float(np.dot(ws, f) + bs)
            return ll, sc, lf, (0, 0)

def train_seq_model(train_apps, model, epochs=32, lr=6e-3, is_sel4=False):
    def gather(wd):
        ss, smi, smt, sv, sy, sw = [], [], [], [], [], []
        for a in train_apps:
            d = wd[a]; y = d["y"]; cnt = Counter(y.tolist())
            w = np.array([1.0 / (len(cnt) * max(1, cnt[int(c)])) for c in y], dtype=np.float32)
            ss.append(d["s"]); smi.append(d["mi"]); smt.append(d["mt"]); sv.append(d["v"])
            sy.append(y); sw.append(w)
        return [torch.tensor(np.concatenate(x), device=DEVICE) for x in (ss, smi, smt, sv, sy, sw)]
    ls, lmi, lmt, lv, ly, lw = gather(WL)
    ss, smi, smt, sv, sy, sw = gather(WS)

    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    B = 512; nl, ns = len(ly), len(sy)
    for ep in range(epochs):
        model.train()
        tau = max(0.06, 2.0 * (0.86 ** ep)) if is_sel4 else 1.0
        hard = (is_sel4 and ep >= epochs - 8)
        pl = torch.randperm(nl, device=DEVICE); ps = torch.randperm(ns, device=DEVICE)
        for step in range(max(nl // B, ns // B, 6)):
            il = pl[(step * B) % nl : ((step * B) % nl) + B]
            is_ = ps[(step * B) % ns : ((step * B) % ns) + B]
            if len(il) < 32: il = pl[:min(B, nl)]
            if len(is_) < 32: is_ = ps[:min(B, ns)]
            ll, _ = model.forward_seq(ls[il], lmi[il], lmt[il], lv[il], tau=tau, hard=hard)
            l1 = (F.cross_entropy(ll, ly[il], reduction="none") * lw[il]).sum() / (lw[il].sum() + 1e-8)
            _, lso = model.forward_seq(ss[is_], smi[is_], smt[is_], sv[is_], tau=tau, hard=hard)
            l2 = (F.binary_cross_entropy_with_logits(lso, sy[is_].float(), reduction="none") * sw[is_]).sum() / (sw[is_].sum() + 1e-8)
            loss = l1 + l2 + (model.tv_loss() if hasattr(model, "tv_loss") else 0.0)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        sched.step()
    model.eval()
    return model

def train_heimdall(train_apps, epochs=35):
    xl, yl, wl, xs, ys, ws = [], [], [], [], [], []
    for a in train_apps:
        d = DATA[a]; wdl = WL[a]; wds = WS[a]
        cl = Counter(wdl["y"].tolist()); wla = np.array([1.0/(len(cl)*max(1,cl[int(c)])) for c in wdl["y"]], dtype=np.float32)
        xl.append(d["scalars"][wdl["ev"]][:, STATELESS_COLS]); yl.append(wdl["y"]); wl.append(wla)
        cs = Counter(wds["y"].tolist()); wsa = np.array([1.0/(len(cs)*max(1,cs[int(c)])) for c in wds["y"]], dtype=np.float32)
        xs.append(d["scalars"][wds["ev"]][:, STATELESS_COLS]); ys.append(wds["y"]); ws.append(wsa)
    xl, yl, wl, xs, ys, ws = [torch.tensor(np.concatenate(x), device=DEVICE) for x in (xl, yl, wl, xs, ys, ws)]
    mlp = HeimdallMLP(len(STATELESS_COLS)).to(DEVICE)
    opt = torch.optim.AdamW(mlp.parameters(), lr=5e-3, weight_decay=2e-3)
    B = 512; nl, ns = len(yl), len(ys)
    for ep in range(epochs):
        mlp.train(); pl = torch.randperm(nl, device=DEVICE); ps = torch.randperm(ns, device=DEVICE)
        for step in range(max(nl // B, ns // B, 6)):
            il = pl[(step * B) % nl : ((step * B) % nl) + B]; is_ = ps[(step * B) % ns : ((step * B) % ns) + B]
            if len(il) < 32: il = pl[:min(B, nl)]
            if len(is_) < 32: is_ = ps[:min(B, ns)]
            ll, _ = mlp(xl[il]); l1 = (F.cross_entropy(ll, yl[il], reduction="none") * wl[il]).sum() / (wl[il].sum() + 1e-8)
            _, ls = mlp(xs[is_]); l2 = (F.binary_cross_entropy_with_logits(ls, ys[is_].float(), reduction="none") * ws[is_]).sum() / (ws[is_].sum() + 1e-8)
            opt.zero_grad(); (l1 + l2).backward(); opt.step()
    mlp.eval(); return mlp

def eval_basic(train_apps, test_app):
    lt = defaultdict(lambda: np.ones(3)*0.5); lu = defaultdict(lambda: np.ones(3)*0.5)
    st = defaultdict(lambda: np.ones(2)*0.5); su = defaultdict(lambda: np.ones(2)*0.5)
    for a in train_apps:
        d = DATA[a]; toks = d["basic_tok"]; tids = d["tid"]
        pt = {}; prev = np.full(d["N"], -1, dtype=np.int32)
        for i in range(d["N"]): t = int(tids[i]); prev[i] = pt.get(t, -1); pt[t] = int(toks[i])
        cl = Counter(d["life_y"].tolist())
        for ei, y in zip(d["life_ev"], d["life_y"]):
            w = 100.0 / max(1, cl[int(y)]); lt[(int(prev[ei]), int(toks[ei]))][int(y)] += w; lu[int(toks[ei])][int(y)] += w
        sy = d["scan_y"]; cs = Counter(sy.tolist()); step = max(1, len(sy)//3000)
        for ei, y in zip(d["scan_ev"][::step], sy[::step]):
            w = 100.0 / max(1, cs[int(y)]//step); st[(int(prev[ei]), int(toks[ei]))][int(y)] += w; su[int(toks[ei])][int(y)] += w
    td = DATA[test_app]; toks = td["basic_tok"]; tids = td["tid"]
    pt = {}; prev = np.full(td["N"], -1, dtype=np.int32)
    for i in range(td["N"]): t = int(tids[i]); prev[i] = pt.get(t, -1); pt[t] = int(toks[i])
    lp = [int(np.argmax(lt[(int(prev[ei]), int(toks[ei]))] if (int(prev[ei]), int(toks[ei])) in lt else lu[int(toks[ei])])) for ei in td["life_ev"]]
    ss = [float((st[(int(prev[ei]), int(toks[ei]))] if (int(prev[ei]), int(toks[ei])) in st else su[int(toks[ei])])[1] /
                np.sum(st[(int(prev[ei]), int(toks[ei]))] if (int(prev[ei]), int(toks[ei])) in st else su[int(toks[ei])])) for ei in td["scan_ev"]]
    return np.array(lp, dtype=np.int64), np.array(ss, dtype=np.float32)

def eval_slowloop_online(d, feats, W0, b0):
    y = d["life_y"]; tb = d["life_birth_s"]; td_s = d["life_death_s"]; n = len(y)
    W = W0.copy(); b = b0.copy()
    fb = sorted([(td_s[i], i, int(y[i])) if td_s[i] >= 0 else (tb[i] + 22.0, i, 2) for i in range(n)])
    fp = 0; preds = np.zeros(n, dtype=np.int64)
    for i in range(n):
        while fp < len(fb) and fb[fp][0] <= tb[i]:
            _, fi, fy = fb[fp]; xf = feats[fi]
            lg = W @ xf + b; lg -= lg.max(); pr = np.exp(lg); pr /= pr.sum()
            gr = pr; gr[fy] -= 1.0; cw = 2.0 if fy > 0 else 0.5
            W -= (0.05 * cw) * np.outer(gr, xf); b -= (0.05 * cw) * gr; fp += 1
        preds[i] = int(np.argmax(W @ feats[i] + b))
    return preds

METHODS = [
    "PCStream (FAST'19)",
    "Heimdall (EuroSys'25, Faithful)",
    "BASIC-Prefetch (SYSTOR'26)",
    "Cross-Entity Multi-EWMA + 3L-MLP",
    "KSSM-InodeOnly (No h_task)",
    "KSSM-Uncoupled (gamma=eta=0)",
    "KSSM-FixedSWAR (46-58ns)",
    "KSSM-TrueSelective-Float64D",
    "KSSM-4Band-SelSWAR (38-50ns, QAT)",
    "KSSM-4Band-SelSWAR (+SlowLoop)",
]
res = {m: {a: None for a in APPS} for m in METHODS}
saved_orig5_sel4 = {} # save 9-app-trained LOAO predictions on ORIG5 for FDP WAF comparison!

print("\n" + "="*116, flush=True)
print(" EXP 2: 10-APP REAL CORPUS (569K Kernel Events) — 10-Fold Leave-One-App-Out Zero-Shot Benchmark", flush=True)
print("="*116, flush=True)

for fold_i, held in enumerate(APPS):
    tf0 = time.time()
    tr_apps = [a for a in APPS if a != held]
    td = DATA[held]; yl = td["life_y"]; ys = td["scan_y"]
    print(f"\n>>> FOLD {fold_i+1}/10: Held-Out Unseen App = {held.upper():11s} (life_0s={len(yl):4d}, scan={len(ys):6d})", flush=True)

    # 1. PCStream (Zero-Shot -> 0 hits across binaries -> majority class 0)
    a, b, f1 = cls_metrics(yl, np.zeros_like(yl))
    res["PCStream (FAST'19)"][held] = (a, b, f1, float("nan"))

    # 2. Heimdall Faithful
    hm = train_heimdall(tr_apps)
    with torch.no_grad():
        ll_h, _ = hm(torch.tensor(td["scalars"][td["life_ev"]][:, STATELESS_COLS], device=DEVICE))
        _, ls_h = hm(torch.tensor(td["scalars"][td["scan_ev"]][:, STATELESS_COLS], device=DEVICE))
    a, b, f1 = cls_metrics(yl, ll_h.argmax(-1).cpu().numpy()); auc = roc_auc_exact(ys, ls_h.cpu().numpy())
    res["Heimdall (EuroSys'25, Faithful)"][held] = (a, b, f1, auc)
    print(f"  {'Heimdall (Faithful)':34s} | Acc={a*100:5.1f}% BalAcc={b*100:5.1f}% F1={f1*100:5.1f}% | AUC={auc:.4f}", flush=True)

    # 3. BASIC-Prefetcher
    pb, sb = eval_basic(tr_apps, held)
    a, b, f1 = cls_metrics(yl, pb); auc = roc_auc_exact(ys, sb)
    res["BASIC-Prefetch (SYSTOR'26)"][held] = (a, b, f1, auc)

    # 4. Cross-Entity Multi-EWMA + 3L-MLP (Strongest Reviewer Ablation)
    ew = train_seq_model(tr_apps, CrossEntityEWMA_MLP(64), epochs=30, lr=5e-3)
    lle, sce, _ = ew.stream_eval(td)
    a, b, f1 = cls_metrics(yl, np.argmax(lle, -1)); auc = roc_auc_exact(ys, sce)
    res["Cross-Entity Multi-EWMA + 3L-MLP"][held] = (a, b, f1, auc)
    print(f"  {'Cross-Entity Multi-EWMA + 3L-MLP':34s} | Acc={a*100:5.1f}% BalAcc={b*100:5.1f}% F1={f1*100:5.1f}% | AUC={auc:.4f}", flush=True)

    # 5-9. KSSM Variants (Ablations + FixedSWAR + FloatSel + 4-Band Selective SWAR Hard-STE QAT!)
    for lbl, mode, is_s4 in [
        ("KSSM-InodeOnly (No h_task)",        "inode_only", False),
        ("KSSM-Uncoupled (gamma=eta=0)",      "uncoupled",  False),
        ("KSSM-FixedSWAR (46-58ns)",          "fixed_swar", False),
        ("KSSM-TrueSelective-Float64D",       "float_sel",  False),
        ("KSSM-4Band-SelSWAR (38-50ns, QAT)", "sel4_swar",  True),
    ]:
        km = train_seq_model(tr_apps, KSSMUnified(mode), epochs=32, lr=7e-3 if is_s4 else 6e-3, is_sel4=is_s4)
        llk, sck, lfk, bnd = km.stream_eval(td)
        pk = np.argmax(llk, -1)
        a, b, f1 = cls_metrics(yl, pk); auc = roc_auc_exact(ys, sck)
        res[lbl][held] = (a, b, f1, auc)
        bstr = f" [SWAR={bnd}]" if is_s4 else ""
        print(f"  {lbl:34s} | Acc={a*100:5.1f}% BalAcc={b*100:5.1f}% F1={f1*100:5.1f}% | AUC={auc:.4f}{bstr}", flush=True)

        if lbl == "KSSM-4Band-SelSWAR (38-50ns, QAT)":
            W0 = km.hl.weight.detach().cpu().numpy(); b0 = km.hl.bias.detach().cpu().numpy()
            pon = eval_slowloop_online(td, lfk, W0, b0)
            ao, bo, f1o = cls_metrics(yl, pon)
            res["KSSM-4Band-SelSWAR (+SlowLoop)"][held] = (ao, bo, f1o, auc)
            print(f"  {'KSSM-4Band-SelSWAR (+SlowLoop)':34s} | Acc={ao*100:5.1f}% BalAcc={bo*100:5.1f}% F1={f1o*100:5.1f}% | AUC={auc:.4f}", flush=True)
            if held in ORIG5:
                saved_orig5_sel4[held] = {"zs": pk, "feat": lfk, "W0": W0, "b0": b0}
    print(f"  [Fold {fold_i+1}/10 done in {time.time()-tf0:.1f}s]", flush=True)

def print_summary(title, app_subset):
    print("\n" + "="*116, flush=True)
    print(f" {title}", flush=True)
    print("="*116, flush=True)
    print(f"{'Method':36s} | {'Life0s Acc':>10s} | {'Life0s BalAcc':>13s} | {'Life0s MacroF1':>14s} | {'One-Shot AUC':>12s}", flush=True)
    print("-"*116, flush=True)
    for m in METHODS:
        accs = [res[m][a][0] for a in app_subset]
        bals = [res[m][a][1] for a in app_subset]
        f1s  = [res[m][a][2] for a in app_subset]
        aucs = [res[m][a][3] for a in app_subset if not math.isnan(res[m][a][3])]
        as_ = f"{np.mean(aucs):.4f}" if aucs else "N/A"
        print(f"{m:36s} | {np.mean(accs)*100:9.2f}% | {np.mean(bals)*100:12.2f}% | {np.mean(f1s)*100:13.2f}% | {as_:>12s}", flush=True)

print_summary("SUMMARY A: Full 10-App Leave-One-App-Out (LOAO) Zero-Shot Macro-Average", APPS)
print_summary("SUMMARY B: Scaling Impact on Original 5 Apps (Trained on 9 Apps vs Previous 4 Apps!)", ORIG5)

# Save 10-app LOAO results & orig5 predictions for downstream FDP WAF verification!
with open("/home/yjw/workspace/research/KSSM/exp2_scale_distill/loao_10app_results.pkl", "wb") as f:
    pickle.dump({"res": res, "orig5_sel4": saved_orig5_sel4}, f)
print("\nSaved /home/yjw/workspace/research/KSSM/exp2_scale_distill/loao_10app_results.pkl", flush=True)
