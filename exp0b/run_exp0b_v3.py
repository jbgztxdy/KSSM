import os, sys, time, math, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "3", "Must run with CUDA_VISIBLE_DEVICES=3!"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

PKL_PATH = "/home/yjw/workspace/research/KSSM/exp0b/dataset_exp0b_v3.pkl"
APPS = ["rocksdb", "sqlite", "pytorch", "llm", "etl"]

with open(PKL_PATH, "rb") as f:
    DATA = pickle.load(f)

def roc_auc_exact(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_s = y_score[order]
    i = 0; n = len(sorted_s)
    while i < n:
        j = i + 1
        while j < n and sorted_s[j] == sorted_s[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return float((np.sum(ranks[y_true == 1]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

def cls_metrics(y_true, y_pred, n_cls=3):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    acc = float(np.mean(y_true == y_pred)) if len(y_true) > 0 else 0.0
    recalls, f1s = [], []
    for c in range(n_cls):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        if tp + fn > 0:
            r = tp / (tp + fn)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recalls.append(r)
            f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    return acc, (float(np.mean(recalls)) if recalls else 0.0), (float(np.mean(f1s)) if f1s else 0.0)

WIN_LEN = 32
MAX_LOCAL_INOS = 10

def build_causal_windows(app_d, target_ev_indices, labels, max_samples=None):
    tid_arr = app_d["tid"]; inst_arr = app_d["inst"]; sg = app_d["s_gram"]; N = app_d["N"]
    idx_list = np.arange(len(target_ev_indices))
    if max_samples is not None and len(idx_list) > max_samples:
        rng = np.random.RandomState(42)
        uniq = np.unique(labels)
        per_c = max(1, max_samples // len(uniq))
        chosen = []
        for c in uniq:
            ci = idx_list[labels == c]
            if len(ci) > per_c:
                ci = rng.choice(ci, size=per_c, replace=False)
            chosen.append(ci)
        idx_list = np.sort(np.concatenate(chosen))

    sel_ev = target_ev_indices[idx_list]; sel_y = labels[idx_list]; M = len(sel_ev)
    tid_evs = defaultdict(list); inst_evs = defaultdict(list)
    for e_i in range(N):
        tid_evs[int(tid_arr[e_i])].append(e_i)
        inst_evs[int(inst_arr[e_i])].append(e_i)
    tid_evs = {k: np.array(v, dtype=np.int32) for k, v in tid_evs.items()}
    inst_evs = {k: np.array(v, dtype=np.int32) for k, v in inst_evs.items()}

    win_s = np.zeros((M, WIN_LEN, 5), dtype=np.int64)
    win_loc_ino = np.zeros((M, WIN_LEN), dtype=np.int64)
    win_same_tid = np.zeros((M, WIN_LEN), dtype=np.float32)
    win_valid = np.zeros((M, WIN_LEN), dtype=np.float32)

    for m in range(M):
        ev_i = int(sel_ev[m]); t_tgt = int(tid_arr[ev_i]); i_tgt = int(inst_arr[ev_i])
        te = tid_evs[t_tgt]; tp = int(np.searchsorted(te, ev_i, side="right"))
        t_hist = te[max(0, tp - 20):tp]
        ie = inst_evs[i_tgt]; ip = int(np.searchsorted(ie, ev_i, side="right"))
        i_hist = ie[max(0, ip - 18):ip]
        merged = np.unique(np.concatenate([t_hist, i_hist]))
        if len(merged) > WIN_LEN: merged = merged[-WIN_LEN:]
        L = len(merged); off = WIN_LEN - L
        local_map = {i_tgt: 0}; nxt = 1
        for k, ei in enumerate(merged):
            pos = off + k
            win_s[m, pos] = sg[ei]
            win_valid[m, pos] = 1.0
            win_same_tid[m, pos] = 1.0 if int(tid_arr[ei]) == t_tgt else 0.0
            ii = int(inst_arr[ei])
            if ii not in local_map:
                local_map[ii] = nxt if nxt < MAX_LOCAL_INOS else (MAX_LOCAL_INOS - 1)
                if nxt < MAX_LOCAL_INOS: nxt += 1
            win_loc_ino[m, pos] = local_map[ii]
    return {"ev": sel_ev, "y": sel_y, "win_s": win_s, "win_loc_ino": win_loc_ino,
            "win_same_tid": win_same_tid, "win_valid": win_valid}

SHARED_D = [1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4]
SHARED_C = [2,2,3,3, 3,3,4,4, 4,4,5,5, 5,5,5,5]
SUB_D    = [1,1,2,3, 1,2,3,4, 1,2,3,4, 1,2,3,4]
SUB_C    = [2,3,3,4, 2,3,4,5, 2,3,4,5, 2,3,4,5]

def get_band_limits(mode):
    lims = np.zeros(64, dtype=np.float32)
    ds = SHARED_D if mode == "swar_shared" else SUB_D
    cs = SHARED_C if mode == "swar_shared" else SUB_C
    n_slots = 5.0 if mode == "swar_shared" else 2.0
    for g in range(16):
        margin = (2.0 ** (-ds[g]) - 2.0 ** (-cs[g])) * 32.0 * 0.85
        lims[g*4:(g+1)*4] = margin / n_slots
    return lims

class KSSMModelV3(nn.Module):
    def __init__(self, mode="swar_shared", dim=64):
        super().__init__()
        self.mode = mode
        self.dim = dim
        self.emb_i = nn.Embedding(192, dim)
        self.emb_t = nn.Embedding(192, dim) if mode in ("float64", "uncoupled") else None
        nn.init.normal_(self.emb_i.weight, 0.0, 0.12)
        if self.emb_t is not None:
            nn.init.normal_(self.emb_t.weight, 0.0, 0.12)

        if mode in ("float64", "uncoupled", "inode_only"):
            self.logit_ai = nn.Parameter(torch.full((dim,), 1.0))
            self.logit_at = nn.Parameter(torch.full((dim,), 1.0))
            self.raw_g = nn.Parameter(torch.randn(dim) * 0.12)
            self.raw_e = nn.Parameter(torch.randn(dim) * 0.12)
            self.register_buffer("emb_lim", torch.full((dim,), 1.5))
        else:
            ds = SHARED_D if mode == "swar_shared" else SUB_D
            cs = SHARED_C if mode == "swar_shared" else SUB_C
            a = np.zeros(dim, dtype=np.float32); g = np.zeros(dim, dtype=np.float32)
            for grp in range(16):
                a[grp*4:(grp+1)*4] = 1.0 - (2.0 ** (-ds[grp]))
                g[grp*4:(grp+1)*4] = 2.0 ** (-cs[grp])
            self.register_buffer("fixed_a", torch.tensor(a))
            self.register_buffer("fixed_g", torch.tensor(g))
            self.register_buffer("emb_lim", torch.tensor(get_band_limits(mode)))

        self.drop = nn.Dropout(0.15)
        ro_in = dim if mode == "inode_only" else dim * 2
        self.head_life = nn.Linear(ro_in, 3)
        self.head_scan = nn.Linear(ro_in, 1)

    def tv_smoothness_loss(self):
        """1D Total-Variation smoothness across ordinal bucket neighborhoods so unseen/rare
        buckets extrapolate smoothly with ZERO runtime cost!"""
        w = torch.tanh(self.emb_i.weight) * self.emb_lim
        # s5 relative churn buckets: 160..175 (quiet) and 176..191 (active)
        l_s5 = F.mse_loss(w[160:175], w[161:176]) + F.mse_loss(w[176:191], w[177:192])
        # s2 stack recurrence dist 0..15 within each rw_q band
        l_s2 = 0.0
        for q in range(4):
            base = 32 + q * 16
            l_s2 = l_s2 + F.mse_loss(w[base:base+15], w[base+1:base+16])
        # s3 file size buckets (step 4 across 8 size buckets for each region)
        l_s3 = F.mse_loss(w[96:124], w[100:128])
        return 0.25 * (l_s5 + 0.5 * l_s2 + 0.5 * l_s3)

    def get_tables(self):
        ti = torch.tanh(self.emb_i.weight) * self.emb_lim
        tt = (torch.tanh(self.emb_t.weight) * self.emb_lim) if self.emb_t is not None else ti
        return ti, tt

    def forward_windows(self, win_s, win_loc_ino, win_same_tid, win_valid):
        B, W, _ = win_s.shape
        D = self.dim
        tbl_i, tbl_t = self.get_tables()
        offs = torch.tensor([0, 32, 96, 128, 160], device=win_s.device, dtype=torch.long)
        g_idx = win_s + offs
        if self.mode == "swar_subspace":
            w = tbl_i[g_idx]
            bi_all = torch.cat([
                w[:, :, 0, 0:16]  + w[:, :, 1, 0:16],
                w[:, :, 0, 16:32] + w[:, :, 2, 16:32],
                w[:, :, 0, 32:48] + w[:, :, 3, 32:48],
                w[:, :, 0, 48:64] + w[:, :, 4, 48:64],
            ], dim=-1)
            bt_all = bi_all
        else:
            bi_all = tbl_i[g_idx].sum(dim=2)
            bt_all = tbl_t[g_idx].sum(dim=2) if self.emb_t is not None else bi_all

        if self.mode in ("float64", "uncoupled", "inode_only"):
            ai = torch.sigmoid(self.logit_ai) * 0.90
            at = torch.sigmoid(self.logit_at) * 0.90
            if self.mode in ("uncoupled", "inode_only"):
                gm = torch.zeros_like(ai); et = torch.zeros_like(at)
            else:
                gm = torch.tanh(self.raw_g) * 0.25
                et = torch.tanh(self.raw_e) * 0.25
        else:
            ai = self.fixed_a; at = self.fixed_a
            gm = self.fixed_g; et = -self.fixed_g

        h_stack = torch.zeros(B, MAX_LOCAL_INOS, D, device=win_s.device)
        h_task = torch.zeros(B, D, device=win_s.device)
        one_hots = F.one_hot(win_loc_ino, MAX_LOCAL_INOS).to(bi_all.dtype) # [B, W, K]

        for t in range(W):
            v = win_valid[:, t:t+1]
            same_t = win_same_tid[:, t:t+1]
            mk = one_hots[:, t, :].unsqueeze(-1) # [B, K, 1]
            hi_old = (h_stack * mk).sum(dim=1)   # [B, D] fast gather without indexing
            ht_eff = h_task * same_t

            hi_new = ai * hi_old + gm * ht_eff + bi_all[:, t, :]
            ht_new = at * h_task + et * hi_old + bt_all[:, t, :]
            if self.mode in ("float64", "uncoupled", "inode_only"):
                hi_new = torch.clamp(hi_new, -32.0, 32.0)
                ht_new = torch.clamp(ht_new, -32.0, 32.0)

            hi_upd = hi_old * (1.0 - v) + hi_new * v
            h_stack = h_stack * (1.0 - mk) + hi_upd.unsqueeze(1) * mk
            upd_t = v * same_t
            h_task = h_task * (1.0 - upd_t) + ht_new * upd_t

        h_i0 = h_stack[:, 0, :]
        h_final = self.drop(h_i0 if self.mode == "inode_only" else torch.cat([h_i0, h_task], dim=-1))
        return self.head_life(h_final), self.head_scan(h_final).squeeze(-1)

    @torch.no_grad()
    def stream_eval_app(self, app_d, bit_exact_swar=False):
        N = app_d["N"]
        sg = app_d["s_gram"]; tids = app_d["tid"]; insts = app_d["inst"]
        n_t = app_d["num_tids"]; n_i = app_d["num_insts"]
        life_set = {int(e): idx for idx, e in enumerate(app_d["life_ev"])}
        scan_set = {int(e): idx for idx, e in enumerate(app_d["scan_ev"])}
        life_logits = np.zeros((len(app_d["life_ev"]), 3), dtype=np.float32)
        scan_scores = np.zeros(len(app_d["scan_ev"]), dtype=np.float32)
        ro_dim = self.dim if self.mode == "inode_only" else self.dim * 2
        life_feats = np.zeros((len(app_d["life_ev"]), ro_dim), dtype=np.float32)

        ti_t, tt_t = self.get_tables()
        ei_w = ti_t.detach().cpu().numpy(); et_w = tt_t.detach().cpu().numpy()
        w_life = self.head_life.weight.detach().cpu().numpy()
        b_life = self.head_life.bias.detach().cpu().numpy()
        w_scan = self.head_scan.weight.detach().cpu().numpy()[0]
        b_scan = float(self.head_scan.bias.detach().cpu().item())

        if bit_exact_swar and self.mode in ("swar_shared", "swar_subspace"):
            emb_u16 = (np.round(ei_w * 256.0).astype(np.int32) + 2048).astype(np.uint16)
            ds = np.repeat(SHARED_D if self.mode == "swar_shared" else SUB_D, 4).astype(np.int32)
            cs = np.repeat(SHARED_C if self.mode == "swar_shared" else SUB_C, 4).astype(np.int32)
            n_sl = 5 if self.mode == "swar_shared" else 2
            c_pos = (n_sl * 2048 + (16384 >> cs) - (16384 >> ds)).astype(np.int32)
            c_neg = (n_sl * 2048 - (16384 >> cs) - (16384 >> ds)).astype(np.int32)

            H_i = np.full((n_i, 64), 16384, dtype=np.int32)
            H_t = np.full((n_t, 64), 16384, dtype=np.int32)
            min_s, max_s = 16384, 16384
            wl_q = np.round(w_life * 256.0).astype(np.int32)
            bl_q = np.round(b_life * 65536.0).astype(np.int32)
            ws_q = np.round(w_scan * 256.0).astype(np.int32)
            bs_q = int(round(b_scan * 65536.0))

            for e in range(N):
                tid = int(tids[e]); iid = int(insts[e])
                s1 = int(sg[e, 0]); s2 = 32 + int(sg[e, 1])
                s3 = 96 + int(sg[e, 2]); s4 = 128 + int(sg[e, 3]); s5 = 160 + int(sg[e, 4])
                oi = H_i[iid]; ot = H_t[tid]
                if self.mode == "swar_shared":
                    b_sum = emb_u16[s1].astype(np.int32) + emb_u16[s2] + emb_u16[s3] + emb_u16[s4] + emb_u16[s5]
                else:
                    b_sum = np.empty(64, dtype=np.int32)
                    b_sum[0:16]  = emb_u16[s1, 0:16].astype(np.int32)  + emb_u16[s2, 0:16]
                    b_sum[16:32] = emb_u16[s1, 16:32].astype(np.int32) + emb_u16[s3, 16:32]
                    b_sum[32:48] = emb_u16[s1, 32:48].astype(np.int32) + emb_u16[s4, 32:48]
                    b_sum[48:64] = emb_u16[s1, 48:64].astype(np.int32) + emb_u16[s5, 48:64]
                ni = oi - (oi >> ds) + (ot >> cs) + b_sum - c_pos
                nt = ot - (ot >> ds) - (oi >> cs) + b_sum - c_neg
                H_i[iid] = ni; H_t[tid] = nt
                if (e & 511) == 0:
                    mn = min(int(ni.min()), int(nt.min())); mx = max(int(ni.max()), int(nt.max()))
                    if mn < min_s: min_s = mn
                    if mx > max_s: max_s = mx
                if e in life_set:
                    idx = life_set[e]; fq = np.concatenate([ni - 16384, nt - 16384])
                    life_logits[idx] = (wl_q @ fq + bl_q) / 65536.0
                    life_feats[idx] = fq / 256.0
                if e in scan_set:
                    idx = scan_set[e]; fq = np.concatenate([ni - 16384, nt - 16384])
                    scan_scores[idx] = float(np.dot(ws_q, fq) + bs_q) / 65536.0
            return life_logits, scan_scores, life_feats, (min_s, max_s)
        else:
            ai = (torch.sigmoid(self.logit_ai) * 0.90).detach().cpu().numpy()
            at = (torch.sigmoid(self.logit_at) * 0.90).detach().cpu().numpy()
            if self.mode in ("uncoupled", "inode_only"):
                gm = np.zeros_like(ai); et = np.zeros_like(at)
            else:
                gm = (torch.tanh(self.raw_g) * 0.25).detach().cpu().numpy()
                et = (torch.tanh(self.raw_e) * 0.25).detach().cpu().numpy()
            H_i = np.zeros((n_i, 64), dtype=np.float32); H_t = np.zeros((n_t, 64), dtype=np.float32)
            for e in range(N):
                tid = int(tids[e]); iid = int(insts[e])
                s1 = int(sg[e, 0]); s2 = 32 + int(sg[e, 1])
                s3 = 96 + int(sg[e, 2]); s4 = 128 + int(sg[e, 3]); s5 = 160 + int(sg[e, 4])
                oi = H_i[iid]; ot = H_t[tid]
                bi = ei_w[s1] + ei_w[s2] + ei_w[s3] + ei_w[s4] + ei_w[s5]
                bt = et_w[s1] + et_w[s2] + et_w[s3] + et_w[s4] + et_w[s5]
                ni = np.clip(ai * oi + gm * ot + bi, -32.0, 32.0)
                nt = np.clip(at * ot + et * oi + bt, -32.0, 32.0)
                H_i[iid] = ni; H_t[tid] = nt
                if e in life_set:
                    idx = life_set[e]; feat = ni if self.mode == "inode_only" else np.concatenate([ni, nt])
                    life_logits[idx] = w_life @ feat + b_life; life_feats[idx] = feat
                if e in scan_set:
                    idx = scan_set[e]; feat = ni if self.mode == "inode_only" else np.concatenate([ni, nt])
                    scan_scores[idx] = float(np.dot(w_scan, feat) + b_scan)
            return life_logits, scan_scores, life_feats, (0, 0)

class HeimdallMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.head_life = nn.Linear(32, 3)
        self.head_scan = nn.Linear(32, 1)
    def forward(self, x):
        h = self.net(x); return self.head_life(h), self.head_scan(h).squeeze(-1)

STATELESS_COLS = [0,1,2,3,4,5,6,7,8,10,11,12,13,15]
AUG_COLS       = list(range(16))

print("Building causal windows (WIN_LEN=32)...", flush=True)
WIN_LIFE = {app: build_causal_windows(DATA[app], DATA[app]["life_ev"], DATA[app]["life_y"], max_samples=450) for app in APPS}
WIN_SCAN = {app: build_causal_windows(DATA[app], DATA[app]["scan_ev"], DATA[app]["scan_y"], max_samples=1200) for app in APPS}

def train_kssm(train_apps, mode, epochs=35, lr=8e-3):
    def gather(wd):
        ws, wi, wt, wv, wy, ww = [], [], [], [], [], []
        for app in train_apps:
            a = wd[app]; y = a["y"]; cnt = Counter(y.tolist())
            w = np.array([1.0 / (len(cnt) * max(1, cnt[int(c)])) for c in y], dtype=np.float32)
            ws.append(a["win_s"]); wi.append(a["win_loc_ino"])
            wt.append(a["win_same_tid"]); wv.append(a["win_valid"])
            wy.append(y); ww.append(w)
        return [torch.tensor(np.concatenate(x), device=DEVICE) for x in (ws, wi, wt, wv, wy, ww)]
    ls, li, lt, lv, ly, lw = gather(WIN_LIFE)
    ss, si, st, sv, sy, sw = gather(WIN_SCAN)

    model = KSSMModelV3(mode=mode, dim=64).to(DEVICE)
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
            ll, _ = model.forward_windows(ls[il], li[il], lt[il], lv[il])
            l1 = (F.cross_entropy(ll, ly[il], reduction="none") * lw[il]).sum() / (lw[il].sum() + 1e-8)
            _, lso = model.forward_windows(ss[is_], si[is_], st[is_], sv[is_])
            l2 = (F.binary_cross_entropy_with_logits(lso, sy[is_].float(), reduction="none") * sw[is_]).sum() / (sw[is_].sum() + 1e-8)
            loss = l1 + l2 + model.tv_smoothness_loss()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()
    model.eval()
    return model

def train_heimdall(train_apps, cols, epochs=40, lr=5e-3):
    xl, yl, wl, xs, ys, ws = [], [], [], [], [], []
    for app in train_apps:
        d = DATA[app]; wdl = WIN_LIFE[app]; wds = WIN_SCAN[app]
        yla = wdl["y"]; cl = Counter(yla.tolist())
        wla = np.array([1.0 / (len(cl) * max(1, cl[int(c)])) for c in yla], dtype=np.float32)
        xl.append(d["scalars"][wdl["ev"]][:, cols]); yl.append(yla); wl.append(wla)
        ysa = wds["y"]; cs = Counter(ysa.tolist())
        wsa = np.array([1.0 / (len(cs) * max(1, cs[int(c)])) for c in ysa], dtype=np.float32)
        xs.append(d["scalars"][wds["ev"]][:, cols]); ys.append(ysa); ws.append(wsa)
    xl, yl, wl, xs, ys, ws = [torch.tensor(np.concatenate(x), device=DEVICE) for x in (xl, yl, wl, xs, ys, ws)]
    mlp = HeimdallMLP(len(cols)).to(DEVICE)
    opt = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    B = 512; nl, ns = len(yl), len(ys)
    for ep in range(epochs):
        mlp.train()
        pl = torch.randperm(nl, device=DEVICE); ps = torch.randperm(ns, device=DEVICE)
        for step in range(max(nl // B, ns // B, 4)):
            il = pl[(step * B) % nl : ((step * B) % nl) + B]
            is_ = ps[(step * B) % ns : ((step * B) % ns) + B]
            if len(il) < 32: il = pl[:min(B, nl)]
            if len(is_) < 32: is_ = ps[:min(B, ns)]
            ll, _ = mlp(xl[il])
            l1 = (F.cross_entropy(ll, yl[il], reduction="none") * wl[il]).sum() / (wl[il].sum() + 1e-8)
            _, lso = mlp(xs[is_])
            l2 = (F.binary_cross_entropy_with_logits(lso, ys[is_].float(), reduction="none") * ws[is_]).sum() / (ws[is_].sum() + 1e-8)
            opt.zero_grad(); (l1 + l2).backward(); opt.step()
        sched.step()
    mlp.eval()
    return mlp

def eval_pcstream_zs(train_apps, test_app):
    pc_tbl = defaultdict(lambda: np.zeros(3)); gcnt = np.zeros(3)
    for app in train_apps:
        d = DATA[app]
        for ei, y in zip(d["life_ev"], d["life_y"]):
            pc_tbl[int(d["pc_sigs"][ei])][int(y)] += 1; gcnt[int(y)] += 1
    def_c = int(np.argmax(gcnt)); td = DATA[test_app]
    return np.array([int(np.argmax(pc_tbl[int(td["pc_sigs"][ei])])) if int(td["pc_sigs"][ei]) in pc_tbl else def_c for ei in td["life_ev"]], dtype=np.int64)

def eval_basic(train_apps, test_app):
    lt = defaultdict(lambda: np.ones(3)*0.5); lu = defaultdict(lambda: np.ones(3)*0.5)
    st = defaultdict(lambda: np.ones(2)*0.5); su = defaultdict(lambda: np.ones(2)*0.5)
    for app in train_apps:
        d = DATA[app]; toks = d["basic_tok"]; tids = d["tid"]
        pt = {}; prev = np.full(d["N"], -1, dtype=np.int32)
        for i in range(d["N"]):
            t = int(tids[i]); prev[i] = pt.get(t, -1); pt[t] = int(toks[i])
        cl = Counter(d["life_y"].tolist())
        for ei, y in zip(d["life_ev"], d["life_y"]):
            w = 100.0 / max(1, cl[int(y)]); c = int(toks[ei]); p = int(prev[ei])
            lt[(p, c)][int(y)] += w; lu[c][int(y)] += w
        sy = d["scan_y"]; cs = Counter(sy.tolist()); step = max(1, len(sy)//4000)
        for ei, y in zip(d["scan_ev"][::step], sy[::step]):
            w = 100.0 / max(1, cs[int(y)]//step); c = int(toks[ei]); p = int(prev[ei])
            st[(p, c)][int(y)] += w; su[c][int(y)] += w
    td = DATA[test_app]; toks = td["basic_tok"]; tids = td["tid"]
    pt = {}; prev = np.full(td["N"], -1, dtype=np.int32)
    for i in range(td["N"]):
        t = int(tids[i]); prev[i] = pt.get(t, -1); pt[t] = int(toks[i])
    lp = [int(np.argmax(lt[(int(prev[ei]), int(toks[ei]))] if (int(prev[ei]), int(toks[ei])) in lt else lu[int(toks[ei])])) for ei in td["life_ev"]]
    ss = [float((st[(int(prev[ei]), int(toks[ei]))] if (int(prev[ei]), int(toks[ei])) in st else su[int(toks[ei])])[1] /
                np.sum(st[(int(prev[ei]), int(toks[ei]))] if (int(prev[ei]), int(toks[ei])) in st else su[int(toks[ei])])) for ei in td["scan_ev"]]
    return np.array(lp, dtype=np.int64), np.array(ss, dtype=np.float32)

METHODS = [
    "PCStream (FAST'19)",
    "Heimdall (EuroSys'25, Faithful)",
    "Heimdall+ (w/ RelChurn+RW)",
    "BASIC-Prefetch (SYSTOR'26)",
    "KSSM-InodeOnly (No h_task)",
    "KSSM-Uncoupled (gamma=eta=0)",
    "KSSM-Float64D (Coupled)",
    "KSSM-GuardSWAR-Shared(58ns)",
    "KSSM-GuardSWAR-Subspace(46ns)",
]
res = {m: {"acc": [], "bal": [], "f1": [], "auc": []} for m in METHODS}
store = {}

print("\n" + "="*118, flush=True)
print(" EXP 0-B (v3: Scale-Invariant Relative Grammar + TV-Smoothed Tables) — 5-Fold LOAO Zero-Shot Showdown", flush=True)
print("="*118, flush=True)

for held_out in APPS:
    t0 = time.time()
    train_apps = [a for a in APPS if a != held_out]
    td = DATA[held_out]; yl = td["life_y"]; ys = td["scan_y"]
    print(f"\n>>> FOLD: Held-Out Unseen App = {held_out.upper()} (life_0s={len(yl)}, scan={len(ys)})", flush=True)

    p = eval_pcstream_zs(train_apps, held_out)
    a, b, f1 = cls_metrics(yl, p)
    res["PCStream (FAST'19)"]["acc"].append(a); res["PCStream (FAST'19)"]["bal"].append(b)
    res["PCStream (FAST'19)"]["f1"].append(f1); res["PCStream (FAST'19)"]["auc"].append(0.5)
    print(f"  {'PCStream (FAST19)':32s} | Acc={a*100:5.1f}% BalAcc={b*100:5.1f}% F1={f1*100:5.1f}% | AUC=  N/A", flush=True)

    for hn, cols in [("Heimdall (EuroSys'25, Faithful)", STATELESS_COLS), ("Heimdall+ (w/ RelChurn+RW)", AUG_COLS)]:
        mlp = train_heimdall(train_apps, cols)
        with torch.no_grad():
            ll, _ = mlp(torch.tensor(td["scalars"][td["life_ev"]][:, cols], device=DEVICE))
            _, ls = mlp(torch.tensor(td["scalars"][td["scan_ev"]][:, cols], device=DEVICE))
        p = ll.argmax(-1).cpu().numpy(); s = ls.cpu().numpy()
        a, b, f1 = cls_metrics(yl, p); auc = roc_auc_exact(ys, s)
        res[hn]["acc"].append(a); res[hn]["bal"].append(b); res[hn]["f1"].append(f1); res[hn]["auc"].append(auc)
        print(f"  {hn:32s} | Acc={a*100:5.1f}% BalAcc={b*100:5.1f}% F1={f1*100:5.1f}% | AUC={auc:.4f}", flush=True)

    p, s = eval_basic(train_apps, held_out)
    a, b, f1 = cls_metrics(yl, p); auc = roc_auc_exact(ys, s)
    res["BASIC-Prefetch (SYSTOR'26)"]["acc"].append(a); res["BASIC-Prefetch (SYSTOR'26)"]["bal"].append(b)
    res["BASIC-Prefetch (SYSTOR'26)"]["f1"].append(f1); res["BASIC-Prefetch (SYSTOR'26)"]["auc"].append(auc)
    print(f"  {'BASIC-Prefetch (SYSTOR26)':32s} | Acc={a*100:5.1f}% BalAcc={b*100:5.1f}% F1={f1*100:5.1f}% | AUC={auc:.4f}", flush=True)

    for lbl, mode, bit_ex in [
        ("KSSM-InodeOnly (No h_task)",     "inode_only",    False),
        ("KSSM-Uncoupled (gamma=eta=0)",   "uncoupled",     False),
        ("KSSM-Float64D (Coupled)",        "float64",       False),
        ("KSSM-GuardSWAR-Shared(58ns)",    "swar_shared",   True),
        ("KSSM-GuardSWAR-Subspace(46ns)",  "swar_subspace", True),
    ]:
        km = train_kssm(train_apps, mode=mode)
        ll, sc, lf, bnd = km.stream_eval_app(td, bit_exact_swar=bit_ex)
        p = np.argmax(ll, axis=-1)
        a, b, f1 = cls_metrics(yl, p); auc = roc_auc_exact(ys, sc)
        res[lbl]["acc"].append(a); res[lbl]["bal"].append(b); res[lbl]["f1"].append(f1); res[lbl]["auc"].append(auc)
        bstr = f" [SWAR={bnd}]" if bit_ex else ""
        print(f"  {lbl:32s} | Acc={a*100:5.1f}% BalAcc={b*100:5.1f}% F1={f1*100:5.1f}% | AUC={auc:.4f}{bstr}", flush=True)
        if lbl == "KSSM-GuardSWAR-Shared(58ns)":
            store[held_out] = {"pred": p, "feat": lf, "w": km.head_life.weight.detach().cpu().numpy(), "b": km.head_life.bias.detach().cpu().numpy()}
    print(f"  [Fold done in {time.time()-t0:.1f}s]", flush=True)

print("\n" + "="*118, flush=True)
print(" SUMMARY TABLE (v3): 5-App LOAO Zero-Shot Macro-Average (Unseen App, 0th-Second Prediction)", flush=True)
print("="*118, flush=True)
print(f"{'Method':34s} | {'Life0s OverallAcc':>17s} | {'Life0s BalAcc':>13s} | {'Life0s MacroF1':>14s} | {'One-Shot AUC':>12s}", flush=True)
print("-"*118, flush=True)
for m in METHODS:
    r = res[m]
    as_ = f"{np.mean(r['auc']):.4f}" if m != "PCStream (FAST'19)" else "N/A"
    print(f"{m:34s} | {np.mean(r['acc'])*100:16.2f}% | {np.mean(r['bal'])*100:12.2f}% | {np.mean(r['f1'])*100:13.2f}% | {as_:>12s}", flush=True)

print("\n" + "="*118, flush=True)
print(" PART 2: Workload-Shift Cold-Start Blind-Window Showdown (Balanced Acc % across 5 unseen apps)", flush=True)
print("="*118, flush=True)
windows = [(0.0, 10.0), (10.0, 30.0), (30.0, 65.0), (0.0, 65.0)]
tr = {k: defaultdict(list) for k in [
    "eFDP (replan=420s, paper)", "eFDP (replan=30s, agile)", "PCStream (Online Death)",
    "KSSM-GuardSWAR (Zero-Shot)", "KSSM-GuardSWAR (+SlowLoop)"
]}
for app in APPS:
    d = DATA[app]; y = d["life_y"]; tb = d["life_birth_s"]; td_s = d["life_death_s"]; pcs = d["pc_sigs"][d["life_ev"]]; n = len(y)
    for name, trep in [("eFDP (replan=420s, paper)", 420.0), ("eFDP (replan=30s, agile)", 30.0)]:
        preds = np.zeros(n, dtype=np.int64)
        obs_d = sorted([(td_s[i], int(y[i])) for i in range(n) if td_s[i] >= 0])
        lc = 0
        for i in range(n):
            if tb[i] >= trep:
                bd = math.floor(tb[i] / trep) * trep
                ob = [c for dt_t, c in obs_d if dt_t <= bd]
                if ob: lc = Counter(ob).most_common(1)[0][0]
            preds[i] = lc
        for wi, (lo, hi) in enumerate(windows):
            m = (tb >= lo) & (tb < hi)
            if np.sum(m) > 0: tr[name][wi].append(cls_metrics(y[m], preds[m])[1])

    pc_t = defaultdict(lambda: np.zeros(3))
    pend = sorted([(td_s[i], int(pcs[i]), int(y[i])) for i in range(n) if td_s[i] >= 0])
    dp = 0; p_pc = np.zeros(n, dtype=np.int64)
    for i in range(n):
        while dp < len(pend) and pend[dp][0] <= tb[i]:
            pc_t[pend[dp][1]][pend[dp][2]] += 1; dp += 1
        p_pc[i] = int(np.argmax(pc_t[int(pcs[i])])) if int(pcs[i]) in pc_t else 0
    for wi, (lo, hi) in enumerate(windows):
        m = (tb >= lo) & (tb < hi)
        if np.sum(m) > 0: tr["PCStream (Online Death)"][wi].append(cls_metrics(y[m], p_pc[m])[1])

    pk0 = store[app]["pred"]
    for wi, (lo, hi) in enumerate(windows):
        m = (tb >= lo) & (tb < hi)
        if np.sum(m) > 0: tr["KSSM-GuardSWAR (Zero-Shot)"][wi].append(cls_metrics(y[m], pk0[m])[1])

    W = store[app]["w"].copy(); b = store[app]["b"].copy(); feats = store[app]["feat"]
    fb = sorted([(td_s[i], i, int(y[i])) if td_s[i] >= 0 else (tb[i] + 22.0, i, 2) for i in range(n)])
    fp = 0; pkon = np.zeros(n, dtype=np.int64)
    for i in range(n):
        while fp < len(fb) and fb[fp][0] <= tb[i]:
            _, idx_f, y_f = fb[fp]; xf = feats[idx_f]
            lg = W @ xf + b; lg -= lg.max(); pr = np.exp(lg); pr /= pr.sum()
            gr = pr; gr[y_f] -= 1.0; cw = 2.0 if y_f > 0 else 0.5
            W -= (0.05 * cw) * np.outer(gr, xf); b -= (0.05 * cw) * gr; fp += 1
        pkon[i] = int(np.argmax(W @ feats[i] + b))
    for wi, (lo, hi) in enumerate(windows):
        m = (tb >= lo) & (tb < hi)
        if np.sum(m) > 0: tr["KSSM-GuardSWAR (+SlowLoop)"][wi].append(cls_metrics(y[m], pkon[m])[1])

print(f"{'Method':34s} | {'0-10s (Shift)':>14s} | {'10-30s':>12s} | {'30-65s':>12s} | {'Full 0-65s':>12s}", flush=True)
print("-"*118, flush=True)
for k, dw in tr.items():
    vs = [f"{np.mean(dw[wi])*100:11.2f}%" for wi in range(4)]
    print(f"{k:34s} | {vs[0]:>14s} | {vs[1]:>12s} | {vs[2]:>12s} | {vs[3]:>12s}", flush=True)

# Train final production Guard-SWAR-Shared model on all 5 apps & export uint16 baked tables for Exp 0-C!
print("\nExporting full-trained Guard-SWAR-Shared uint16 lookup tables for Exp 0-C...", flush=True)
prod_m = train_kssm(APPS, mode="swar_shared", epochs=40, lr=8e-3)
ti, _ = prod_m.get_tables()
emb_u16 = (np.round(ti.detach().cpu().numpy() * 256.0).astype(np.int32) + 2048).astype(np.uint16)
np.savez("/home/yjw/workspace/research/KSSM/exp0b/kssm_baked_weights.npz",
         emb_u16=emb_u16,
         w_scan=prod_m.head_scan.weight.detach().cpu().numpy()[0],
         b_scan=float(prod_m.head_scan.bias.detach().cpu().item()),
         w_life=prod_m.head_life.weight.detach().cpu().numpy(),
         b_life=prod_m.head_life.bias.detach().cpu().numpy())
print("Exported /home/yjw/workspace/research/KSSM/exp0b/kssm_baked_weights.npz successfully!", flush=True)
