import os, sys, time, math, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict

# Enforce GPU 3 strictly
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "3", "Must run with CUDA_VISIBLE_DEVICES=3!"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

PKL_PATH = "/home/yjw/workspace/research/KSSM/exp0b/dataset_exp0b.pkl"
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
    i = 0
    n = len(sorted_s)
    while i < n:
        j = i + 1
        while j < n and sorted_s[j] == sorted_s[i]:
            j += 1
        avg_r = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_r
        i = j
    sum_pos_ranks = np.sum(ranks[y_true == 1])
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

def cls_metrics(y_true, y_pred, n_cls=3):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    acc = float(np.mean(y_true == y_pred)) if len(y_true) > 0 else 0.0
    recalls = []
    f1s = []
    for c in range(n_cls):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        if tp + fn > 0:
            r = tp / (tp + fn)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recalls.append(r)
            f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    bal_acc = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return acc, bal_acc, macro_f1

WIN_LEN = 32
MAX_LOCAL_INOS = 12

def build_causal_windows(app_d, target_ev_indices, labels, max_samples=None, balanced=False):
    tid_arr = app_d["tid"]
    inst_arr = app_d["inst"]
    sg = app_d["s_gram"]
    N = app_d["N"]

    idx_list = np.arange(len(target_ev_indices))
    if max_samples is not None and len(idx_list) > max_samples:
        if balanced:
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
        else:
            step = np.linspace(0, len(idx_list) - 1, max_samples, dtype=np.int64)
            idx_list = idx_list[step]

    sel_ev = target_ev_indices[idx_list]
    sel_y = labels[idx_list]
    M = len(sel_ev)

    tid_evs = defaultdict(list)
    inst_evs = defaultdict(list)
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
        ev_i = int(sel_ev[m])
        t_tgt = int(tid_arr[ev_i])
        i_tgt = int(inst_arr[ev_i])

        te = tid_evs[t_tgt]
        tp = int(np.searchsorted(te, ev_i, side="right"))
        t_hist = te[max(0, tp - 24):tp]

        ie = inst_evs[i_tgt]
        ip = int(np.searchsorted(ie, ev_i, side="right"))
        i_hist = ie[max(0, ip - 16):ip]

        merged = np.unique(np.concatenate([t_hist, i_hist]))
        if len(merged) > WIN_LEN:
            merged = merged[-WIN_LEN:]

        L = len(merged)
        off = WIN_LEN - L
        local_map = {i_tgt: 0}
        nxt_slot = 1
        for k, ei in enumerate(merged):
            pos = off + k
            win_s[m, pos] = sg[ei]
            win_valid[m, pos] = 1.0
            win_same_tid[m, pos] = 1.0 if int(tid_arr[ei]) == t_tgt else 0.0
            ii = int(inst_arr[ei])
            if ii not in local_map:
                local_map[ii] = nxt_slot if nxt_slot < MAX_LOCAL_INOS else (MAX_LOCAL_INOS - 1)
                if nxt_slot < MAX_LOCAL_INOS:
                    nxt_slot += 1
            win_loc_ino[m, pos] = local_map[ii]

    return {
        "ev": sel_ev,
        "y": sel_y,
        "win_s": win_s,
        "win_loc_ino": win_loc_ino,
        "win_same_tid": win_same_tid,
        "win_valid": win_valid,
    }

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

class KSSMModel(nn.Module):
    def __init__(self, mode="float64", dim=64):
        super().__init__()
        self.mode = mode
        self.dim = dim
        self.emb_i = nn.Embedding(192, dim)
        self.emb_t = nn.Embedding(192, dim) if mode in ("float64", "no_coupling") else None
        nn.init.normal_(self.emb_i.weight, 0.0, 0.15)
        if self.emb_t is not None:
            nn.init.normal_(self.emb_t.weight, 0.0, 0.15)

        if mode in ("float64", "no_coupling"):
            self.logit_ai = nn.Parameter(torch.full((32, dim), 1.2))
            self.logit_at = nn.Parameter(torch.full((32, dim), 1.2))
            self.raw_g = nn.Parameter(torch.randn(32, dim) * 0.15)
            self.raw_e = nn.Parameter(torch.randn(32, dim) * 0.15)
        else:
            ds = SHARED_D if mode == "swar_shared" else SUB_D
            cs = SHARED_C if mode == "swar_shared" else SUB_C
            a = np.zeros(dim, dtype=np.float32)
            g = np.zeros(dim, dtype=np.float32)
            for grp in range(16):
                a[grp*4:(grp+1)*4] = 1.0 - (2.0 ** (-ds[grp]))
                g[grp*4:(grp+1)*4] = 2.0 ** (-cs[grp])
            self.register_buffer("fixed_a", torch.tensor(a))
            self.register_buffer("fixed_g", torch.tensor(g))
            self.register_buffer("emb_lim", torch.tensor(get_band_limits(mode)))

        self.head_life = nn.Linear(dim * 2, 3)
        self.head_scan = nn.Linear(dim * 2, 1)

    def get_constrained_emb(self, s_idx, is_task=False):
        offs = torch.tensor([0, 32, 96, 128, 160], device=s_idx.device, dtype=torch.long)
        g_idx = s_idx + offs
        if self.mode in ("float64", "no_coupling"):
            tbl = self.emb_t if (is_task and self.emb_t is not None) else self.emb_i
            return tbl(g_idx).sum(dim=1)
        elif self.mode == "swar_shared":
            w = torch.tanh(self.emb_i(g_idx)) * self.emb_lim
            return w.sum(dim=1)
        elif self.mode == "swar_subspace":
            w = torch.tanh(self.emb_i(g_idx)) * self.emb_lim
            b0 = w[:, 0, 0:16]  + w[:, 1, 0:16]
            b1 = w[:, 0, 16:32] + w[:, 2, 16:32]
            b2 = w[:, 0, 32:48] + w[:, 3, 32:48]
            b3 = w[:, 0, 48:64] + w[:, 4, 48:64]
            return torch.cat([b0, b1, b2, b3], dim=-1)

    def forward_windows(self, win_s, win_loc_ino, win_same_tid, win_valid):
        B, W, _ = win_s.shape
        D = self.dim
        h_inos = torch.zeros(B, MAX_LOCAL_INOS, D, device=win_s.device)
        h_task = torch.zeros(B, D, device=win_s.device)
        b_arange = torch.arange(B, device=win_s.device)

        for t in range(W):
            v = win_valid[:, t:t+1]
            if v.sum() == 0:
                continue
            st = win_s[:, t, :]
            ino_slot = win_loc_ino[:, t]
            same_t = win_same_tid[:, t:t+1]

            hi_old = h_inos[b_arange, ino_slot]
            ht_eff = h_task * same_t

            bi = self.get_constrained_emb(st, is_task=False)
            if self.mode in ("float64", "no_coupling"):
                bt = self.get_constrained_emb(st, is_task=True)
                s1 = st[:, 0]
                ai = torch.sigmoid(self.logit_ai[s1]) * 0.96
                at = torch.sigmoid(self.logit_at[s1]) * 0.96
                if self.mode == "no_coupling":
                    gm = torch.zeros_like(ai)
                    et = torch.zeros_like(at)
                else:
                    gm = torch.tanh(self.raw_g[s1]) * 0.35
                    et = torch.tanh(self.raw_e[s1]) * 0.35
                hi_new = torch.clamp(ai * hi_old + gm * ht_eff + bi, -32.0, 32.0)
                ht_new = torch.clamp(at * h_task + et * hi_old + bt, -32.0, 32.0)
            else:
                hi_new = self.fixed_a * hi_old + self.fixed_g * ht_eff + bi
                ht_new = self.fixed_a * h_task - self.fixed_g * hi_old + bi

            hi_upd = hi_old * (1.0 - v) + hi_new * v
            h_inos = h_inos.clone()
            h_inos[b_arange, ino_slot] = hi_upd
            upd_t = v * same_t
            h_task = h_task * (1.0 - upd_t) + ht_new * upd_t

        h_final = torch.cat([h_inos[:, 0, :], h_task], dim=-1)
        return self.head_life(h_final), self.head_scan(h_final).squeeze(-1)

    @torch.no_grad()
    def stream_eval_app(self, app_d, bit_exact_swar=False):
        N = app_d["N"]
        sg = app_d["s_gram"]
        tids = app_d["tid"]
        insts = app_d["inst"]
        n_t = app_d["num_tids"]
        n_i = app_d["num_insts"]

        life_set = {int(e): idx for idx, e in enumerate(app_d["life_ev"])}
        scan_set = {int(e): idx for idx, e in enumerate(app_d["scan_ev"])}
        life_logits = np.zeros((len(app_d["life_ev"]), 3), dtype=np.float32)
        scan_scores = np.zeros(len(app_d["scan_ev"]), dtype=np.float32)
        life_feats  = np.zeros((len(app_d["life_ev"]), self.dim * 2), dtype=np.float32)

        w_life = self.head_life.weight.detach().cpu().numpy()
        b_life = self.head_life.bias.detach().cpu().numpy()
        w_scan = self.head_scan.weight.detach().cpu().numpy()[0]
        b_scan = float(self.head_scan.bias.detach().cpu().item())

        if bit_exact_swar and self.mode in ("swar_shared", "swar_subspace"):
            w_float = (torch.tanh(self.emb_i.weight) * self.emb_lim).detach().cpu().numpy()
            emb_q = np.round(w_float * 256.0).astype(np.int32)
            emb_u16 = (emb_q + 2048).astype(np.uint16)

            ds = np.repeat(SHARED_D if self.mode == "swar_shared" else SUB_D, 4).astype(np.int32)
            cs = np.repeat(SHARED_C if self.mode == "swar_shared" else SUB_C, 4).astype(np.int32)
            n_sl = 5 if self.mode == "swar_shared" else 2
            c_pos = (n_sl * 2048 + (16384 >> cs) - (16384 >> ds)).astype(np.int32)
            c_neg = (n_sl * 2048 - (16384 >> cs) - (16384 >> ds)).astype(np.int32)

            H_i = np.full((n_i, 64), 16384, dtype=np.int32)
            H_t = np.full((n_t, 64), 16384, dtype=np.int32)
            min_state, max_state = 16384, 16384

            wl_q = np.round(w_life * 256.0).astype(np.int32)
            bl_q = np.round(b_life * 65536.0).astype(np.int32)
            ws_q = np.round(w_scan * 256.0).astype(np.int32)
            bs_q = int(round(b_scan * 65536.0))

            for e in range(N):
                tid = int(tids[e]); iid = int(insts[e])
                s1 = int(sg[e, 0])
                s2 = 32 + int(sg[e, 1])
                s3 = 96 + int(sg[e, 2])
                s4 = 128 + int(sg[e, 3])
                s5 = 160 + int(sg[e, 4])

                oi = H_i[iid]
                ot = H_t[tid]

                if self.mode == "swar_shared":
                    b_sum = (emb_u16[s1].astype(np.int32) + emb_u16[s2] +
                             emb_u16[s3] + emb_u16[s4] + emb_u16[s5])
                else:
                    b_sum = np.empty(64, dtype=np.int32)
                    b_sum[0:16]  = emb_u16[s1, 0:16].astype(np.int32)  + emb_u16[s2, 0:16]
                    b_sum[16:32] = emb_u16[s1, 16:32].astype(np.int32) + emb_u16[s3, 16:32]
                    b_sum[32:48] = emb_u16[s1, 32:48].astype(np.int32) + emb_u16[s4, 32:48]
                    b_sum[48:64] = emb_u16[s1, 48:64].astype(np.int32) + emb_u16[s5, 48:64]

                ni = oi - (oi >> ds) + (ot >> cs) + b_sum - c_pos
                nt = ot - (ot >> ds) - (oi >> cs) + b_sum - c_neg
                H_i[iid] = ni
                H_t[tid] = nt

                if (e & 255) == 0:
                    mn = min(int(ni.min()), int(nt.min()))
                    mx = max(int(ni.max()), int(nt.max()))
                    if mn < min_state: min_state = mn
                    if mx > max_state: max_state = mx

                if e in life_set:
                    idx = life_set[e]
                    feat_q = np.concatenate([ni - 16384, nt - 16384])
                    life_logits[idx] = (wl_q @ feat_q + bl_q) / 65536.0
                    life_feats[idx] = feat_q / 256.0
                if e in scan_set:
                    idx = scan_set[e]
                    feat_q = np.concatenate([ni - 16384, nt - 16384])
                    scan_scores[idx] = float(np.dot(ws_q, feat_q) + bs_q) / 65536.0

            return life_logits, scan_scores, life_feats, (min_state, max_state)
        else:
            ei_w = self.emb_i.weight.detach().cpu().numpy()
            et_w = self.emb_t.weight.detach().cpu().numpy() if self.emb_t is not None else ei_w
            ai_w = (torch.sigmoid(self.logit_ai) * 0.96).detach().cpu().numpy()
            at_w = (torch.sigmoid(self.logit_at) * 0.96).detach().cpu().numpy()
            if self.mode == "no_coupling":
                gm_w = np.zeros_like(ai_w); et_c = np.zeros_like(at_w)
            else:
                gm_w = (torch.tanh(self.raw_g) * 0.35).detach().cpu().numpy()
                et_c = (torch.tanh(self.raw_e) * 0.35).detach().cpu().numpy()

            H_i = np.zeros((n_i, 64), dtype=np.float32)
            H_t = np.zeros((n_t, 64), dtype=np.float32)
            for e in range(N):
                tid = int(tids[e]); iid = int(insts[e])
                s1 = int(sg[e, 0])
                s2 = 32 + int(sg[e, 1]); s3 = 96 + int(sg[e, 2])
                s4 = 128 + int(sg[e, 3]); s5 = 160 + int(sg[e, 4])
                oi = H_i[iid]; ot = H_t[tid]
                bi = ei_w[s1] + ei_w[s2] + ei_w[s3] + ei_w[s4] + ei_w[s5]
                bt = et_w[s1] + et_w[s2] + et_w[s3] + et_w[s4] + et_w[s5]
                ni = np.clip(ai_w[s1] * oi + gm_w[s1] * ot + bi, -32.0, 32.0)
                nt = np.clip(at_w[s1] * ot + et_c[s1] * oi + bt, -32.0, 32.0)
                H_i[iid] = ni; H_t[tid] = nt
                if e in life_set:
                    idx = life_set[e]
                    feat = np.concatenate([ni, nt])
                    life_logits[idx] = w_life @ feat + b_life
                    life_feats[idx] = feat
                if e in scan_set:
                    idx = scan_set[e]
                    feat = np.concatenate([ni, nt])
                    scan_scores[idx] = float(np.dot(w_scan, feat) + b_scan)
            return life_logits, scan_scores, life_feats, (0, 0)

class HeimdallMLP(nn.Module):
    def __init__(self, in_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.head_life = nn.Linear(32, 3)
        self.head_scan = nn.Linear(32, 1)

    def forward(self, x):
        h = self.net(x)
        return self.head_life(h), self.head_scan(h).squeeze(-1)

def eval_basic_markov(train_apps, test_app):
    life_trans = defaultdict(lambda: np.ones(3, dtype=np.float64) * 0.5)
    life_uni   = defaultdict(lambda: np.ones(3, dtype=np.float64) * 0.5)
    scan_trans = defaultdict(lambda: np.ones(2, dtype=np.float64) * 0.5)
    scan_uni   = defaultdict(lambda: np.ones(2, dtype=np.float64) * 0.5)

    for app in train_apps:
        d = DATA[app]
        toks = d["basic_tok"]; tids = d["tid"]
        prev_tok_t = {}
        prev_of_ev = np.full(d["N"], -1, dtype=np.int32)
        for i in range(d["N"]):
            t = int(tids[i])
            prev_of_ev[i] = prev_tok_t.get(t, -1)
            prev_tok_t[t] = int(toks[i])

        ly = d["life_y"]
        cnt_l = Counter(ly.tolist())
        for ev_i, y in zip(d["life_ev"], ly):
            w = 100.0 / max(1, cnt_l[int(y)])
            cur = int(toks[ev_i]); prv = int(prev_of_ev[ev_i])
            life_trans[(prv, cur)][int(y)] += w
            life_uni[cur][int(y)] += w

        sy = d["scan_y"]
        cnt_s = Counter(sy.tolist())
        step = max(1, len(sy) // 4000)
        for ev_i, y in zip(d["scan_ev"][::step], sy[::step]):
            w = 100.0 / max(1, cnt_s[int(y)] // step)
            cur = int(toks[ev_i]); prv = int(prev_of_ev[ev_i])
            scan_trans[(prv, cur)][int(y)] += w
            scan_uni[cur][int(y)] += w

    td = DATA[test_app]
    toks = td["basic_tok"]; tids = td["tid"]
    prev_tok_t = {}
    prev_of_ev = np.full(td["N"], -1, dtype=np.int32)
    for i in range(td["N"]):
        t = int(tids[i])
        prev_of_ev[i] = prev_tok_t.get(t, -1)
        prev_tok_t[t] = int(toks[i])

    life_preds = []
    for ev_i in td["life_ev"]:
        cur = int(toks[ev_i]); prv = int(prev_of_ev[ev_i])
        p = life_trans[(prv, cur)] if (prv, cur) in life_trans else (life_uni[cur] if cur in life_uni else np.array([1.0, 0.0, 0.0]))
        life_preds.append(int(np.argmax(p)))

    scan_scores = []
    for ev_i in td["scan_ev"]:
        cur = int(toks[ev_i]); prv = int(prev_of_ev[ev_i])
        p = scan_trans[(prv, cur)] if (prv, cur) in scan_trans else (scan_uni[cur] if cur in scan_uni else np.array([1.0, 1.0]))
        scan_scores.append(float(p[1] / (p[0] + p[1])))

    return np.array(life_preds, dtype=np.int64), np.array(scan_scores, dtype=np.float32)

def eval_pcstream_zeroshot(train_apps, test_app):
    pc_tbl = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    global_cnt = np.zeros(3, dtype=np.float64)
    for app in train_apps:
        d = DATA[app]
        for ev_i, y in zip(d["life_ev"], d["life_y"]):
            pc_tbl[int(d["pc_sigs"][ev_i])][int(y)] += 1.0
            global_cnt[int(y)] += 1.0
    default_cls = int(np.argmax(global_cnt))
    td = DATA[test_app]
    preds = []
    hits = 0
    for ev_i in td["life_ev"]:
        sig = int(td["pc_sigs"][ev_i])
        if sig in pc_tbl:
            preds.append(int(np.argmax(pc_tbl[sig])))
            hits += 1
        else:
            preds.append(default_cls)
    return np.array(preds, dtype=np.int64), hits

print("Pre-building causal coupled-entity training windows for all 5 real apps...")
t_b0 = time.time()
WIN_LIFE = {}
WIN_SCAN = {}
for app in APPS:
    d = DATA[app]
    WIN_LIFE[app] = build_causal_windows(d, d["life_ev"], d["life_y"], max_samples=450, balanced=True)
    WIN_SCAN[app] = build_causal_windows(d, d["scan_ev"], d["scan_y"], max_samples=1200, balanced=True)
    print(f"  [{app:8s}] train_life_wins={len(WIN_LIFE[app]['y']):4d} {dict(Counter(WIN_LIFE[app]['y'].tolist()))} | "
          f"train_scan_wins={len(WIN_SCAN[app]['y']):4d} {dict(Counter(WIN_SCAN[app]['y'].tolist()))}")
print(f"Built windows in {time.time() - t_b0:.1f}s")

def train_kssm_fold(train_apps, mode, epochs=45, lr=8e-3):
    def gather_task(win_dict):
        ws, wi, wt, wv, wy, ww = [], [], [], [], [], []
        for app in train_apps:
            wd = win_dict[app]
            y = wd["y"]
            cnt = Counter(y.tolist())
            w = np.array([1.0 / (len(cnt) * max(1, cnt[int(c)])) for c in y], dtype=np.float32)
            ws.append(wd["win_s"]); wi.append(wd["win_loc_ino"])
            wt.append(wd["win_same_tid"]); wv.append(wd["win_valid"])
            wy.append(y); ww.append(w)
        return (
            torch.tensor(np.concatenate(ws), device=DEVICE),
            torch.tensor(np.concatenate(wi), device=DEVICE),
            torch.tensor(np.concatenate(wt), device=DEVICE),
            torch.tensor(np.concatenate(wv), device=DEVICE),
            torch.tensor(np.concatenate(wy), device=DEVICE),
            torch.tensor(np.concatenate(ww), device=DEVICE),
        )

    ls, li, lt, lv, ly, lw = gather_task(WIN_LIFE)
    ss, si, st, sv, sy, sw = gather_task(WIN_SCAN)

    model = KSSMModel(mode=mode, dim=64).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    B = 256
    n_l, n_s = len(ly), len(sy)
    for ep in range(epochs):
        model.train()
        pl = torch.randperm(n_l, device=DEVICE)
        ps = torch.randperm(n_s, device=DEVICE)
        steps = max(n_l // B, n_s // B, 4)
        for step in range(steps):
            il = pl[(step * B) % n_l : ((step * B) % n_l) + B]
            is_ = ps[(step * B) % n_s : ((step * B) % n_s) + B]
            if len(il) < 16: il = pl[:B]
            if len(is_) < 16: is_ = ps[:B]

            logits_l, _ = model.forward_windows(ls[il], li[il], lt[il], lv[il])
            loss_l = (F.cross_entropy(logits_l, ly[il], reduction="none") * lw[il]).sum() / (lw[il].sum() + 1e-8)

            _, logits_s = model.forward_windows(ss[is_], si[is_], st[is_], sv[is_])
            loss_s = (F.binary_cross_entropy_with_logits(logits_s, sy[is_].float(), reduction="none") * sw[is_]).sum() / (sw[is_].sum() + 1e-8)

            loss = loss_l + loss_s
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()
    model.eval()
    return model

def train_heimdall_fold(train_apps, epochs=45, lr=5e-3):
    xl, yl, wl = [], [], []
    xs, ys, ws = [], [], []
    for app in train_apps:
        d = DATA[app]
        wdl = WIN_LIFE[app]; wds = WIN_SCAN[app]
        yl_a = wdl["y"]; cl = Counter(yl_a.tolist())
        wl_a = np.array([1.0 / (len(cl) * max(1, cl[int(c)])) for c in yl_a], dtype=np.float32)
        xl.append(d["scalars"][wdl["ev"]]); yl.append(yl_a); wl.append(wl_a)

        ys_a = wds["y"]; cs = Counter(ys_a.tolist())
        ws_a = np.array([1.0 / (len(cs) * max(1, cs[int(c)])) for c in ys_a], dtype=np.float32)
        xs.append(d["scalars"][wds["ev"]]); ys.append(ys_a); ws.append(ws_a)

    xl = torch.tensor(np.concatenate(xl), device=DEVICE)
    yl = torch.tensor(np.concatenate(yl), device=DEVICE)
    wl = torch.tensor(np.concatenate(wl), device=DEVICE)
    xs = torch.tensor(np.concatenate(xs), device=DEVICE)
    ys = torch.tensor(np.concatenate(ys), device=DEVICE)
    ws = torch.tensor(np.concatenate(ws), device=DEVICE)

    mlp = HeimdallMLP(16).to(DEVICE)
    opt = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    B = 256
    n_l, n_s = len(yl), len(ys)
    for ep in range(epochs):
        mlp.train()
        pl = torch.randperm(n_l, device=DEVICE)
        ps = torch.randperm(n_s, device=DEVICE)
        steps = max(n_l // B, n_s // B, 4)
        for step in range(steps):
            il = pl[(step * B) % n_l : ((step * B) % n_l) + B]
            is_ = ps[(step * B) % n_s : ((step * B) % n_s) + B]
            if len(il) < 16: il = pl[:B]
            if len(is_) < 16: is_ = ps[:B]
            ll, _ = mlp(xl[il])
            loss_l = (F.cross_entropy(ll, yl[il], reduction="none") * wl[il]).sum() / (wl[il].sum() + 1e-8)
            _, ls = mlp(xs[is_])
            loss_s = (F.binary_cross_entropy_with_logits(ls, ys[is_].float(), reduction="none") * ws[is_]).sum() / (ws[is_].sum() + 1e-8)
            loss = loss_l + loss_s
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    mlp.eval()
    return mlp

METHODS = [
    "PCStream (FAST'19)",
    "Heimdall (EuroSys'25)",
    "BASIC-Prefetch (SYSTOR'26)",
    "KSSM-NoTaskCoupling",
    "KSSM-Float64D",
    "KSSM-GuardSWAR-Shared(58ns)",
    "KSSM-GuardSWAR-Subspace(46ns)",
]

results = {m: {"life_acc": [], "life_bal": [], "life_f1": [], "scan_auc": []} for m in METHODS}
per_app_store = {}

print("\n" + "="*110)
print(" EXP 0-B: 5-Fold Leave-One-App-Out (LOAO) Zero-Shot Evaluation on Real Linux 5.15 VFS Traces")
print("="*110)

for held_out in APPS:
    train_apps = [a for a in APPS if a != held_out]
    td = DATA[held_out]
    y_life = td["life_y"]
    y_scan = td["scan_y"]

    print(f"\n>>> FOLD: Held-Out Unseen App = {held_out.upper()} (life_0s={len(y_life)}, scan={len(y_scan)})")
    per_app_store[held_out] = {}

    p_pc, pc_hits = eval_pcstream_zeroshot(train_apps, held_out)
    acc, bal, f1 = cls_metrics(y_life, p_pc)
    results["PCStream (FAST'19)"]["life_acc"].append(acc)
    results["PCStream (FAST'19)"]["life_bal"].append(bal)
    results["PCStream (FAST'19)"]["life_f1"].append(f1)
    results["PCStream (FAST'19)"]["scan_auc"].append(0.5000)
    print(f"  {'PCStream (FAST19)':30s} | Life0s Acc={acc*100:5.1f}% BalAcc={bal*100:5.1f}% F1={f1*100:5.1f}% (PC hits={pc_hits}) | Scan AUC=  N/A")

    mlp = train_heimdall_fold(train_apps)
    with torch.no_grad():
        xl_t = torch.tensor(td["scalars"][td["life_ev"]], device=DEVICE)
        xs_t = torch.tensor(td["scalars"][td["scan_ev"]], device=DEVICE)
        ll_h, _ = mlp(xl_t)
        _, ls_h = mlp(xs_t)
        p_h = ll_h.argmax(dim=-1).cpu().numpy()
        s_h = ls_h.cpu().numpy()
    acc, bal, f1 = cls_metrics(y_life, p_h)
    auc = roc_auc_exact(y_scan, s_h)
    results["Heimdall (EuroSys'25)"]["life_acc"].append(acc)
    results["Heimdall (EuroSys'25)"]["life_bal"].append(bal)
    results["Heimdall (EuroSys'25)"]["life_f1"].append(f1)
    results["Heimdall (EuroSys'25)"]["scan_auc"].append(auc)
    print(f"  {'Heimdall (EuroSys25)':30s} | Life0s Acc={acc*100:5.1f}% BalAcc={bal*100:5.1f}% F1={f1*100:5.1f}% | Scan AUC={auc:.4f}")

    p_b, s_b = eval_basic_markov(train_apps, held_out)
    acc, bal, f1 = cls_metrics(y_life, p_b)
    auc = roc_auc_exact(y_scan, s_b)
    results["BASIC-Prefetch (SYSTOR'26)"]["life_acc"].append(acc)
    results["BASIC-Prefetch (SYSTOR'26)"]["life_bal"].append(bal)
    results["BASIC-Prefetch (SYSTOR'26)"]["life_f1"].append(f1)
    results["BASIC-Prefetch (SYSTOR'26)"]["scan_auc"].append(auc)
    print(f"  {'BASIC-Prefetch (SYSTOR26)':30s} | Life0s Acc={acc*100:5.1f}% BalAcc={bal*100:5.1f}% F1={f1*100:5.1f}% | Scan AUC={auc:.4f}")

    for label, mode, bit_ex in [
        ("KSSM-NoTaskCoupling",           "no_coupling",   False),
        ("KSSM-Float64D",                 "float64",       False),
        ("KSSM-GuardSWAR-Shared(58ns)",   "swar_shared",   True),
        ("KSSM-GuardSWAR-Subspace(46ns)", "swar_subspace", True),
    ]:
        km = train_kssm_fold(train_apps, mode=mode)
        l_log, s_sc, l_feat, bounds = km.stream_eval_app(td, bit_exact_swar=bit_ex)
        p_k = np.argmax(l_log, axis=-1)
        acc, bal, f1 = cls_metrics(y_life, p_k)
        auc = roc_auc_exact(y_scan, s_sc)
        results[label]["life_acc"].append(acc)
        results[label]["life_bal"].append(bal)
        results[label]["life_f1"].append(f1)
        results[label]["scan_auc"].append(auc)
        bnd_str = f" [SWAR bounds={bounds}]" if bit_ex else ""
        print(f"  {label:30s} | Life0s Acc={acc*100:5.1f}% BalAcc={bal*100:5.1f}% F1={f1*100:5.1f}% | Scan AUC={auc:.4f}{bnd_str}")
        if label == "KSSM-GuardSWAR-Shared(58ns)":
            per_app_store[held_out]["kssm_swar_pred"] = p_k
            per_app_store[held_out]["kssm_swar_feat"] = l_feat
            per_app_store[held_out]["kssm_swar_w"] = km.head_life.weight.detach().cpu().numpy()
            per_app_store[held_out]["kssm_swar_b"] = km.head_life.bias.detach().cpu().numpy()

print("\n" + "="*110)
print(" SUMMARY TABLE: 5-App Leave-One-App-Out (LOAO) Macro-Average (Unseen App, 0th-Second Prediction)")
print("="*110)
print(f"{'Method':32s} | {'Life0s OverallAcc':>17s} | {'Life0s BalAcc':>13s} | {'Life0s MacroF1':>14s} | {'One-Shot AUC':>12s}")
print("-"*110)
for m in METHODS:
    r = results[m]
    auc_str = f"{np.mean(r['scan_auc']):.4f}" if m != "PCStream (FAST'19)" else "N/A"
    print(f"{m:32s} | {np.mean(r['life_acc'])*100:16.2f}% | {np.mean(r['life_bal'])*100:12.2f}% | {np.mean(r['life_f1'])*100:13.2f}% | {auc_str:>12s}")

print("\n" + "="*110)
print(" PART 2: Workload-Shift Blind-Window Showdown (0-10s / 10-30s / 30-65s After Unseen App Starts)")
print("="*110)

windows = [(0.0, 10.0), (10.0, 30.0), (30.0, 65.0), (0.0, 65.0)]
win_names = ["0-10s (Shift)", "10-30s", "30-65s", "Full 0-65s"]
trackers = {
    "eFDP (replan=420s, paper)":  defaultdict(list),
    "eFDP (replan=30s, agile)":   defaultdict(list),
    "PCStream (Online Death)":    defaultdict(list),
    "KSSM-GuardSWAR (Zero-Shot)": defaultdict(list),
    "KSSM-GuardSWAR (+SlowLoop)": defaultdict(list),
}

for app in APPS:
    d = DATA[app]
    y = d["life_y"]; tb = d["life_birth_s"]; td_s = d["life_death_s"]
    pcs = d["pc_sigs"][d["life_ev"]]
    n = len(y)

    for name, t_replan in [("eFDP (replan=420s, paper)", 420.0), ("eFDP (replan=30s, agile)", 30.0)]:
        preds = np.zeros(n, dtype=np.int64)
        obs_deaths = sorted([(td_s[i], int(y[i])) for i in range(n) if td_s[i] >= 0])
        last_cls = 0
        for i in range(n):
            cur_t = tb[i]
            if cur_t >= t_replan:
                bound = math.floor(cur_t / t_replan) * t_replan
                obs = [c for dt_t, c in obs_deaths if dt_t <= bound]
                if obs:
                    last_cls = Counter(obs).most_common(1)[0][0]
            preds[i] = last_cls
        for wi, (lo, hi) in enumerate(windows):
            m = (tb >= lo) & (tb < hi)
            if np.sum(m) > 0:
                _, bal, _ = cls_metrics(y[m], preds[m])
                trackers[name][wi].append(bal)

    pc_online_tbl = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    pending = sorted([(td_s[i], int(pcs[i]), int(y[i])) for i in range(n) if td_s[i] >= 0])
    dp = 0
    preds_pc = np.zeros(n, dtype=np.int64)
    for i in range(n):
        cur_t = tb[i]
        while dp < len(pending) and pending[dp][0] <= cur_t:
            _, sig_d, cls_d = pending[dp]
            pc_online_tbl[sig_d][cls_d] += 1.0
            dp += 1
        sig = int(pcs[i])
        preds_pc[i] = int(np.argmax(pc_online_tbl[sig])) if sig in pc_online_tbl else 0
    for wi, (lo, hi) in enumerate(windows):
        m = (tb >= lo) & (tb < hi)
        if np.sum(m) > 0:
            _, bal, _ = cls_metrics(y[m], preds_pc[m])
            trackers["PCStream (Online Death)"][wi].append(bal)

    preds_k0 = per_app_store[app]["kssm_swar_pred"]
    for wi, (lo, hi) in enumerate(windows):
        m = (tb >= lo) & (tb < hi)
        if np.sum(m) > 0:
            _, bal, _ = cls_metrics(y[m], preds_k0[m])
            trackers["KSSM-GuardSWAR (Zero-Shot)"][wi].append(bal)

    W = per_app_store[app]["kssm_swar_w"].copy()
    b = per_app_store[app]["kssm_swar_b"].copy()
    feats = per_app_store[app]["kssm_swar_feat"]
    feedback = sorted([(td_s[i], i, int(y[i])) if td_s[i] >= 0 else (tb[i] + 22.0, i, 2) for i in range(n)])
    fp = 0
    preds_kon = np.zeros(n, dtype=np.int64)
    for i in range(n):
        cur_t = tb[i]
        while fp < len(feedback) and feedback[fp][0] <= cur_t:
            _, idx_f, y_f = feedback[fp]
            xf = feats[idx_f]
            lg = W @ xf + b; lg -= lg.max()
            pr = np.exp(lg); pr /= pr.sum()
            grad = pr; grad[y_f] -= 1.0
            cw = 2.0 if y_f > 0 else 0.5
            W -= (0.05 * cw) * np.outer(grad, xf)
            b -= (0.05 * cw) * grad
            fp += 1
        preds_kon[i] = int(np.argmax(W @ feats[i] + b))
    for wi, (lo, hi) in enumerate(windows):
        m = (tb >= lo) & (tb < hi)
        if np.sum(m) > 0:
            _, bal, _ = cls_metrics(y[m], preds_kon[m])
            trackers["KSSM-GuardSWAR (+SlowLoop)"][wi].append(bal)

print(f"{'Method (Balanced Acc % across 5 unseen apps)':34s} | {win_names[0]:>14s} | {win_names[1]:>12s} | {win_names[2]:>12s} | {win_names[3]:>12s}")
print("-"*110)
for m_name, d_w in trackers.items():
    vals = [f"{np.mean(d_w[wi])*100:11.2f}%" if d_w[wi] else "     N/A    " for wi in range(4)]
    print(f"{m_name:34s} | {vals[0]:>14s} | {vals[1]:>12s} | {vals[2]:>12s} | {vals[3]:>12s}")
