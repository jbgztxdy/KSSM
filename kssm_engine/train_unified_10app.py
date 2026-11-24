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

WIN_LEN = 32
def build_seqs(d, target_evs, labels, max_n=360):
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
        tid_evs[int(tid_arr[ei])].append(ei); inst_evs[int(inst_arr[ei])].append(ei)
    tid_evs = {k: np.array(v, dtype=np.int32) for k, v in tid_evs.items()}
    inst_evs = {k: np.array(v, dtype=np.int32) for k, v in inst_evs.items()}

    s = np.zeros((M, WIN_LEN, 5), dtype=np.int64)
    mi = np.zeros((M, WIN_LEN), dtype=np.float32); mt = np.zeros((M, WIN_LEN), dtype=np.float32)
    v = np.zeros((M, WIN_LEN), dtype=np.float32)
    for m in range(M):
        ei = int(evs[m]); tt = int(tid_arr[ei]); ti = int(inst_arr[ei])
        te = tid_evs[tt]; tp = int(np.searchsorted(te, ei, side="right"))
        ie = inst_evs[ti]; ip = int(np.searchsorted(ie, ei, side="right"))
        merged = np.unique(np.concatenate([te[max(0, tp - 24):tp], ie[max(0, ip - 16):ip]]))
        if len(merged) > WIN_LEN: merged = merged[-WIN_LEN:]
        off = WIN_LEN - len(merged)
        s[m, off:] = sg[merged]; v[m, off:] = 1.0
        mi[m, off:] = (inst_arr[merged] == ti).astype(np.float32)
        mt[m, off:] = (tid_arr[merged] == tt).astype(np.float32)
    return {"y": ys, "s": s, "mi": mi, "mt": mt, "v": v}

WL = {a: build_seqs(DATA[a], DATA[a]["life_ev"], DATA[a]["life_y"], max_n=360) for a in APPS}
WS = {a: build_seqs(DATA[a], DATA[a]["scan_ev"], DATA[a]["scan_y"], max_n=800) for a in APPS}

class KSSMSel4S1Only(nn.Module):
    """Pruned OOV-safe 4-Band Selective Guard-SWAR 64D:
    Uses ONLY primitive-indexed g_s1[32, 4, 4] (128 bytes total!) instead of 1024-slot g_pair,
    eliminating (s1, s4) OOV mis-routing to HOLD while keeping exact 37-49 ns eBPF speed!"""
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(192, 64)
        nn.init.normal_(self.emb.weight, 0, 0.15)
        # Initialize slightly biased away from HOLD so active primitives don't collapse at epoch 0
        self.g_s1 = nn.Parameter(torch.tensor([[[ -0.3, 0.2, 0.2, 0.0 ]]*4]*32, dtype=torch.float32))
        self.register_buffer("a_i_vals", torch.tensor([1.0, 0.875, 0.5, 0.0]))
        self.register_buffer("g_i_vals", torch.tensor([0.0, 0.0625, 0.25, 0.5]))
        self.register_buffer("a_t_vals", torch.tensor([1.0, 0.875, 0.5, 0.75]))
        self.register_buffer("g_t_vals", torch.tensor([0.0, 0.0625, 0.25, 0.0]))
        self.register_buffer("in_scale", torch.tensor([0.0, 1.0, 1.0, 1.0]))
        self.drop = nn.Dropout(0.15)
        self.hl = nn.Linear(128, 3)
        self.hs = nn.Linear(128, 1)

    def tv_loss(self):
        w = torch.tanh(self.emb.weight) * 0.75
        l5 = F.mse_loss(w[160:175], w[161:176]) + F.mse_loss(w[176:191], w[177:192])
        l2 = sum(F.mse_loss(w[32+q*16:32+q*16+15], w[32+q*16+1:32+q*16+16]) for q in range(4))
        return 0.2 * (l5 + 0.5 * l2)

    def forward_seq(self, s, mi, mt, v, tau=1.0, hard=False):
        B, L, _ = s.shape
        offs = torch.tensor([0, 32, 96, 128, 160], device=s.device)
        w = torch.tanh(self.emb(s + offs)) * 0.75
        b_all = torch.cat([
            w[:, :, 0, 0:16]  + w[:, :, 1, 0:16],
            w[:, :, 0, 16:32] + w[:, :, 2, 16:32],
            w[:, :, 0, 32:48] + w[:, :, 3, 32:48],
            w[:, :, 0, 48:64] + w[:, :, 4, 48:64],
        ], dim=-1)
        probs = F.softmax(self.g_s1[s[:, :, 0]] / tau, dim=-1) # [B, L, 4, 4]
        if hard:
            idx = probs.argmax(dim=-1, keepdim=True)
            oh = torch.zeros_like(probs).scatter_(-1, idx, 1.0)
            probs = (oh - probs).detach() + probs

        ai  = (probs * self.a_i_vals).sum(-1).repeat_interleave(16, dim=-1)
        gi  = (probs * self.g_i_vals).sum(-1).repeat_interleave(16, dim=-1)
        at  = (probs * self.a_t_vals).sum(-1).repeat_interleave(16, dim=-1)
        gt  = (probs * self.g_t_vals).sum(-1).repeat_interleave(16, dim=-1)
        is_ = (probs * self.in_scale).sum(-1).repeat_interleave(16, dim=-1)

        h_t = torch.zeros(B, 64, device=s.device); h_i = torch.zeros(B, 64, device=s.device)
        for t in range(L):
            vt = v[:, t:t+1] * mt[:, t:t+1]; vi = v[:, t:t+1] * mi[:, t:t+1]
            bt = b_all[:, t, :] * is_[:, t, :]
            hi_n = ai[:, t, :] * h_i + gi[:, t, :] * (h_t * mt[:, t:t+1]) + bt
            ht_n = at[:, t, :] * h_t - gt[:, t, :] * (h_i * mi[:, t:t+1]) + bt
            h_i = h_i * (1.0 - vi) + hi_n * vi
            h_t = h_t * (1.0 - vt) + ht_n * vt
        feat = self.drop(torch.cat([h_i, h_t], dim=-1))
        return self.hl(feat), self.hs(feat).squeeze(-1)

    @torch.no_grad()
    def stream_eval(self, d):
        w_f = (torch.tanh(self.emb.weight) * 0.75).detach().cpu().numpy()
        emb_u16 = (np.round(w_f * 256.0).astype(np.int32) + 2048).astype(np.uint16)
        reg_tbl = self.g_s1.argmax(dim=-1).cpu().numpy().astype(np.uint8) # [32, 4]
        wl_q = np.round(self.hl.weight.detach().cpu().numpy() * 256.0).astype(np.int32)
        bl_q = np.round(self.hl.bias.detach().cpu().numpy() * 65536.0).astype(np.int32)
        ws_q = np.round(self.hs.weight.detach().cpu().numpy()[0] * 256.0).astype(np.int32)
        bs_q = int(round(float(self.hs.bias.detach().cpu().item()) * 65536.0))

        N = d["N"]; tids = d["tid"]; insts = d["inst"]; sg = d["s_gram"]
        H_i = np.full((d["num_insts"], 64), 16384, dtype=np.int32)
        H_t = np.full((d["num_tids"], 64), 16384, dtype=np.int32)
        lset = {int(e): i for i, e in enumerate(d["life_ev"])}
        sset = {int(e): i for i, e in enumerate(d["scan_ev"])}
        ll = np.zeros((len(d["life_ev"]), 3), dtype=np.float32)
        sc = np.zeros(len(d["scan_ev"]), dtype=np.float32)

        s0 = sg[:, 0].astype(np.int32); s1 = 32 + sg[:, 1].astype(np.int32)
        s2 = 96 + sg[:, 2].astype(np.int32); s3 = 128 + sg[:, 3].astype(np.int32); s4 = 160 + sg[:, 4].astype(np.int32)
        for e in range(N):
            tid = int(tids[e]); iid = int(insts[e])
            i0 = int(s0[e]); regs = reg_tbl[i0]
            oi = H_i[iid]; ot = H_t[tid]
            sb = (int(s1[e]), int(s2[e]), int(s3[e]), int(s4[e]))
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
            if e in lset:
                fq = np.concatenate([oi - 16384, ot - 16384])
                ll[lset[e]] = (wl_q @ fq + bl_q) / 65536.0
            if e in sset:
                fq = np.concatenate([oi - 16384, ot - 16384])
                sc[sset[e]] = float(np.dot(ws_q, fq) + bs_q) / 65536.0
        return ll, sc

def train_s1only(train_apps, epochs=32):
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
    m = KSSMSel4S1Only().to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=7e-3, weight_decay=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    B = 512; nl, ns = len(ly), len(sy)
    for ep in range(epochs):
        m.train(); tau = max(0.06, 2.0 * (0.86 ** ep)); hard = (ep >= epochs - 8)
        pl = torch.randperm(nl, device=DEVICE); ps = torch.randperm(ns, device=DEVICE)
        for step in range(max(nl // B, ns // B, 6)):
            il = pl[(step*B)%nl : ((step*B)%nl)+B]; is_ = ps[(step*B)%ns : ((step*B)%ns)+B]
            if len(il) < 32: il = pl[:min(B, nl)]
            if len(is_) < 32: is_ = ps[:min(B, ns)]
            ll, _ = m.forward_seq(ls[il], lmi[il], lmt[il], lv[il], tau=tau, hard=hard)
            l1 = (F.cross_entropy(ll, ly[il], reduction="none") * lw[il]).sum() / (lw[il].sum() + 1e-8)
            _, lso = m.forward_seq(ss[is_], smi[is_], smt[is_], sv[is_], tau=tau, hard=hard)
            l2 = (F.binary_cross_entropy_with_logits(lso, sy[is_].float(), reduction="none") * sw[is_]).sum() / (sw[is_].sum() + 1e-8)
            opt.zero_grad(); (l1 + l2 + m.tv_loss()).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0); opt.step()
        sched.step()
    m.eval()
    return m

print("=== VERIFYING PRUNED g_s1[32,4] FIX ON PREVIOUSLY-OOV FOLDS (ROCKSDB & LOG_SEGMENT) ===", flush=True)
for held in ["rocksdb", "log_segment"]:
    tr = [a for a in APPS if a != held]
    km = train_s1only(tr, epochs=32)
    ll, sc = km.stream_eval(DATA[held])
    a, b, f1 = cls_metrics(DATA[held]["life_y"], np.argmax(ll, -1))
    auc = roc_auc_exact(DATA[held]["scan_y"], sc)
    print(f"  Pruned g_s1[32,4] on {held.upper():11s}: Acc={a*100:5.1f}% BalAcc={b*100:5.1f}% F1={f1*100:5.1f}% | AUC={auc:.4f}", flush=True)

print("\n=== TRAINING FULL 10-APP UNIFIED PRODUCTION MODEL & EXPORTING kssm_10app_unified.bin ===", flush=True)
prod = train_s1only(APPS, epochs=36)
w_f = (torch.tanh(prod.emb.weight) * 0.75).detach().cpu().numpy()
emb_u16 = (np.round(w_f * 256.0).astype(np.int32) + 2048).astype(np.uint16)
emb_u64 = np.zeros((192, 16), dtype=np.uint64)
for s in range(192):
    for g in range(16):
        v = np.uint64(0)
        for l in range(4): v |= (np.uint64(emb_u16[s, g*4+l]) << np.uint64(l*16))
        emb_u64[s, g] = v

reg_s1 = prod.g_s1.argmax(dim=-1).cpu().numpy().astype(np.uint8) # [32, 4] = 128 bytes!
print("Learned 4-Band Selective Regimes (0=HOLD,1=SLOW,2=FAST,3=RESET) across s1=0..18:")
for s1 in range(19):
    print(f"  s1={s1:2d}: {reg_s1[s1].tolist()}")

ws_q = np.clip(np.round(prod.hs.weight.detach().cpu().numpy()[0] * 256.0), -32768, 32767).astype(np.int16)
bs_q = np.int32(round(float(prod.hs.bias.detach().cpu().item()) * 65536.0))
wl_q = np.clip(np.round(prod.hl.weight.detach().cpu().numpy() * 256.0), -32768, 32767).astype(np.int16)
bl_q = np.round(prod.hl.bias.detach().cpu().numpy() * 65536.0).astype(np.int32)

# Export two clean binary files matching include/kssm_abi.h:
#   models/kssm_base.bin (24,704 B = 24,576 B emb + 128 B reg_s1)
#   models/kssm_head.bin ( 1,040 B = w_scan_i[64] + w_scan_t[64] + w_life_i[3][64] + w_life_t[3][64] + b_scan + b_life[3])
with open("models/kssm_base.bin", "wb") as f:
    f.write(emb_u64.tobytes())  # 24576 B
    f.write(reg_s1.tobytes())   #   128 B
with open("models/kssm_head.bin", "wb") as f:
    f.write(ws_q[:64].tobytes())       # 128 B
    f.write(ws_q[64:128].tobytes())    # 128 B
    f.write(wl_q[:, :64].tobytes())    # 384 B
    f.write(wl_q[:, 64:128].tobytes()) # 384 B
    f.write(struct.pack("<i", int(bs_q))) # 4 B
    f.write(bl_q.tobytes())               # 12 B
print("Exported models/kssm_base.bin (24,704 B) & models/kssm_head.bin (1,040 B)!")
