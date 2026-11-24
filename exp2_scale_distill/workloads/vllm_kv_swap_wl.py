import os, sys, time, random, threading
import torch

base = sys.argv[1]; dur = int(sys.argv[2]) if len(sys.argv) > 2 else 65
os.makedirs(base, exist_ok=True)
d_sys  = os.path.join(base, "hot_sys_prefix")
d_lora = os.path.join(base, "cold_lora_pool")
d_req  = os.path.join(base, "req_kv_swap")
d_sess = os.path.join(base, "sess_prefix")
d_dur  = os.path.join(base, "durable_lora")
for d in (d_sys, d_lora, d_req, d_sess, d_dur): os.makedirs(d, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hot shared system prompt KV prefix (512 KB, repeatedly read -> Hot Reusable)
sys_p = os.path.join(d_sys, "system_prompt_kv.pt")
torch.save(torch.randn(256, 512), sys_p)

# 14 distinct cold LoRA weight pools (16 MB each, scanned once -> One-Shot Scan)
lora_pool = []
for i in range(14):
    p = os.path.join(d_lora, f"lora_bank_{i:02d}.bin")
    with open(p, "wb") as f: f.write(os.urandom(16 * 1024 * 1024))
    lora_pool.append(p)

stop_evt = threading.Event()

def hot_sys_reader():
    rng = random.Random(88)
    while not stop_evt.is_set():
        with open(sys_p, "rb") as f:
            for _ in range(8):
                f.seek(rng.randint(0, 60) * 4096)
                _ = f.read(8192)
        time.sleep(0.03)

def cold_lora_loader():
    for p in lora_pool:
        if stop_evt.is_set(): break
        with open(p, "rb") as f:
            while not stop_evt.is_set():
                if not f.read(524288): break
        time.sleep(2.8)

def kv_swap_worker():
    req_q, sess_q = [], []
    seq = 0
    # Real GPU 3 KV tensor (512x1024 fp16 = 1 MB) moved D2H -> NVMe!
    kv_gpu = torch.randn(512, 1024, dtype=torch.float16, device=device)
    while not stop_evt.is_set():
        seq += 1
        now = time.time()
        # Simulate GPU 3 attention update then D2H offload
        kv_gpu.mul_(0.99).add_(0.01)
        cpu_buf = kv_gpu.to("cpu", non_blocking=False).numpy().tobytes()

        # 1. Short-lived preempted request KV swap (2.5s TTL)
        rp = os.path.join(d_req, f"preempt_kv_{seq}.kv")
        with open(rp, "wb") as f: f.write(cpu_buf)
        req_q.append((now, rp))

        # 2. Medium-lived multi-turn session prefix cache (16s TTL, read back once!)
        if seq % 2 == 0:
            sp = os.path.join(d_sess, f"sess_pfx_{seq}.kv")
            with open(sp, "wb") as f: f.write(cpu_buf)
            with open(sp, "rb") as f: _ = f.read(262144)
            sess_q.append((now, sp))

        # 3. Long-lived durable compiled adapter (reads system prefix + fsync + rename)
        if seq % 4 == 0:
            with open(sys_p, "rb") as f: _ = f.read()
            dp_t = os.path.join(d_dur, f"lora_save_{seq}.tmp")
            dp_f = os.path.join(d_dur, f"lora_save_{seq}.pt")
            with open(dp_t, "wb") as f:
                f.write(cpu_buf); f.flush(); os.fsync(f.fileno())
            os.rename(dp_t, dp_f)

        while req_q and (now - req_q[0][0]) >= 2.5:
            _, p = req_q.pop(0)
            if os.path.exists(p): os.unlink(p)
        while sess_q and (now - sess_q[0][0]) >= 16.0:
            _, p = sess_q.pop(0)
            if os.path.exists(p): os.unlink(p)
        time.sleep(0.5)

ts = [threading.Thread(target=hot_sys_reader), threading.Thread(target=cold_lora_loader), threading.Thread(target=kv_swap_worker)]
for t in ts: t.start()
time.sleep(dur)
stop_evt.set()
for t in ts: t.join()
print(f"vllm_kv_swap completed ({dur}s on {device})")
