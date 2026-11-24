import os, sys, time, random, mmap, threading
from safetensors import safe_open

base = sys.argv[1]
dur = int(sys.argv[2]) if len(sys.argv) > 2 else 65
os.makedirs(base, exist_ok=True)
dir_req   = os.path.join(base, "req_offload")
dir_sess  = os.path.join(base, "session_kv")
dir_dur   = os.path.join(base, "durable_out")
for d in (dir_req, dir_sess, dir_dur):
    os.makedirs(d, exist_ok=True)

model_dirs = [
    "/mnt/nvme12/paul/models/DeepSeek-R1-Distill-Qwen-14B",
    "/mnt/nvme12/paul/models/Meta-Llama-3-8B-Instruct",
]
st_files = []
hot_meta = []
for md in model_dirs:
    if os.path.isdir(md):
        for fn in sorted(os.listdir(md)):
            fp = os.path.join(md, fn)
            if fn.endswith(".safetensors"):
                st_files.append(fp)
            elif fn.endswith(".json"):
                hot_meta.append(fp)

stop_evt = threading.Event()

def weight_loader():
    """Real sequential & mmap reading of .safetensors weight shards (One-Shot Scan)"""
    idx = 0
    while not stop_evt.is_set() and st_files:
        fp = st_files[idx % len(st_files)]
        idx += 1
        # Alternate between safetensors safe_open (mmap) and direct sequential read
        if idx % 2 == 0:
            with safe_open(fp, framework="pt") as f:
                keys = list(f.keys())[:12]
                for k in keys:
                    if stop_evt.is_set(): break
                    t = f.get_tensor(k)
                    _ = float(t.view(-1)[0].item())
                    time.sleep(0.05)
        else:
            # Read 128 MB chunk sequentially per pass
            with open(fp, "rb") as f:
                f.seek(((idx * 134217728) % max(1, os.path.getsize(fp) - 134217728)))
                for _ in range(256): # 256 * 512KB = 128 MB
                    if stop_evt.is_set(): break
                    blk = f.read(524288)
                    if not blk: break
        time.sleep(0.2)

def hot_meta_reader():
    """Repeatedly reads tokenizer.json / config.json (Hot Reusable)"""
    while not stop_evt.is_set() and hot_meta:
        for fp in hot_meta:
            with open(fp, "rb") as f:
                _ = f.read(65536)
        time.sleep(0.08)

def kv_lifecycle_worker():
    """Creates realistic 3-tier lifetime files during LLM serving:
       - request-level KV spill (short-lived 1-5s)
       - session-level prefix cache (medium-lived 15-25s)
       - durable LoRA/output archive (long-lived, preceded by model meta read + fsync)"""
    req_q = []
    sess_q = []
    seq = 0
    payload = os.urandom(2 * 1024 * 1024) # 2 MB uniform payload size across tiers!
    while not stop_evt.is_set():
        seq += 1
        now = time.time()
        # 1. Short-lived request KV spill
        rp = os.path.join(dir_req, f"req_kv_{seq}.bin")
        with open(rp, "wb") as f:
            f.write(payload)
        req_q.append((now, rp))

        # 2. Medium-lived session prefix cache (every 2 steps)
        if seq % 2 == 0:
            sp = os.path.join(dir_sess, f"sess_kv_{seq}.bin")
            with open(sp, "wb") as f:
                f.write(payload)
            # Read it back once during session reuse
            with open(sp, "rb") as f:
                _ = f.read(262144)
            sess_q.append((now, sp))

        # 3. Long-lived durable output/adapter (every 4 steps, after reading hot config + fsync)
        if seq % 4 == 0:
            if hot_meta:
                with open(hot_meta[0], "rb") as f: _ = f.read()
            dp = os.path.join(dir_dur, f"adapter_out_{seq}.bin")
            with open(dp, "wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())

        # Reap expired files
        while req_q and (now - req_q[0][0]) >= 3.0:
            _, p = req_q.pop(0)
            if os.path.exists(p): os.unlink(p)
        while sess_q and (now - sess_q[0][0]) >= 17.0:
            _, p = sess_q.pop(0)
            if os.path.exists(p): os.unlink(p)

        time.sleep(0.9)

threads = [
    threading.Thread(target=weight_loader),
    threading.Thread(target=hot_meta_reader),
    threading.Thread(target=kv_lifecycle_worker),
]
for t in threads: t.start()
time.sleep(dur)
stop_evt.set()
for t in threads: t.join()
print(f"LLM safetensors workload completed ({dur}s, shards={len(st_files)})")
