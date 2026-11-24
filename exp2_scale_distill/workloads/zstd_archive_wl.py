import os, sys, time, random, subprocess, threading

base = sys.argv[1]; dur = int(sys.argv[2]) if len(sys.argv) > 2 else 65
os.makedirs(base, exist_ok=True)
d_man = os.path.join(base, "manifest")
d_raw = os.path.join(base, "raw_input")
d_tmp = os.path.join(base, "stage_extract")
d_mid = os.path.join(base, "incr_zst")
d_arc = os.path.join(base, "full_zst")
for d in (d_man, d_raw, d_tmp, d_mid, d_arc): os.makedirs(d, exist_ok=True)

man_p = os.path.join(d_man, "catalog.idx")
with open(man_p, "wb") as f: f.write(os.urandom(256 * 1024))

raw_inputs = []
for i in range(14):
    p = os.path.join(d_raw, f"raw_chunk_{i:02d}.bin")
    # Semi-compressible structured data for realistic zstd work
    with open(p, "wb") as f:
        blk = (bytes([i & 255]) * 1024 + os.urandom(1024)) * 512 # 1 MB
        for _ in range(16): f.write(blk) # 16 MB
    raw_inputs.append(p)

stop_evt = threading.Event()

def hot_catalog():
    rng = random.Random(31)
    while not stop_evt.is_set():
        with open(man_p, "rb") as f:
            for _ in range(8):
                f.seek(rng.randint(0, 50) * 4096)
                _ = f.read(8192)
        time.sleep(0.03)

def zstd_stream_reader():
    for p in raw_inputs:
        if stop_evt.is_set(): break
        # Real zstd streaming read -> /dev/null (pure One-Shot Scan!)
        subprocess.run(["zstd", "-1", "-q", "-c", p], stdout=subprocess.DEVNULL, check=True)
        time.sleep(2.5)

def archive_pipeline():
    tmp_q, mid_q = [], []
    seq = 0
    src_blk = (b"RECORD_DATA_0123456789" * 64 + os.urandom(512)) * 1024 # ~1.9 MB
    while not stop_evt.is_set():
        seq += 1
        now = time.time()
        # 1. Short-lived temp raw staging file (< 3s)
        tp = os.path.join(d_tmp, f"stage_{seq}.dat")
        with open(tp, "wb") as f: f.write(src_blk)
        tmp_q.append((now, tp))

        # 2. Medium-lived incremental zstd archive (16s TTL)
        if seq % 2 == 0:
            mp = os.path.join(d_mid, f"incr_{seq}.zst")
            subprocess.run(["zstd", "-1", "-q", "-f", tp, "-o", mp], check=True)
            mid_q.append((now, mp))

        # 3. Long-lived full archive (reads catalog + compresses + fsync)
        if seq % 4 == 0:
            with open(man_p, "rb") as f: _ = f.read()
            ap = os.path.join(d_arc, f"full_{seq}.zst")
            subprocess.run(["zstd", "-3", "-q", "-f", tp, "-o", ap], check=True)
            with open(ap, "rb+") as f: os.fsync(f.fileno())

        while tmp_q and (now - tmp_q[0][0]) >= 2.5:
            _, p = tmp_q.pop(0)
            if os.path.exists(p): os.unlink(p)
        while mid_q and (now - mid_q[0][0]) >= 16.0:
            _, p = mid_q.pop(0)
            if os.path.exists(p): os.unlink(p)
        time.sleep(0.55)

ts = [threading.Thread(target=hot_catalog), threading.Thread(target=zstd_stream_reader), threading.Thread(target=archive_pipeline)]
for t in ts: t.start()
time.sleep(dur)
stop_evt.set()
for t in ts: t.join()
print(f"zstd_archive completed ({dur}s)")
