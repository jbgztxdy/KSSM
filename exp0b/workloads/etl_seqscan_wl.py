import os, sys, time, random, threading

base = sys.argv[1]
dur = int(sys.argv[2]) if len(sys.argv) > 2 else 65
os.makedirs(base, exist_ok=True)
dir_raw   = os.path.join(base, "raw_lake")
dir_idx   = os.path.join(base, "hot_idx")
dir_s1    = os.path.join(base, "stage1_tmp")
dir_s2    = os.path.join(base, "stage2_mid")
dir_s3    = os.path.join(base, "stage3_out")
for d in (dir_raw, dir_idx, dir_s1, dir_s2, dir_s3):
    os.makedirs(d, exist_ok=True)

# Pre-create large raw files (32 MB each) & hot small lookup indexes (256 KB each)
raw_files = []
for i in range(8):
    fp = os.path.join(dir_raw, f"raw_part_{i:02d}.dat")
    with open(fp, "wb") as f:
        f.write(os.urandom(32 * 1024 * 1024))
    raw_files.append(fp)

idx_files = []
for i in range(4):
    fp = os.path.join(dir_idx, f"lookup_idx_{i}.idx")
    with open(fp, "wb") as f:
        f.write(os.urandom(256 * 1024))
    idx_files.append(fp)

stop_evt = threading.Event()

def seq_scanner():
    """Streams distinct raw lake files sequentially (One-Shot Scan)"""
    i = 0
    while not stop_evt.is_set():
        fp = raw_files[i % len(raw_files)]
        i += 1
        with open(fp, "rb") as f:
            while not stop_evt.is_set():
                blk = f.read(524288)
                if not blk: break
        time.sleep(0.3)

def hot_index_reader():
    """High-frequency random reads on small hot index files (Hot Reusable)"""
    rng = random.Random(77)
    while not stop_evt.is_set():
        fp = idx_files[rng.randint(0, len(idx_files) - 1)]
        with open(fp, "rb") as f:
            for _ in range(12):
                f.seek(rng.randint(0, 50) * 4096)
                _ = f.read(8192)
        time.sleep(0.04)

def etl_pipeline():
    """3-stage external merge ETL with uniform 2MB file sizes across lifetime tiers"""
    s1_q, s2_q = [], []
    seq = 0
    buf = os.urandom(2 * 1024 * 1024)
    while not stop_evt.is_set():
        seq += 1
        now = time.time()
        # Stage 1: temp sort spill (short-lived ~2-4s)
        p1 = os.path.join(dir_s1, f"sort_run_{seq}.tmp")
        with open(p1, "wb") as f:
            f.write(buf)
        s1_q.append((now, p1))

        # Stage 2: intermediate merged partition (medium-lived ~16-20s)
        if seq % 2 == 0 and s1_q:
            # Read Stage 1 spill before writing Stage 2
            with open(s1_q[-1][1], "rb") as f: _ = f.read(262144)
            p2 = os.path.join(dir_s2, f"merged_part_{seq}.mid")
            with open(p2, "wb") as f:
                f.write(buf)
            s2_q.append((now, p2))

        # Stage 3: final durable archive (long-lived, reads stage2 + hot index + fsync + rename)
        if seq % 4 == 0:
            if s2_q:
                with open(s2_q[-1][1], "rb") as f: _ = f.read(262144)
            with open(idx_files[0], "rb") as f: _ = f.read(65536)
            p3_tmp = os.path.join(dir_s3, f"final_{seq}.parquet.tmp")
            p3_fin = os.path.join(dir_s3, f"final_{seq}.parquet")
            with open(p3_tmp, "wb") as f:
                f.write(buf)
                f.flush()
                os.fsync(f.fileno())
            os.rename(p3_tmp, p3_fin)

        while s1_q and (now - s1_q[0][0]) >= 3.5:
            _, p = s1_q.pop(0)
            if os.path.exists(p): os.unlink(p)
        while s2_q and (now - s2_q[0][0]) >= 17.0:
            _, p = s2_q.pop(0)
            if os.path.exists(p): os.unlink(p)
        time.sleep(0.8)

threads = [
    threading.Thread(target=seq_scanner),
    threading.Thread(target=hot_index_reader),
    threading.Thread(target=etl_pipeline),
]
for t in threads: t.start()
time.sleep(dur)
stop_evt.set()
for t in threads: t.join()
print(f"ETL seqscan workload completed ({dur}s)")
