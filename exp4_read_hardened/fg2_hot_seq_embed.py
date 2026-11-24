import os, sys, time
import numpy as np

fpath = sys.argv[1]
dur_s = int(sys.argv[2]) if len(sys.argv) > 2 else 35
fsize = os.path.getsize(fpath)

# Warm into 8 GiB cgroup page cache before starting measurement!
fd = os.open(fpath, os.O_RDONLY)
while os.read(fd, 524288): pass
os.close(fd)

pass_ms = []
tot_bytes = 0
t_start = time.time()
while time.time() - t_start < dur_s:
    tp0 = time.perf_counter()
    fd = os.open(fpath, os.O_RDONLY)
    b = 0
    while True:
        ch = os.read(fd, 524288)
        if not ch: break
        b += len(ch)
    os.close(fd)
    dt_ms = (time.perf_counter() - tp0) * 1000.0
    pass_ms.append(dt_ms)
    tot_bytes += b
    # Brief inter-batch compute pause (15ms) simulating real inference/ranking batch interval
    time.sleep(0.015)

el = time.time() - t_start
arr = np.array(pass_ms)
print(f"FG2_HOTSEQ passes={len(arr)} bw_mbps={tot_bytes/(1024**2)/el:.1f} "
      f"mean_ms={arr.mean():.1f} p50_ms={np.percentile(arr,50):.1f} p95_ms={np.percentile(arr,95):.1f} p99_ms={np.percentile(arr,99):.1f}")
