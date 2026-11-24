import os, sys, time, random, subprocess, threading

base = sys.argv[1]; dur = int(sys.argv[2]) if len(sys.argv) > 2 else 65
os.makedirs(base, exist_ok=True)
d_hdr = os.path.join(base, "include")
d_big = os.path.join(base, "big_sources")
d_tmp = os.path.join(base, "obj_tmp")
d_mid = os.path.join(base, "cache_lib")
d_bin = os.path.join(base, "dist_bin")
for d in (d_hdr, d_big, d_tmp, d_mid, d_bin): os.makedirs(d, exist_ok=True)

# Hot shared header repeatedly read by every compile unit (Hot Reusable)
hdr_p = os.path.join(d_hdr, "common.hpp")
with open(hdr_p, "w") as f:
    f.write("#pragma once\n#include <cstdint>\n")
    for i in range(400):
        f.write(f"static inline uint64_t helper_{i}(uint64_t x) {{ return x * {i+3}ULL ^ 0x9e3779b9ULL; }}\n")

# Large pre-generated source blobs scanned sequentially once each (One-Shot Scan)
big_blobs = []
for i in range(16):
    p = os.path.join(d_big, f"gen_table_{i:02d}.dat")
    with open(p, "wb") as f: f.write(os.urandom(16 * 1024 * 1024))
    big_blobs.append(p)

stop_evt = threading.Event()

def hot_header_reader():
    rng = random.Random(55)
    while not stop_evt.is_set():
        with open(hdr_p, "rb") as f:
            for _ in range(8):
                f.seek(rng.randint(0, 6) * 4096)
                _ = f.read(4096)
        time.sleep(0.03)

def big_source_scanner():
    for p in big_blobs:
        if stop_evt.is_set(): break
        with open(p, "rb") as f:
            while not stop_evt.is_set():
                if not f.read(524288): break
        time.sleep(2.5)

def build_loop():
    mid_q = []
    seq = 0
    while not stop_evt.is_set():
        seq += 1
        now = time.time()
        # 1. Real g++ compile -> short-lived .cpp & .o (unlinked right after build, < 3s!)
        src_p = os.path.join(d_tmp, f"unit_{seq}.cpp")
        obj_p = os.path.join(d_tmp, f"unit_{seq}.o")
        with open(src_p, "w") as f:
            f.write(f'#include "common.hpp"\nint main() {{ volatile uint64_t s = {seq}; for(int i=0;i<10;i++) s += helper_0(s); return (int)(s & 1); }}\n')
        subprocess.run(["g++", "-O1", f"-I{d_hdr}", "-c", src_p, "-o", obj_p], check=True)
        # Read object & link into medium-lived build cache artifact or durable release binary
        is_release = (seq % 4 == 0)
        if is_release:
            # Read header metadata + link durable binary + fsync (Long-lived!)
            with open(hdr_p, "rb") as f: _ = f.read()
            out_p = os.path.join(d_bin, f"release_{seq}")
            subprocess.run(["g++", obj_p, "-o", out_p], check=True)
            with open(out_p, "rb+") as f: os.fsync(f.fileno())
        else:
            # Medium-lived cached artifact evicted after 16s
            out_p = os.path.join(d_mid, f"cached_{seq}")
            subprocess.run(["g++", obj_p, "-o", out_p], check=True)
            mid_q.append((now, out_p))
        # Immediately unlink short-lived temp source & object
        if os.path.exists(src_p): os.unlink(src_p)
        if os.path.exists(obj_p): os.unlink(obj_p)

        while mid_q and (now - mid_q[0][0]) >= 16.0:
            _, oldp = mid_q.pop(0)
            if os.path.exists(oldp): os.unlink(oldp)
        time.sleep(0.35)

ts = [threading.Thread(target=hot_header_reader), threading.Thread(target=big_source_scanner), threading.Thread(target=build_loop)]
for t in ts: t.start()
time.sleep(dur)
stop_evt.set()
for t in ts: t.join()
print(f"compiler_build completed ({dur}s)")
