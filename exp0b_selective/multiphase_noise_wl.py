import os, sys, time, random

base = sys.argv[1]
mode = sys.argv[2] # "train" or "test"
dur  = int(sys.argv[3]) if len(sys.argv) > 3 else 55

os.makedirs(base, exist_ok=True)
dir_cold  = os.path.join(base, "cold_shards")
dir_hot   = os.path.join(base, "hot_idx")
dir_noise = os.path.join(base, "sys_noise")
dir_out   = os.path.join(base, "shared_out") # SAME output dir for ALL 3 lifetime classes!
dir_stg   = os.path.join(base, "staging")
for d in (dir_cold, dir_hot, dir_noise, dir_out, dir_stg):
    os.makedirs(d, exist_ok=True)

rng = random.Random(101 if mode == "train" else 2026)

# Pre-create cold large shards (16 MB each) for Phase A sequential scan
cold_files = []
for i in range(6):
    p = os.path.join(dir_cold, f"cold_{i}.dat")
    with open(p, "wb") as f: f.write(os.urandom(16 * 1024 * 1024))
    cold_files.append(p)

# Pre-create hot index files (256 KB each) for Phase B random point lookups
hot_files = []
for i in range(4):
    p = os.path.join(dir_hot, f"hot_{i}.idx")
    with open(p, "wb") as f: f.write(os.urandom(256 * 1024))
    hot_files.append(p)

# Pre-create realistic background noise files: append log, heartbeat config, metrics file
log_file = os.path.join(dir_noise, "worker.log")
hb_file  = os.path.join(dir_noise, "heartbeat.cfg")
met_file = os.path.join(dir_noise, "metrics.bin")
with open(log_file, "wb") as f: f.write(b"L" * 4096)
with open(hb_file,  "wb") as f: f.write(b"H" * 8192)
with open(met_file, "wb") as f: f.write(b"M" * 16384)

def emit_real_noise_io(k_noise):
    """Real interleaved OS I/O noise: small append logs, small config heartbeat reads,
    and small metrics overwrites on separate system noise files."""
    if k_noise <= 0:
        return
    f_log = open(log_file, "ab")
    f_hb  = open(hb_file, "rb")
    f_met = open(met_file, "r+b")
    buf_log = b"N" * 4096
    buf_met = b"Q" * 4096
    for j in range(k_noise):
        kind = j % 3
        if kind == 0:
            # Small O_APPEND log write (128KB chunk or 4KB -> write 132KB so collector emits 1 event per noise step!)
            f_log.write(buf_log * 33)
            f_log.flush()
        elif kind == 1:
            # Small config/heartbeat open+read
            f_hb.seek(0)
            _ = f_hb.read(4096)
            # Re-open heartbeat periodically to generate real OPEN_RO noise too
            if j % 6 == 1:
                with open(hb_file, "rb") as tf: _ = tf.read(4096)
        else:
            # Small metrics overwrite at offset 0
            f_met.seek(0)
            f_met.write(buf_met)
            f_met.flush()
    f_log.close(); f_hb.close(); f_met.close()

# Keep metadata log of (out_ino, true_phase_cls, k_noise) so we can break down accuracy BY exact K_noise!
meta_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces", f"mp_{mode}_meta.csv")
meta_f = open(meta_path, "w")
meta_f.write("ino,phase_cls,k_noise,birth_t\n")

payload = os.urandom(512 * 1024) # Uniform 512KB payload for ALL 3 classes!
short_q = []
med_q   = []

# Noise schedule:
#   Train: K in {0, 3, 8}
#   Test : K in {0, 5, 15, 35} (tests out-of-distribution high interleaved noise K=15, 35!)
k_choices = [0, 3, 8] if mode == "train" else [0, 5, 15, 35]

t0 = time.time()
seq = 0
while time.time() - t0 < dur:
    seq += 1
    phase = rng.choice([0, 1, 2]) # 0=short, 1=medium, 2=long
    k_noise = rng.choice(k_choices)

    # ---- Upstream Behavioral Phase (the real causal signal!) ----
    if phase == 2:
        # Phase A (Long-lived Archive): sequential scan of cold large shard (4 x 512KB = 2MB)
        cf = cold_files[seq % len(cold_files)]
        with open(cf, "rb") as f:
            f.seek(((seq * 2 * 1024 * 1024) % (12 * 1024 * 1024)))
            for _ in range(4):
                _ = f.read(524288)
    elif phase == 0:
        # Phase B (Short-lived Spill): random small lookups on hot index files
        hf = hot_files[seq % len(hot_files)]
        with open(hf, "rb") as f:
            for _ in range(6):
                f.seek(rng.randint(0, 55) * 4096)
                _ = f.read(8192)
    else:
        # Phase C (Medium-lived Checkpoint): staging write + fsync + rename barrier
        sp = os.path.join(dir_stg, f"stg_{seq}.tmp")
        sp2 = os.path.join(dir_stg, f"stg_{seq}.done")
        with open(sp, "wb") as f:
            f.write(b"S" * 131072)
            f.flush()
            os.fsync(f.fileno())
        os.rename(sp, sp2)
        os.unlink(sp2)

    # ---- Interleaved Real System Noise I/O (K steps between phase & file birth!) ----
    emit_real_noise_io(k_noise)

    # ---- Birth of Target File: IDENTICAL dir_out, IDENTICAL flags, IDENTICAL 512KB size, SAME thread! ----
    out_p = os.path.join(dir_out, f"obj_{seq:05d}.bin")
    with open(out_p, "wb") as f:
        ino = os.fstat(f.fileno()).st_ino
        meta_f.write(f"{ino},{phase},{k_noise},{time.time()-t0:.4f}\n")
        f.write(payload)

    now = time.time()
    if phase == 0:
        short_q.append((now, out_p))
    elif phase == 1:
        med_q.append((now, out_p))
    # phase == 2 stays alive permanently!

    # Reap short (<3s) and medium (15s) files
    while short_q and (now - short_q[0][0]) >= 2.5:
        _, p = short_q.pop(0)
        if os.path.exists(p): os.unlink(p)
    while med_q and (now - med_q[0][0]) >= 15.0:
        _, p = med_q.pop(0)
        if os.path.exists(p): os.unlink(p)

    # Cap log file size so disk stays tiny
    if os.path.getsize(log_file) > 32 * 1024 * 1024:
        with open(log_file, "wb") as f: f.write(b"L" * 4096)

meta_f.close()
print(f"Multiphase-noise [{mode}] completed: {seq} objects in {time.time()-t0:.1f}s")
