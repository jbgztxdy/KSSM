import os, sys, time, random, subprocess, threading

base = sys.argv[1]; dur = int(sys.argv[2]) if len(sys.argv) > 2 else 65
os.makedirs(base, exist_ok=True)
repo = os.path.join(base, "repo")
d_cold = os.path.join(base, "cold_packs")
d_tmp  = os.path.join(base, "tmp_patch")
d_mid  = os.path.join(base, "branch_snap")
d_rel  = os.path.join(base, "tag_archive")
for d in (repo, d_cold, d_tmp, d_mid, d_rel): os.makedirs(d, exist_ok=True)

subprocess.run(["git", "init", "-q", repo], check=True)
subprocess.run(["git", "-C", repo, "config", "user.email", "kssm@test.local"], check=True)
subprocess.run(["git", "-C", repo, "config", "user.name", "kssm"], check=True)
with open(os.path.join(repo, "README.md"), "w") as f: f.write("kssm git test\n" * 2000)
subprocess.run(["git", "-C", repo, "add", "."], check=True)
subprocess.run(["git", "-C", repo, "commit", "-q", "-m", "init"], check=True)

cold_packs = []
for i in range(14):
    p = os.path.join(d_cold, f"hist_pack_{i:02d}.pack")
    with open(p, "wb") as f: f.write(os.urandom(16 * 1024 * 1024))
    cold_packs.append(p)

stop_evt = threading.Event()

def hot_index_poller():
    idx_p = os.path.join(repo, ".git", "index")
    rdm_p = os.path.join(repo, "README.md")
    while not stop_evt.is_set():
        for p in (idx_p, rdm_p):
            if os.path.exists(p):
                with open(p, "rb") as f: _ = f.read(32768)
        time.sleep(0.03)

def cold_pack_scanner():
    for p in cold_packs:
        if stop_evt.is_set(): break
        with open(p, "rb") as f:
            while not stop_evt.is_set():
                if not f.read(524288): break
        time.sleep(2.8)

def git_worker():
    tmp_q, mid_q = [], []
    seq = 0
    buf = os.urandom(512 * 1024)
    while not stop_evt.is_set():
        seq += 1
        now = time.time()
        # Real git commit creates short-lived .git/index.lock & loose objects (< 2s)
        fp = os.path.join(repo, f"mod_{seq % 8}.txt")
        with open(fp, "w") as f: f.write(f"commit {seq} " + ("x" * 8192) + "\n")
        subprocess.run(["git", "-C", repo, "add", fp], check=True)
        subprocess.run(["git", "-C", repo, "commit", "-q", "-m", f"c{seq}"], check=True)
        if seq % 8 == 0:
            subprocess.run(["git", "-C", repo, "gc", "--quiet"], check=True)

        # Short-lived diff/patch scratch
        tp = os.path.join(d_tmp, f"diff_{seq}.patch")
        with open(tp, "wb") as f: f.write(buf)
        tmp_q.append((now, tp))

        # Medium-lived branch snapshot (16s TTL)
        if seq % 2 == 0:
            mp = os.path.join(d_mid, f"snap_{seq}.bundle")
            with open(mp, "wb") as f: f.write(buf)
            mid_q.append((now, mp))

        # Long-lived tag archive (reads README + fsync + rename)
        if seq % 4 == 0:
            with open(os.path.join(repo, "README.md"), "rb") as f: _ = f.read()
            rp_t = os.path.join(d_rel, f"tag_{seq}.tar.tmp")
            rp_f = os.path.join(d_rel, f"tag_{seq}.tar")
            with open(rp_t, "wb") as f:
                f.write(buf); f.flush(); os.fsync(f.fileno())
            os.rename(rp_t, rp_f)

        while tmp_q and (now - tmp_q[0][0]) >= 2.5:
            _, p = tmp_q.pop(0)
            if os.path.exists(p): os.unlink(p)
        while mid_q and (now - mid_q[0][0]) >= 16.0:
            _, p = mid_q.pop(0)
            if os.path.exists(p): os.unlink(p)
        time.sleep(0.5)

ts = [threading.Thread(target=hot_index_poller), threading.Thread(target=cold_pack_scanner), threading.Thread(target=git_worker)]
for t in ts: t.start()
time.sleep(dur)
stop_evt.set()
for t in ts: t.join()
print(f"git_vcs completed ({dur}s)")
