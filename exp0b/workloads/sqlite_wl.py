import os, sys, time, random, sqlite3, threading, shutil

base = sys.argv[1]
dur = int(sys.argv[2]) if len(sys.argv) > 2 else 65
os.makedirs(base, exist_ok=True)
dir_perm = os.path.join(base, "perm")
dir_rot  = os.path.join(base, "shards")
dir_tmp  = os.path.join(base, "tmp_tx")
for d in (dir_perm, dir_rot, dir_tmp):
    os.makedirs(d, exist_ok=True)

stop_evt = threading.Event()

# 1. Persistent primary DBs (DELETE journal mode -> creates/unlinks real short-lived -journal files < 1s!)
#    and also a large scan-table DB for full-table scans vs hot point queries
main_db = os.path.join(dir_perm, "main_oltp.db")
conn = sqlite3.connect(main_db)
conn.execute("PRAGMA journal_mode=DELETE;")
conn.execute("PRAGMA synchronous=NORMAL;")
conn.execute("CREATE TABLE IF NOT EXISTS kv (k INTEGER PRIMARY KEY, v BLOB);")
conn.executemany("INSERT OR REPLACE INTO kv VALUES (?, ?)", [(i, os.urandom(512)) for i in range(8000)])
conn.commit()
conn.close()

# Pre-create several persistent archive DBs
for i in range(16):
    p = os.path.join(dir_perm, f"archive_{i}.db")
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, payload BLOB);")
    c.executemany("INSERT INTO t VALUES (?, ?)", [(j, bytes([j & 255]) * 1024) for j in range(14000)])
    c.commit()
    c.close()

def tx_writer():
    c = sqlite3.connect(main_db, timeout=30)
    c.execute("PRAGMA journal_mode=DELETE;")
    c.execute("PRAGMA synchronous=FULL;")
    rng = random.Random(11)
    idx = 0
    while not stop_evt.is_set():
        # Each commit creates & deletes a real SQLite -journal file (< 100ms short-lived!)
        batch = [(rng.randint(0, 7999), bytes([(idx + j) & 255]) * 512) for j in range(25)]
        c.executemany("INSERT OR REPLACE INTO kv VALUES (?, ?)", batch)
        c.commit()
        idx += 1
        # Periodically create a short-lived temp export DB (deleted within 2-6s)
        if idx % 10 == 0:
            tp = os.path.join(dir_tmp, f"export_{idx}.db")
            tc = sqlite3.connect(tp)
            tc.execute("CREATE TABLE e (id INT, val BLOB);")
            tc.executemany("INSERT INTO e VALUES (?, ?)", [(j, b"X" * 1024) for j in range(400)])
            tc.commit()
            tc.close()
            # Read once then unlink shortly after
            with open(tp, "rb") as f:
                _ = f.read()
            time.sleep(0.3)
            if os.path.exists(tp):
                os.unlink(tp)
        time.sleep(0.08)
    c.close()

def shard_rotator():
    """Creates rolling time-window SQLite shards: medium-lived (15-28s) + occasional promoted long-lived shards"""
    active = []
    seq = 0
    while not stop_evt.is_set():
        seq += 1
        # Every 4th shard is promoted to permanent archive (long-lived) after reading main_db metadata
        is_perm = (seq % 4 == 0)
        if is_perm:
            # Realistic context: read main DB header/index before writing durable snapshot
            with open(main_db, "rb") as f:
                _ = f.read(131072)
            sp = os.path.join(dir_perm, f"snap_{seq}.db")
        else:
            sp = os.path.join(dir_rot, f"shard_{seq}.db")
        c = sqlite3.connect(sp)
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("CREATE TABLE s (id INTEGER PRIMARY KEY, val BLOB);")
        c.executemany("INSERT INTO s VALUES (?, ?)", [(j, bytes([seq & 255]) * 1024) for j in range(500)])
        c.commit()
        c.close()
        if not is_perm:
            active.append((time.time(), sp))
        # Evict shards older than 16s (medium-lived 10-50s!)
        now = time.time()
        while active and (now - active[0][0]) >= 16.0:
            _, oldp = active.pop(0)
            for ext in ("", "-wal", "-shm"):
                if os.path.exists(oldp + ext):
                    os.unlink(oldp + ext)
        time.sleep(1.5)

def query_worker():
    """Mixed hot point lookups (Hot Reusable) + full-table scans on archive DBs (One-Shot Scan)"""
    c = sqlite3.connect(main_db, timeout=30)
    rng = random.Random(99)
    step = 0
    while not stop_evt.is_set():
        # Hot point queries on keys 0..399
        for _ in range(40):
            k = rng.randint(0, 399)
            c.execute("SELECT length(v) FROM kv WHERE k=?", (k,)).fetchone()
        # Periodic full scan of a distinct archive DB
        if step % 12 == 0 and (step // 12) < 16:
            arch = os.path.join(dir_perm, f"archive_{(step // 6) % 6}.db")
            ac = sqlite3.connect(arch)
            ac.execute("SELECT COUNT(*), SUM(length(payload)) FROM t;").fetchone()
            ac.close()
        step += 1
        time.sleep(0.05)
    c.close()

threads = [
    threading.Thread(target=tx_writer),
    threading.Thread(target=shard_rotator),
    threading.Thread(target=query_worker),
]
for t in threads: t.start()
time.sleep(dur)
stop_evt.set()
for t in threads: t.join()
print(f"SQLite workload completed ({dur}s)")
