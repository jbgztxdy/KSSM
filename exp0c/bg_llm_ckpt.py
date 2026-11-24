import os, sys, time, threading

ckpt_dir = sys.argv[1]
dur_s    = int(sys.argv[2]) if len(sys.argv) > 2 else 45
mode     = sys.argv[3] if len(sys.argv) > 3 else "mixed" # "llm_only" or "mixed"

os.makedirs(ckpt_dir, exist_ok=True)

model_dirs = [
    "/mnt/nvme12/paul/models/DeepSeek-R1-Distill-Qwen-14B",
    "/mnt/nvme12/paul/models/Meta-Llama-3-8B-Instruct",
]
st_files = []
for md in model_dirs:
    if os.path.isdir(md):
        for fn in sorted(os.listdir(md)):
            if fn.endswith(".safetensors"):
                st_files.append(os.path.join(md, fn))

# Pre-evict ONLY these .safetensors files before starting so every run starts from identical cold disk state!
for fp in st_files:
    fd = os.open(fp, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)

stop_evt = threading.Event()
llm_bytes = 0
ckpt_bytes = 0
llm_lock = threading.Lock()
ckpt_lock = threading.Lock()

def llm_loader():
    global llm_bytes
    idx = 0
    buf = bytearray(1024 * 1024) # 1 MB read buffer
    while not stop_evt.is_set() and st_files:
        fp = st_files[idx % len(st_files)]
        idx += 1
        fd = os.open(fp, os.O_RDONLY)
        try:
            while not stop_evt.is_set():
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                with llm_lock:
                    llm_bytes += len(chunk)
        finally:
            os.close(fd)

def ckpt_writer():
    global ckpt_bytes
    if mode != "mixed":
        return
    # Write 1 GiB checkpoint every 6 seconds (periodic realistic ML checkpointing)
    chunk = os.urandom(1024 * 1024) # 1 MB
    step = 0
    while not stop_evt.is_set():
        step += 1
        cp = os.path.join(ckpt_dir, f"step_{step:03d}.ckpt")
        t_w0 = time.time()
        with open(cp, "wb") as f:
            for _ in range(1024): # 1 GiB
                if stop_evt.is_set():
                    break
                f.write(chunk)
                with ckpt_lock:
                    ckpt_bytes += len(chunk)
            f.flush()
            os.fsync(f.fileno())
        # Keep only latest 2 checkpoints on disk to avoid wasting space
        if step > 2:
            oldp = os.path.join(ckpt_dir, f"step_{step - 2:03d}.ckpt")
            if os.path.exists(oldp):
                os.unlink(oldp)
        rem = 6.0 - (time.time() - t_w0)
        if rem > 0 and not stop_evt.is_set():
            stop_evt.wait(rem)

t0 = time.time()
t1 = threading.Thread(target=llm_loader)
t2 = threading.Thread(target=ckpt_writer)
t1.start(); t2.start()
time.sleep(dur_s)
stop_evt.set()
t1.join(); t2.join()
el = time.time() - t0

print(f"BG_RESULT elapsed={el:.2f} llm_gib={llm_bytes/(1024**3):.2f} llm_mbps={llm_bytes/(1024**2)/el:.1f} "
      f"ckpt_gib={ckpt_bytes/(1024**3):.2f} ckpt_mbps={ckpt_bytes/(1024**2)/el:.1f}")
